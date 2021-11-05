# code for tracking training dynamics for transformers+adapters.
import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "12"
try:
    # utilites for getting/printing args etc..
    from cartography_adapters.utils import *
    from cartography_adapters.models import *
except ImportError:
    from utils import *
    from models import *
    
import torch
import transformers
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from transformers import (
    AdamW,
    BertModel,
    AutoConfig,
    BertConfig,
    RobertaModel,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
# set logging level for transformers
transformers.logging.set_verbosity_error()
# the default trainer config.
class TrainerNamespace(argparse.Namespace):
    def __init__(self):
        super(TrainerNamespace, self).__init__()
    
    def update(self, **kwargs):
        '''update and (or) append arguments using kwargs/dict.'''
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        
    def _serialize(self):
        '''serialize object attributes to dictionary.'''
        return vars(self)      
#     def save(self, path: Union[str, Path]):
#         '''save training arguments.'''
#         path = str(path)
#         args_dict = self._serialize()
#         with open(path, "w") as f:
#             json.dump(args_dict, f, indent=4)
TrainConfig = TrainerNamespace()
TrainConfig.lr = 1.099e-5 # initial learning rate.
TrainConfig.eps = 1e-8 # Adam epsilon.
TrainConfig.seed = 2021 # random seed to be used.
TrainConfig.num_epochs = 24 # max number of epochs.
TrainConfig.patience = 3
TrainConfig.batch_size = 32 # training batch size.
TrainConfig.num_classes = 3 # number of classes in target task.
TrainConfig.num_workers = 4 # number of threads for dataloading.
TrainConfig.device = "cuda:0" # device for training.
TrainConfig.weight_decay = 0 # weight decay for params.
TrainConfig.task_name = "MNLI" 
TrainConfig.warmup_steps = 0
TrainConfig.grad_accumulation_steps = 1 
TrainConfig.target_metric = "acc" # target metric is used for model saving.
# directory for storing training dynamics.
TrainConfig.train_dy_dir = "rob_base_mnli_adapter_multinli" 
    
    
class TrainingDynamics:
    def __init__(self, model_type="roberta", model_name_or_path: str="roberta-base",
                 tokenizer_name_or_path: str="../roberta-base-tok", **trainer_config):
        self.trainer_config = TrainConfig
        self.trainer_config.update(**trainer_config)
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.trainer_config.num_classes,            
        )
        if model_type == "roberta":
            self.model = RobertaForCartography(model_config)
        elif model_type == "bert":
            self.model = BertForCartography(model_config)
        else:
            raise TypeError("Model type not implemented")
        self.model_config = model_config
        self.model_type = model_type
        # accumulate all arguments.
        self.trainer_config.update(**vars(model_config))
        # create optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.trainer_config.lr,
            eps=self.trainer_config.eps,
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(
            tokenizer_name_or_path
        )
        self.model.to(self.trainer_config.device)
    
    def train(self, path: Union[str, Path]):
        # create datasets and dataloaders.
        path = str(path)
        self.trainset = GLUEDataset(
            path=path, tokenizer=self.tokenizer, 
            task_name=self.trainer_config.task_name,
        )
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.trainer_config.batch_size,  
            shuffle=True, num_workers=self.trainer_config.num_workers
        )
        # create scheduler.
        total_steps = len(self.trainloader) // self.trainer_config.grad_accumulation_steps * self.trainer_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_training_steps=total_steps,
            num_warmup_steps=self.trainer_config.warmup_steps
        )
        self.train_acc = 0
        self.best_epoch = 0
        self.current_epoch = 0
        self.model.zero_grad()
        self.trainer_config.train_dy_dir = os.path.join(
            self.trainer_config.train_dy_dir,
            "training_dynamics"
        )
        os.makedirs(self.trainer_config.train_dy_dir, exist_ok=True)
        for i in range(self.trainer_config.num_epochs):
            file_path = os.path.join(
                self.trainer_config.train_dy_dir, 
                f"dynamics_epoch_{i}.jsonl"
            )
            train_dy_file_ptr = open(file_path, "w")
            self.best_epoch = i
            self.current_epoch = i
            batch_iterator = tqdm(
                enumerate(self.trainloader),
                total=len(self.trainloader),
                desc="",
            )
            for step, batch in batch_iterator:
                desc = f"{self.model_type}-{self.trainer_config.task_name}-train: {i}/{self.trainer_config.num_epochs} {step}/{len(self.trainloader)}"
                batch_iterator.set_description(desc)              
                self.model.train()
                # move inputs to device.
                batch["input_ids"] = batch["input_ids"].to(self.trainer_config.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.trainer_config.device)
                if self.model_type == "bert":
                    batch["token_type_ids"] = batch["token_type_ids"].to(self.trainer_config.device)
                # forward step.
                loss, logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    labels=batch["label"]
                )
                logits = logits.detach().cpu().tolist()
                if self.trainer_config.grad_accumulation_steps > 1:
                    loss = loss / self.trainer_config.grad_accumulation_steps
                for id, logit, label in zip(batch["ids"], logits, batch["label"])
                    logits_dict = {
                        "id": id,
                        f"logits_epoch_{i}": logit,
                        "gold": label 
                    }
                    train_dy_file_ptr.write(
                        json.dumps(logits_dict)+"\n"
                    )
                if (step + 1) % self.trainer_config.grad_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
            # patience condition from original repo: stop training if performance doesn't improve after `patience` no. of epochs.
            if self.current_epoch - self.best_epoch >= self.trainer_config.patience:
                break
            train_dy_file_ptr.close()
        
    def evaluate(self) -> dict:
        '''run eval and return dictionary of metrics (acc. only for MNLI.)'''
        pass
    
    def save(self, ckpt_path: Union[str, Path]):
        ckpt_dir = str(ckpt_dir)
        metrics = self.evaluate()
        ckpt_dict = {
            "epoch": self.current_epoch,
            "metrics": metrics,
            "args": self.trainer_config._serialize(),
            "state_dict": self.model.state_dict(),
        }
        torch.save(ckpt_dict, ckpt_path)
    
    
def hello_world(**args):
    from torch.utils.data import DataLoader
    from transformers import RobertaTokenizer
    try:
        from cartography_adapters.datautils import GLUEDataset
    except ImportError:
        from .datautils import GLUEDataset
    # number of worker threads to be used for dataloading.
    num_workers = args.get("num_workers", 1)
    # batch size for training
    batch_size = args.get("batch_size", 32)
    # config = RobertaConfig()
    trainset = GLUEDataset(
        path="./data/MNLI/original/multinli_1.0_train.jsonl",
        tokenizer=RobertaTokenizer.from_pretrained("../roberta-base-tok"), 
        task_name="MNLI",
    )
    trainloader = DataLoader(trainset, num_workers=num_workers, 
                             shuffle=True, batch_size=batch_size)
    # iterate over trainloader.
    for step, batch in enumerate(trainloader):
        print("id:", batch["id"])
        print("label:", batch["label"])
        print("input_ids:", batch["input_ids"])
        print("attention_mask:", batch["attention_mask"])
        break
    trainer = TrainingDynamics("roberta", "roberta-base")
    trainer.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    
    
if __name__ == "__main__":
    main()