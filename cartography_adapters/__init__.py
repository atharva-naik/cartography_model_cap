# code for tracking training dynamics for transformers+adapters.
import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "12"

import json
import torch
import numpy as np
import transformers
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertModel,
    AutoConfig,
    BertConfig,
    RobertaModel,
    RobertaConfig,
    BertTokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
try:
    # utilites for getting/printing args etc..
    from cartography_adapters.utils import *
    from cartography_adapters.models import *
    from cartography_adapters.datautils import GLUEDataset
except ImportError:
    from utils import *
    from models import *
    from datautils import GLUEDataset
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
TrainConfig.lr = 1e-4 # 1.099e-5 # initial learning rate.
TrainConfig.eps = 1e-8 # Adam epsilon.
TrainConfig.seed = 2021 # random seed to be used.
TrainConfig.num_epochs = 24 # max number of epochs.
TrainConfig.patience = 3
TrainConfig.batch_size = 96 # training batch size.
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
TrainConfig.save_as = "roberta-base-mnli-adapter-multinli.pt" # name to be given to the saved model.
    
    
class TrainingDynamics:
    def __init__(self, model_type="roberta", model_name_or_path: str="roberta-base",
                 tokenizer_name_or_path: str="../roberta-base-tok", **trainer_config):
        self.trainer_config = TrainConfig
        self.trainer_config.update(**trainer_config)
        pprint_args(vars(self.trainer_config))
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
        self.gpu_mem_manager = GPUMemoryManager(self.trainer_config.device)
        self.best_epoch = 0
        self.current_epoch = 0
    
    def train(self, path: Union[str, Path], eval_path: Union[str, Path, None]=None):
        # create datasets and dataloaders.
        set_seed(self.trainer_config.seed)
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
        self.best_eval_acc = 0
        self.train_loss = 0
        
        self.model.zero_grad()
        train_dy_dir = os.path.join(
            self.trainer_config.train_dy_dir,
            "training_dynamics"
        )
        save_as = os.path.join(
            self.trainer_config.train_dy_dir,
            self.trainer_config.save_as
        )
        os.makedirs(self.trainer_config.train_dy_dir, exist_ok=True)
        for i in range(self.trainer_config.num_epochs):
            epoch_log_path = os.path.join(
                self.trainer_config.train_dy_dir,
                f"epoch_{i}.log"
            )
            with open(epoch_log_path, "w") as f: pass
            file_path = os.path.join(
                train_dy_dir, 
                f"dynamics_epoch_{i}.jsonl"
            )
            with open(file_path, "w") as f: pass
            self.best_epoch = i
            self.current_epoch = i
            batch_iterator = tqdm(
                enumerate(self.trainloader),
                total=len(self.trainloader),
                desc="",
            )
            epoch_log = {}
            for step, batch in batch_iterator:
                desc = f"{self.model_type}:{i}/{self.trainer_config.num_epochs} loss:{self.train_loss/(step+1e-12):.3f} acc:{(100*self.train_acc/(self.trainer_config.batch_size*step+1e-12)):.2f}"
                batch_iterator.set_description(desc)              
                self.model.train()
                # move inputs to device.
                batch["input_ids"] = batch["input_ids"].to(self.trainer_config.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.trainer_config.device)
                if self.model_type == "bert":
                    batch["token_type_ids"] = batch["token_type_ids"].to(self.trainer_config.device)
                batch["label"] = batch["label"].to(self.trainer_config.device)
                # forward step.
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    labels=batch["label"]
                )
                loss = output["loss"]
                logits = output["logits"]
                logits = logits.cpu().detach().tolist()
                loss.backward()
                if self.trainer_config.grad_accumulation_steps > 1:
                    loss = loss / self.trainer_config.grad_accumulation_steps
                # move ids and labels to cpu and convert to list, for logging td.
                ids = batch["id"].cpu().detach().tolist()
                labels = batch["label"].cpu().detach().tolist()
                for id, logit, label in zip(ids, logits, labels):
                    # print(label, id)
                    logits_dict = {
                        "guid": id,
                        f"logits_epoch_{i}": logit,
                        "gold": label 
                    }
                    pred = np.argmax(logit)
                    if pred == label: self.train_acc += 1
                    with open(file_path, "a") as f:
                        f.write(json.dumps(logits_dict)+"\n")
                if (step + 1) % self.trainer_config.grad_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                
                loss = loss.cpu().detach()
                self.train_loss += loss.item()
                epoch_log["loss"] = f"{(self.train_loss/(step+1)):.5f}"
                epoch_log["free_memory"] = self.gpu_mem_manager.free()
                epoch_log["learning_rate"] = self.scheduler.get_last_lr()[0]
                # FOR DEBUGGING
                # if step == 50: break
                with open(epoch_log_path, "a") as f:
                    f.write(json.dumps(epoch_log)+"\n")
                
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
            self.train_loss /= len(self.trainloader)
            self.train_acc /= len(self.trainloader)
            if eval_path:
                eval_acc, _ = self.evaluate(eval_path)
                if eval_acc > self.best_eval_acc:
                    self.best_eval_acc = eval_acc
                    self.best_epoch = self.current_epoch
                    print(f"saving model with best_acc={self.best_eval_acc}, best_epoch={self.best_epoch} at {save_as}")
                    # save model with best eval accuracy.
                    self.save(save_as)
            # patience condition from original repo: stop training if performance doesn't improve after `patience` no. of epochs.
            if self.current_epoch - self.best_epoch >= self.trainer_config.patience:
                break
            self.train_acc = 0
            self.train_loss = 0
        
    def evaluate(self, path: Union[str, Path]) -> dict:
        '''run eval and return dictionary of metrics (acc. only for MNLI.)'''
        path = str(path)
        self.eval_acc = 0
        self.eval_loss = 0
        
        predictions = []
        if not hasattr(self, "evalloader"):
            print("loading eval data")
            self.evalset = GLUEDataset(
                path=path, tokenizer=self.tokenizer, 
                task_name=self.trainer_config.task_name,
            )
            self.evalloader = DataLoader(
                self.evalset, batch_size=self.trainer_config.batch_size,  
                shuffle=False, num_workers=self.trainer_config.num_workers
            )
        batch_iterator = tqdm(
            enumerate(self.evalloader),
            total=len(self.evalloader),
            desc="",
        )
        for step, batch in batch_iterator:
            self.model.eval()
            desc = f"{self.model_type}: loss:{self.eval_loss/(step+1e-12):.3f} acc:{(100*self.eval_acc/(self.trainer_config.batch_size*step+1e-12)):.2f}"
            batch_iterator.set_description(desc)   
            
            with torch.no_grad():
                # move inputs to device.
                batch["input_ids"] = batch["input_ids"].to(self.trainer_config.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.trainer_config.device)
                if self.model_type == "bert":
                    batch["token_type_ids"] = batch["token_type_ids"].to(self.trainer_config.device)
                batch["label"] = batch["label"].to(self.trainer_config.device)
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    labels=batch["label"]
                )
            loss = output["loss"]
            logits = output["logits"]
            # accumulate eval loss.
            loss = loss.cpu().detach()
            self.eval_loss += loss.item()
            # move ids, labels and logits to cpu, to save predictions.
            ids = batch["id"].cpu().detach().tolist()
            logits = logits.cpu().detach().tolist()
            labels = batch["label"].cpu().detach().tolist()
            for id, logit, label in zip(ids, logits, labels):
                pred = np.argmax(logit)
                predictions.append({
                    "guid": id,
                    "pred": pred,
                    "gold": label,
                    f"logits_epoch_{self.current_epoch}": logit,
                })
                if pred == label: self.eval_acc += 1
        self.eval_loss /= len(self.evalloader)
        self.eval_acc /= len(self.evalloader)

        return self.eval_acc, predictions
    
    def save(self, ckpt_path: Union[str, Path]):
        ckpt_dict = {
            "epoch": self.current_epoch,
            "metrics": {
                "train_loss": self.train_loss,
                "train_acc": self.train_acc,
                "val_loss": self.eval_loss,
                "val_acc": self.eval_acc,
            },
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