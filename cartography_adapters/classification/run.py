import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm, trange
import os, glob, json, logging, random
# from torch.utils.data.distributed import DistributedSampler # need for multi-GPU code.
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
MODEL_CLASSES = {
    "bert": (
        BertConfig, 
        AdaptedBertForSequenceClassification, 
        BertTokenizer
    ),
    "roberta": (
        RobertaConfig, 
        AdaptedRobertaForSequenceClassification, 
        RobertaTokenizer
    ),
}

class Args:
    '''
    All configuration required for running any dataset using transformers+adapters.
    '''
    def __init__(self, MODEL_CLASSES, processors, configs):
        # Required parameters
        # Directory where task data resides.
        self.data_dir: str = configs["data_dir"]
        # Input data
        self.train: str = configs.get("train", None)
        self.dev: str = configs.get("dev", None)
        self.test: str = configs.get("test", None)
        # One of 'bert', 'roberta', etc.
        self.model_type: str = configs["model_type"]
        assert self.model_type in MODEL_CLASSES.keys()
        # Path to pre-trained model or shortcut name from `ALL_MODELS`.
        self.model_name_or_path: str = configs["model_name_or_path"]
        assert self.model_name_or_path in ("bert-base-uncased", "bert-base-cased", 
                                           "bert-large-cased", "bert-large-uncased", 
                                           "roberta-large", "roberta-base")
        self.task_name: str = configs["task_name"] # The name of the task to train.
        assert self.task_name.lower() in processors.keys()
        self.seed: int = configs["seed"] # Random seed for initialization.
        self.output_dir: str = configs["output_dir"] # store checkpoints and model preds here.
        self.do_train: bool = configs.get("do_train", False) # run training?
        self.do_eval: bool = configs.get("do_eval", False) # run eval on dev set?
        self.do_test: bool = configs.get("do_test", False) # run test?
        # Pretrained config name or path if not the same as `model_name`.
        self.config_name: str = configs.get("config_name", "")
        # Pretrained tokenizer name or path if not the same as `model_name`.
        self.tokenizer_name: str = configs.get("tokenizer_name", "")
        # Where to store the pre-trained models downloaded from s3:// location.
        self.cache_dir: str = configs.get("cache_dir", "")
        # Where to store the feature cache for the model.
        self.features_cache_dir: str = configs.get("features_cache_dir", os.path.join(self.data_dir, f"cache_{self.seed}"))
        # The maximum total input sequence length after tokenization.
        # Sequences longer than this will be truncated,
        # sequences shorter will be padded.
        self.max_seq_length: int = configs.get("max_seq_length", 128)
        # Run evaluation during training after each epoch.
        self.evaluate_during_training: bool = configs.get(
            "evaluate_during_training", True)
        # Run evaluation during training at each logging step.
        self.evaluate_during_training_epoch: bool = configs.get("evaluate_during_training_epoch", False)
        # Set this flag if you are using an uncased model.
        self.do_lower_case: bool = configs.get("do_lower_case", True)
        # Batch size per GPU/CPU for training.
        self.per_gpu_train_batch_size: int = configs.get("per_gpu_train_batch_size", 96)
        # Number of updates steps to accumulate before
        # performing a backward/update pass.
        self.gradient_accumulation_steps: int = configs.get("gradient_accumulation_steps", 1)
        # The initial learning rate for Adam.
        self.learning_rate: float = configs.get("learning_rate", 1e-5)
        self.weight_decay: float = configs.get("weight_decay", 0.0) # weight decay.
        self.adam_epsilon: float = configs.get("adam_epsilon", 1e-8) # epsilon for Adam.
        self.max_grad_norm: float = configs.get("max_grad_norm", 1.0) # Max gradient norm.
        self.num_train_epochs: float = configs.get("num_train_epochs", 3.0) # no. of train epochs.
        # If > 0 : set total number of training steps to perform.
        # Override num_train_epochs.
        self.max_steps: int = configs.get("max_steps", -1)
        self.warmup_steps: int = configs.get("warmup_steps", 0) # Linear warmup over warmup_steps.
        self.logging_steps: int = configs.get("logging_steps", 1000) # Log every X updates steps.
        self.patience: int = configs.get("patience", 5) # early stopping: stop training if performance doesn't improve in `patience` no. of updates.
        self.save_steps: int = configs.get("save_steps", 0) # save checkpoint after every `save_steps` no. of steps.
        self.eval_all_checkpoints: bool = configs.get("eval_all_checkpoints", False) # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number
        self.no_cuda: bool = configs.get("no_cuda", False) # Avoid using CUDA when available
        self.overwrite_output_dir: bool = configs.get("overwrite_output_dir", False) # overwrite output dir.
        self.overwrite_cache: bool = configs.get("overwrite_cache", False) # overwrite cached features for train/dev set.
        # Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
        # self.fp16: bool = configs.get("fp16", False)
        # For fp16 : Apex AMP optimization level selected in # ['O0', 'O1', 'O2', and 'O3'].
        # See details at https://nvidia.github.io/apex/amp.html"
        # self.fp16_opt_level: str = configs.get("fp16_opt_level", "01")
def set_seed(args):
    '''function to set seeds.'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed) 

def get_cli_args():
    '''get commandline arguments'''
    parser = argparse.ArgumentParser()
    # TODO: better alternative to config files.
    parser.add_argument("--config", "-c", required=True, help="Main config file with basic arguments.")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for model.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the (OOD) test set.")
    parser.add_argument("--test", type=os.path.abspath, help="OOD test set.")
    # TODO(SS): Automatically map tasks to OOD test sets.
    config_args = {} # args from config file.
    args_from_cli = parser.parse_args()
    if args_from_cli.config.endswith("json"):
        print("loading CONFIG from JSON")
        config_args = json.load(open(args_from_cli.config))
    elif args_from_cli.config.endswith("jsonnet"):
        print("loading CONFIG from JSONNET")
        config_args = json.loads(_jsonnet.evaluate_file(args_from_cli.config))
    config_args.update(**vars(args_from_cli))
    args = Args(MODEL_CLASSES, processors, config_args)

    return args
    
def main():
    args = get_cli_args()
        
    
if __init__ == "__main__":
    main()