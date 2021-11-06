import os
import argparse
from pprint import pprint
from typing import Union, List, Dict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_cli_args() -> argparse.Namespace:
    '''get command line arguments'''
    parser = argparse.ArgumentParser(description="script to perform dataset cartography with adapters")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", 
                        help="the name/path of model to be used.")
    parser.add_argument("-ua", "--use_adapter", action='store_true', 
                        help="whether to use adapters or not")
    parser.add_argument("-a", "--adapter", 
                        type=str, default="multinli", 
                        help="name of adapter to be used")
    parser.add_argument("-nw", "--num_workers", type=int, default=1, 
                        help="number of workers to be used by DataLoader")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, 
                        help="batch size for traning (and eval).")
    args = parser.parse_args()

    return args

def set_seed(seed):
    '''seed random, numpy and torch'''
    import torch
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if args.n_gpu > 0:
def pprint_args(args: Union[dict, argparse.Namespace], 
                max_key_col_width: int=40, 
                max_val_col_width: int=40) -> None:
    '''pretty print arguments'''
    cols, rows = os.get_terminal_size()
    if isinstance(args, dict):
        args_dict = args
    elif isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    else:
        raise TypeError("type(args) must be in `dict`, `argparse.Namespace`")
    # margin formatting.
    key_col_width = max(len(key) for key in args_dict)
    key_col_width = min(key_col_width, max_key_col_width-1)
    val_col_width = max(len(str(val)) for val in args_dict.values())
    val_col_width = min(val_col_width, max_val_col_width-1)
    def trunc_key(key, width):
        key = str(key)
        if len(key) > width: 
            key = key[:width-3]+"..."
        return key.ljust(width)
    def trunc_val(val, width):
        val = str(val)
        if len(val) > width: 
            val = val[:width-3]+"..."
        return val.rjust(width)
    # print args.
    print("—"*cols)
    for key, val in args_dict.items():
        key = trunc_key(key, key_col_width)
        val = trunc_val(val, val_col_width)
        print(key, ":", val)
    print("—"*cols)
# def get_free_mem():
#     import torch
#     t = torch.cuda.get_device_properities(0).total_memory
#     r = torch.cuda.memory_reserved(0)
#     a = torch.cuda.memory_allocated(0)
#     return r-a
# print("loading GPU mem manager")
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    
    return f"{num:.1f}Yi{suffix}"

class GPUMemoryManager:
    '''getting driver not loaded error here.'''
    def __init__(self, index: Union[str, int]=0):
        nvmlInit()
        if isinstance(index, str):
            index = index.replace("cuda","").replace(":","")
            if index  == "": index = 0
            index = int(index)
        self.handle = nvmlDeviceGetHandleByIndex(index)
        
    def info(self, part='total', human_readable=True):
        nvmlInit()
        info = nvmlDeviceGetMemoryInfo(self.handle)
        mem_usage = getattr(info, part)
        if human_readable:
            return sizeof_fmt(mem_usage)
        return mem_usage
    
    def total(self):
        return self.info("total")
    
    def free(self):
        return self.info('free')
        
    def used(self):
        return self.info('used')
# print("loaded GPU mem manager")