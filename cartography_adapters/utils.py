import os
import argparse
from pprint import pprint
from typing import Union, List, Dict


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