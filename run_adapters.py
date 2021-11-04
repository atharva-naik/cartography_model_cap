import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "20"
import sys
import argparse
from cartography_adapters import main


def get_args():
    '''get command line arguments'''
    parser = argparse.ArgumentParser(description="script to perform dataset cartography with adapters")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", 
                        help="the name/path of model to be used.")
    parser.add_argument("-ua", "--use_adapter", action='store_true', 
                        help="whether to use adapters or not")
    parser.add_argument("-a", "--adapter", 
                        type=str, default="multinli", 
                        help="name of adapter to be used")
    parser.add_argument("")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(**vars(args))
