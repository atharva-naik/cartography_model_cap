import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "20"
import sys
from cartography_adapters import hello_world, get_cli_args, pprint_args


def main():
    # ge commandline arguments
    cli_args = get_cli_args()
    # print arguments.
    pprint_args(cli_args)
    hello_world(**vars(cli_args))

    
if __name__ == "__main__":
    main()