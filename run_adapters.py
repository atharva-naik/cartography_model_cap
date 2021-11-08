import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "20"
import sys
from cartography_adapters import get_cli_args, pprint_args, TrainingDynamics


def main():
    notebook=False
    # ge commandline arguments
    cli_args = get_cli_args()
    # print arguments.
    pprint_args(cli_args)
    # if not notebook:
    #     td = TrainingDynamics("roberta", "roberta-base", "../roberta-base-tok")
    # else:
    td = TrainingDynamics("roberta", "roberta-base", "../roberta-base-tok")
    if notebook:
        td.train(
            "/content/drive/MyDrive/SDM/data/MNLI/original/multinli_1.0_train.jsonl", 
            "/content/drive/MyDrive/SDM/data/MNLI/original/multinli_1.0_dev_matched.jsonl"
        )
    else:
        td.train(
            "./data/MNLI/original/multinli_1.0_train.jsonl", 
            "./data/MNLI/original/multinli_1.0_dev_matched.jsonl"
        )
    # hello_world(**vars(cli_args))
    
if __name__ == "__main__":
    main()