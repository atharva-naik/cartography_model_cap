import os
# comment this out except for KGP servers.
# os.environ['OPENBLAS_NUM_THREADS'] = "20"
import sys
from cartography_adapters import get_cli_args, pprint_args, TrainingDynamics


def main():
    adapter=False
    notebook=False
    # ge commandline arguments
    cli_args = get_cli_args()
    # print arguments.
    pprint_args(cli_args)
    # if not notebook:
    #     td = TrainingDynamics("roberta", "roberta-base", "../roberta-base-tok")
    # else:
    if adapter: print("\x1b[32;1musing adapters\x1b[0m")
    td = TrainingDynamics(
        "roberta", "roberta-base", 
        "../roberta-base-tok", use_adapter=adapter
    )
    td.train(
        "./data/MNLI/original/multinli_1.0_train.jsonl", 
        "./data/MNLI/original/multinli_1.0_dev_matched.jsonl"
    )
#     if notebook:
#         td.train(
#             "/content/drive/MyDrive/SDM/data/MNLI/original/multinli_1.0_train.jsonl", 
#             "/content/drive/MyDrive/SDM/data/MNLI/original/multinli_1.0_dev_matched.jsonl"
#         )
#     else:
#         td.train(
#             "./data/MNLI/original/multinli_1.0_train.jsonl", 
#             "./data/MNLI/original/multinli_1.0_dev_matched.jsonl"
#         )
    # hello_world(**vars(cli_args))
    
if __name__ == "__main__":
    main()