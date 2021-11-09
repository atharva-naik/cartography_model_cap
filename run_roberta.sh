# python -m cartography.selection.train_dy_filtering --plot --task_name SNLI --model_dir output --model roberta-base
# (choose from 'SNLI', 'MNLI', 'QNLI', 'WINOGRANDE')
# python -m cartography.classification.run_glue_v2 -c configs/mnli.json --do_test -o /content/drive/MyDrive/SDM/rob_large_MNLI

# simple roberta-large
# python -m cartography.classification.run_glue_v2 -c configs/mnli-rob-large.json --do_train --do_eval -o rob_large_MNLI

# simple roberta-base
# echo -e "\x1b[34m simple roberta-base -o rob_base_MNLI \x1b[0m"
# python -m cartography.classification.run_glue_v2 -c configs/mnli-rob-base.json --do_train --do_eval -o rob_base_MNLI

# roberta-large with adapters.
# python -m cartography.classification.run_glue_v2 -c configs/mnli-rob-large.json --do_train --do_eval -o rob_large_adapter_MNLI

# roberta-base with adapters.
echo -e "\x1b[34m roberta-base with adapter -o rob_base_adapter_MNLI \x1b[0m"
python -m cartography.classification.run_glue_v2 -c configs/mnli-rob-base.json --do_train --do_eval --use_adapter -o rob_base_adapter_MNLI

# python -m cartography.classification.run_glue_v2 -c configs/mnli.json --do_test -o roberta-MNLI-adapter-MNLI