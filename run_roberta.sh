# python -m cartography.selection.train_dy_filtering --plot --task_name SNLI --model_dir output --model roberta-base
# (choose from 'SNLI', 'MNLI', 'QNLI', 'WINOGRANDE')
python -m cartography.classification.run_glue_v2 -c configs/mnli.json --do_train --do_eval -o output