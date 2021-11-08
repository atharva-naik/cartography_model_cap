python run_glue.py \
  --model_name_or_path roberta-base \
  --dataset_name mnli \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir /rob_base_mnli_glue