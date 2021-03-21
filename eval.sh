API_KEY=$API_KEY WANDB_PROJECT=$WANDB_PROJECT python ray_tune.py \
  --model_name_or_path neuralmind/bert-large-portuguese-cased \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --train_file train_2.csv \
  --validation_file dev_2.csv \
  --max_seq_length 256 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --weight_decay 0.01 \
  --output_dir /home/params_large \
  --time_budget_s $TIME_BUDGET