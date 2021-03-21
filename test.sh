while true
do
  seed=$(shuf -i1-100 -n1)
  if [[ ! -d ./submission_large_$seed ]]
  then
    python run_glue.py \
      --model_name_or_path neuralmind/bert-large-portuguese-cased \
      --do_train \
      --do_eval \
      --seed $seed \
      --evaluation_strategy steps \
      --eval_steps 500 \
      --save_strategy no \
      --train_file train_total.csv \
      --validation_file test_final.csv \
      --max_seq_length 256 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --learning_rate 5e-05 \
      --num_train_epochs 7.5 \
      --weight_decay 0.01 \
      --output_dir ./tmp \
      --overwrite_output_dir \
      --preds_dir ./submission_large_$seed
  fi
done