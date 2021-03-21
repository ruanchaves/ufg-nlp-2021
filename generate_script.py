import itertools
import os


def main():

    warmup_steps = [50, 100, 500]
    lrs = [5e-5, 4e-5, 3e-5, 2e-5]
    half_batch_size = [8, 16]

    task_folder = './'
    project = 'WANDB_PROJECT=ufg_comp'
    params = []
    for item in itertools.product(warmup_steps, lrs, half_batch_size):
        row = {
            'warmup_steps': item[0],
            'learning_rate': item[1],
            'half_batch_size': item[2]
        }
        params.append(row)

    cmd_list = []
    cmd_string = """
    python3 run_glue.py \
    --model_name_or_path microsoft/deberta-large \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --train_file {0} \
    --validation_file {1} \
    --max_seq_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps {2} \
    --learning_rate {3} \
    --num_train_epochs 2 \
    --warmup_steps {4} \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --output_dir {5} \
    --overwrite_output_dir
    """

    folders = [os.path.join(task_folder, str(i)) for i in range(1)]
    for f in folders:
        train_path = os.path.join(f, 'train_2.csv')
        dev_path = os.path.join(f, 'dev_2.csv')
        for p in params:
            output_dir = 'cv_model_' + \
                str(p['half_batch_size']) + \
                '_' + str(p['learning_rate']) + \
                '_' + str(p['warmup_steps'])
            output_dir = os.path.join(f, output_dir)
            cmd = cmd_string.format(
                train_path,
                dev_path,
                p['half_batch_size'],
                p['learning_rate'],
                p['warmup_steps'],
                output_dir
            )
            cmd_list.append(cmd)

    cmd_list = [project + ' ' + x.strip() for x in cmd_list]
    final_cmd = '\n'.join(cmd_list)
    with open('train.sh', 'w+') as f:
        print(final_cmd, file=f)


if __name__ == '__main__':
    main()
