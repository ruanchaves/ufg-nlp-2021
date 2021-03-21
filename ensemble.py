import os
import argparse
import pandas as pd
import numpy as np
import time
from collections import Counter


def save_preds(source, dataset, output):
    df = pd.read_csv(dataset)
    ids = df['Id'].values.tolist()
    preds = ['Id,Category']
    label_dict = {
        0: 'Negativo',
        1: 'Neutro',
        2: 'Positivo'
    }
    a = np.argmax(source, axis=1)
    for idx, item in enumerate(a):
        row = '{0},{1}'.format(ids[idx], label_dict[item])
        preds.append(row)
    preds = '\n'.join(preds)
    with open(output, 'w+') as f:
        print(preds, file=f)


def get_preds(root_dir, submission_prefix, preds_prefix):
    output = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            directory = os.path.join(root, name)
            if submission_prefix in directory:
                preds = os.listdir(directory)
                preds = [x for x in preds if x.startswith(preds_prefix)]
                if preds:
                    preds = sorted(preds)[-1]
                    preds = os.path.join(directory, preds)
                    output.append(preds)
    return output


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--folder',
        type=str,
        default='.'
    )

    parser.add_argument(
        '--submission_prefix',
        type=str,
        default='submission_large'
    )

    parser.add_argument(
        '--preds_prefix',
        type=str,
        default='preds'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='test.csv'
    )

    parser.add_argument(
        '--output_prefix',
        type=str,
        default='submission_best_'
    )

    parser.add_argument(
        '--ensemble_prefix',
        type=str,
        default='ensemble_'
    )

    args = parser.parse_args()
    return args


def build_ensemble(folder, ensemble_prefix, submission_prefix):

    def most_frequent(iterable):
        occurence_count = Counter(iterable)
        return occurence_count.most_common(1)[0][0]

    files = [x for x in os.listdir(folder)]
    content = []
    for filename in files:
        if filename.startswith(submission_prefix) and filename.endswith('.csv'):
            filename_path = os.path.join(folder, filename)
            print('reading {0}'.format(filename))
            with open(filename_path, 'r') as f:
                text = f.read().split('\n')
                content.append(text)

    output = ['Id,Category']

    for i in range(len(content[0])):
        if i:
            cands = [x[i] for x in content]
            cands = [x.split(',') for x in cands]
            row_idx = cands[0][0]
            cand_labels = [x[1].strip() for x in cands]
            top_label = most_frequent(cand_labels)
            new_row = '{0},{1}'.format(row_idx, top_label)
            output.append(new_row)

    output = '\n'.join(output)
    timestamp = str(int(time.time()))
    destination = ensemble_prefix + timestamp + '.csv'
    with open(destination, 'w+') as f:
        print(output, file=f)


def main():
    args = get_args()
    preds = get_preds(args.folder, args.submission_prefix, args.preds_prefix)
    pred_arrays = [np.load(x) for x in preds]

    for idx, array in enumerate(pred_arrays):
        filename = str(preds[idx])
        number = ''.join([x for x in filename if x.isdigit()])
        output_file = args.submission_prefix + '_' + number + '.csv'
        print(filename, output_file)
        save_preds(array, args.dataset, output_file)

    build_ensemble(args.folder, args.ensemble_prefix, args.submission_prefix)


if __name__ == '__main__':
    main()