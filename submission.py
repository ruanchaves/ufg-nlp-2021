import argparse
import pandas as pd
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--source',
        type=str
    )

    parser.add_argument(
        '--dataset',
        type=str
    )

    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv'
    )

    args = parser.parse_args()

    return args

def main():

    args = get_args()
    a = np.load(args.source)
    df = pd.read_csv(args.dataset)
    ids = df['Id'].values.tolist()
    preds = ['Id,Category']
    label_dict = {
        0: 'Negativo',
        1: 'Neutro',
        2: 'Positivo'
    }
    a = np.argmax(a, axis=1)
    print(a)
    for idx, item in enumerate(a):
        row = '{0},{1}'.format(ids[idx], label_dict[item])
        preds.append(row)

    preds = '\n'.join(preds)

    with open(args.output, 'w+') as f:
        print(preds, file=f)

if __name__ == '__main__':
    main()
