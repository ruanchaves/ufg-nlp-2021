import os

def get_preds(root_dir):
    output = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in dirs:
        directory = os.path.join(root, name)
        if 'submission_large' in directory:
            preds = os.listdir(directory)
            preds = [x for x in preds if x.startswith('preds')]
            if preds:
                preds = sorted(preds)[-1]
                preds = os.path.join(directory, preds)
                output.append(preds)
    return output


def main():
    preds = get_preds(".")
    print(preds)

if __name__ == '__main__':
    main()