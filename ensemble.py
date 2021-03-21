import os

for root, dirs, files in os.walk(".", topdown=False):
   for name in dirs:
      directory = os.path.join(root, name)
      if 'submission_large' in directory:
          preds = os.listdir(directory)
          print(preds)
          preds = [ x for x in preds if x.startswith('preds')]
          if preds:
            preds = sorted(preds)[-1]
            print(preds)
        