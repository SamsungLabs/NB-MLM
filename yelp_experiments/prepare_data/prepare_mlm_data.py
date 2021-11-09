import pandas as pd
import random
import argparse

def process(texts):
    return [' '.join(txt.split('\n')).replace('\\n', ' ').replace('\r','').replace('\\','') for txt in texts]



parser = argparse.ArgumentParser()
parser.add_argument("--all", action='store_true')

args = parser.parse_args()


df = pd.read_csv('./../DATA/raw/yelp_review_polarity_csv/train.csv', names=['label','text'], header=None)
texts = df["text"].tolist()

random.seed(42)
random.shuffle(texts)

dev_size = int(0.05*len(texts))

dev_texts = texts[:dev_size]
train_texts = texts[dev_size:]


data_dir = 'yelp-mlm'
if(args.all):
    train_texts += pd.read_csv('./../DATA/raw/yelp_review_polarity_csv/test.csv', names=['label','text'], header=None)["text"].tolist()
    data_dir = 'yelp-mlm-all'

with open("./../DATA/"+data_dir+"/train.texts", "w", encoding="utf8") as f:
    f.write('\n\n'.join(process(train_texts)))

with open("./../DATA/"+data_dir+"/dev.texts", "w", encoding="utf8") as f:
    f.write('\n\n'.join(process(dev_texts)))







