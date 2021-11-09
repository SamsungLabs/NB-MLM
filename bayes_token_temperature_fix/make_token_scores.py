from mask import IdsLevelMasking, SqrtMasking
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from fairseq.data.dictionary import Dictionary

import argparse


def encode(text):
    tokens_a = list(map(str, task_dict.encode_line(text.strip()).numpy().tolist()))
    tokens = ["dump"] + tokens_a
    return tokens


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--second_dict", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--min_df", type=int, default=50)
parser.add_argument("--ngram_range", type=str, default="1;1")
parser.add_argument("--sqrt", default=False, action="store_true")

args = parser.parse_args()
print(args.min_df)
lr = int(args.ngram_range.split(";")[0])
ur = int(args.ngram_range.split(";")[1])
ngram_range = (lr, ur)

task_dict = Dictionary.load(args.second_dict)

if args.sqrt:
    print("sqrt")
    mw = SqrtMasking(MultinomialNB, encode, None, save_rate=None, file=None)
else:
    mw = IdsLevelMasking(MultinomialNB, encode, None, 0.6, save_rate=None, file=None, min_df=args.min_df, ngram_range=ngram_range)
mw.Initialize(args.input_dir + "train.texts.input0.bpe", args.input_dir + "train.labels")

d = {"keys": [], "scores": []}
for k in mw.d.keys():
    d["keys"].append(k)
    d["scores"].append(mw.d[k])
pd.DataFrame.from_dict(d).to_csv(args.output_dir + "token_scores.csv", index=False)
