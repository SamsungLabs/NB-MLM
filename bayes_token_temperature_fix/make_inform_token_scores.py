from mask import MutualInformationMasking
from sklearn.naive_bayes import MultinomialNB
from os import path
from fairseq.data.dictionary import Dictionary
import json
import argparse
import numpy as np


def encode(text):
    tokens_a = list(map(str, task_dict.encode_line(text.strip()).numpy().tolist()))
    tokens = ["dump"] + tokens_a
    return tokens


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--second_dict', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--min_df', type=int, default=50)

args = parser.parse_args()
print(args.min_df)

task_dict = Dictionary.load(args.second_dict)

mw = MutualInformationMasking(MultinomialNB, encode, None, 0.6, save_rate=None, file=None, min_df=args.min_df)
mw.Initialize(path.join(args.input_dir, 'train.texts.input0.bpe'),
              path.join(args.input_dir + 'train.labels'))
lc = mw.left_context
rc = mw.right_context

al = {}
for sw in lc:
    acc_l = np.array([0., 0.])
    for v in lc[sw].values():
        acc_l += v
    al[sw] = acc_l

ar = {}
for fw in rc:
    acc_r = np.array([0., 0.])
    for v in rc[fw].values():
        acc_r += v
    ar[fw] = acc_r

ar_j = {k: v.tolist() for k, v in ar.items()}
al_j = {k: v.tolist() for k, v in al.items()}

lc_j = {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in lc.items()}
rc_j = {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in rc.items()}

with open(path.join(args.output_dir, 'ar.json'), 'w') as outfile:
    json.dump(ar_j, outfile)

with open(path.join(args.output_dir, 'al.json'), 'w') as outfile:
    json.dump(al_j, outfile)

with open(path.join(args.output_dir, 'rc.json'), 'w') as outfile:
    json.dump(rc_j, outfile)

with open(path.join(args.output_dir, 'lc.json'), 'w') as outfile:
    json.dump(lc_j, outfile)
