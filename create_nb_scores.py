
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_df', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--part', type=str, default="")

args = parser.parse_args()

if( (args.min_df).isdigit() ):
    score_dir = 'm_'+args.min_df
else:
    score_dir = 'freq'


if os.path.exists(os.path.join('./', args.dataset + '_experiments', 'SCORES', score_dir) ):
    pass
else:
    if(len(args.part)==0):
        os.system('cd ' + os.path.join('./', 'prepare_data') +';'+ 'bash token_scores.sh' + ' ' + args.dataset + ' ' + args.min_df )
    else:
        os.system('cd ' + os.path.join('./', 'prepare_data') +';'+ 'bash token_scores.sh' + ' ' + args.dataset + ' ' + args.min_df + ' ' + args.part)

