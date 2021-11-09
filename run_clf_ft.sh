#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=run_clf_ft-%j.log

DATASET=$1
ckpts=${@:2}

for x in $ckpts; do echo $x; done | cat -n
for x in $ckpts; do
  echo $x  
  python run_experiment.py --config-path=${DATASET}/clf --restore-file=$x --keep-best-checkpoints=1 --no-epoch-checkpoints --best-checkpoint-metric="accuracy" --keep-interval-updates=1 --keep-last-epochs=1
done
