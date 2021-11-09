#!/bin/bash
#SBATCH --gres=gpu:1 --time 0-10:00:00

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CKPT="$SLURM_ARRAY_TASK_ID"
else
    CKPT="$3"
fi

DATASET="$1"
DATASET_NAME="$(echo "$DATASET" | cut -d'/' -f1)"
DATASET_PART="$(echo "$DATASET" | cut -d'/' -f2)"

if [ "$DATASET_NAME" = "$DATASET_PART" ]; then
    DATASET_PART=""
fi

CONFIG_PATH="$DATASET_NAME"/clf
ROBERTA_PATH="$DATASET_NAME"_experiments/RUNS/"$2"/ckpt/checkpoint"$CKPT".pt

python run_experiment.py --config-path="$CONFIG_PATH" \
    --restore-file="$ROBERTA_PATH" \
    --dataset-part="$DATASET_PART" --seed=2020 --keep-best-checkpoints=1 --no-epoch-checkpoints --best-checkpoint-metric="accuracy" --keep-interval-updates=1 --keep-last-epochs=1
