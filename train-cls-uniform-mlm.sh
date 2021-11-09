#!/bin/bash
#SBATCH --gres=gpu:2 --time 0-30:00:00

if [ -z "$1" ]; then
    echo "Missed config path argument!"
    exit
fi

DATASET="$1"
DATASET_NAME="$(echo "$DATASET" | cut -d'/' -f1)"
DATASET_PART="$(echo "$DATASET" | cut -d'/' -f2)"

if [ "$DATASET_NAME" = "$DATASET_PART" ]; then
    DATASET_PART=""
fi

CONFIG_PATH="$DATASET_NAME"/cls-uniform-mlm

ALPHA=1.0
if [ -n "$2" ]; then
    ALPHA="$2"
fi

ONES_WEIGHT=20.0
if [ -n "$3" ]; then
    ONES_WEIGHT="$3"
fi

python run_experiment.py --config-path="$CONFIG_PATH" \
    --cls-loss-alpha="$ALPHA" \
    --cls-ones-weight="$ONES_WEIGHT" \
    --dataset-part="$DATASET_PART"
