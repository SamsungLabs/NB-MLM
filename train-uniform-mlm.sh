#!/bin/bash
#SBATCH --gres=gpu:1 --time 0-23:00:00

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

CONFIG_PATH="$DATASET_NAME"/uniform-mlm

python run_experiment.py --config-path="$CONFIG_PATH" \
    --dataset-part="$DATASET_PART" ${@:2}
