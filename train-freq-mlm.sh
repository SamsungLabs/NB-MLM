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

CONFIG_PATH="$DATASET_NAME"/freq-mlm

if [ -n "$2" ]; then
    TEMP="$2"
else
    TEMP=1.0
fi

python run_experiment.py --config-path="$CONFIG_PATH" \
    --temperature="$TEMP" \
    --path-to-scores=freq \
    --dataset-part="$DATASET_PART"
