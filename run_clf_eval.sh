#!/bin/bash
#SBATCH --gres=gpu:1 --time 0-01:00:00
DATASET_NAME="$1"
MODE="$2"
CKPT_DIR="$3"

DATA_PATH="$DATASET_NAME"_experiments/DATA/"$DATASET_NAME"-clf
DATA_PATH="$DATA_PATH"-bin
echo $DATA_PATH

python evaluate.py \
  --model-dir "$CKPT_DIR" \
  --data-path "$DATA_PATH" \
  --mode "$MODE" \
  --save-dir "$CKPT_DIR" 
