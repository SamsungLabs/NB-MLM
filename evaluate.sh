#!/bin/bash
#SBATCH --gres=gpu:1 --time 0-01:00:00

if [ -n "$4" ]; then
    CKPT=clf_tune_ckpt_"$4"
else
    CKPT=clf_tune_ckpt_"$SLURM_ARRAY_TASK_ID"
fi

F_SCORE_AVG="macro"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    if [ -n "$4" ]; then
        F_SCORE_AVG="$4"
        CKPT=clf_tune_ckpt_"$SLURM_ARRAY_TASK_ID"
    fi
else
    if [ -n "$5" ]; then
        F_SCORE_AVG="$5"
    fi
fi

DATASET="$1"
DATASET_NAME="$(echo "$DATASET" | cut -d'/' -f1)"
DATASET_PART="$(echo "$DATASET" | cut -d'/' -f2)"

if [ "$DATASET_NAME" = "$DATASET_PART" ]; then
    DATASET_PART=""
fi

MLM_DIR="$2"
MODE="$3"

ROBERTA_DIR="$DATASET_NAME"_experiments/RUNS/"$MLM_DIR"/"$CKPT"
DATA_PATH="$DATASET_NAME"_experiments/DATA/"$DATASET_NAME"-clf

if [ -n "$DATASET_PART" ]; then
    DATA_PATH="$DATA_PATH"-"$DATASET_PART"
fi
DATA_PATH="$DATA_PATH"-bin
echo $ROBERTA_DIR
echo $DATA_PATH
echo $F_SCORE_AVG

python evaluate.py \
  --model-dir "$ROBERTA_DIR"/ckpt \
  --data-path "$DATA_PATH" \
  --mode "$MODE" \
  --save-dir "$ROBERTA_DIR" \
  --f-score-avg "$F_SCORE_AVG"
