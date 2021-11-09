#!/bin/bash
#SBATCH --gres=gpu:1 --time 0-55:00:00

if [ -z "$1" ]; then
    echo "Missed config path argument!"
    exit
fi

DATASET="$1"
DATASET_NAME="$(echo "$DATASET" | cut -d'/' -f1)"
DATASET_PART="$(echo "$DATASET" | cut -d'/' -f2)"


MIN_DF_RE="^[0-9]+$"


if [ "$DATASET_NAME" = "$DATASET_PART" ]; then
    DATASET_PART=""
fi

CONFIG_PATH="$DATASET_NAME"/nb-mlm

if [ -n "$2" ]; then
    TEMP="$2"
else
    TEMP=1.0
fi

if [ -n "$3" ]; then
    MIN_DF="$3"
else
    MIN_DF=50
fi

if ! [[ $3 =~ $MIN_DF_RE ]]; then
    PTS=freq
else
    PTS=m_"$MIN_DF"
fi

if [ -n "$DATASET_PART" ]; then 
    python create_nb_scores.py --dataset="$DATASET_NAME" --min_df="$MIN_DF" --part="$DATASET_PART"
else
    python create_nb_scores.py --dataset="$DATASET_NAME" --min_df="$MIN_DF" 
fi 
echo "$PTS"
python run_experiment.py --config-path="$CONFIG_PATH" \
    --temperature="$TEMP" \
    --path-to-scores="$PTS" \
    --dataset-part="$DATASET_PART" ${@:4} 

