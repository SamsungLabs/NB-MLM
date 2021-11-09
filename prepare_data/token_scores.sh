#!/bin/bash

DATASET=$1

#if [ -n "$SLURM_JOB_ID" ] ; then
#    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
#else
DIR=$(dirname "$(realpath "$0") ")
#fi

MIN_DF_RE="^[0-9]+$"
ORIGIN_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-clf/
OUT_DIR="$DIR"/../"$DATASET"_experiments/SCORES
PY_SCRIPT="$DIR"/../bayes_token_temperature_fix/make_token_scores.py

if [ -n "$3" ]; then
    DICT_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-mlm-"$3"-bin
else
    DICT_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-mlm-bin
fi

DICT_PATH="$DICT_DIR"/input0/dict.txt

if ! [[ -d "$ORIGIN_DIR" ]]; then
    bash "$DIR"/clf_data.sh "$DATASET"
fi

if [ -n "$2" ]; then
    if ! [[ $2 =~ $MIN_DF_RE ]]; then
        OUT_DIR="$OUT_DIR"/freq/
        mkdir -p "$OUT_DIR"
        python "$PY_SCRIPT" --sqrt \
            --input_dir "$ORIGIN_DIR" \
            --second_dict "$DICT_PATH" \
            --output_dir "$OUT_DIR"
    else
        OUT_DIR="$OUT_DIR"/m_"$2"/
        mkdir -p "$OUT_DIR"
        python "$PY_SCRIPT" --min_df "${2}" \
            --input_dir "$ORIGIN_DIR" \
            --second_dict "$DICT_PATH" \
            --output_dir "$OUT_DIR"
    fi
else
    OUT_DIR="$OUT_DIR"/m_50/
    mkdir -p "$OUT_DIR"
    python "$PY_SCRIPT" --min_df 50 \
        --input_dir "$ORIGIN_DIR" \
        --second_dict "$DICT_PATH" \
        --output_dir "$OUT_DIR"
fi
