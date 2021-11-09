#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi

MIN_DF_RE="^[0-9]+$"
ORIGIN_DIR="$DIR"/../DATA/filimdb-clf/
OUT_DIR="$DIR"/../SCORES
PY_SCRIPT="$DIR"/../../bayes_token_temperature_fix/make_token_scores.py
DICT_DIR="$DIR"/../DATA/filimdb-mlm-bin
DICT_PATH="$DICT_DIR"/input0/dict.txt

if ! [[ -d "$ORIGIN_DIR" ]]; then
    bash "$DIR"/clf_data.sh
fi

if [ -n "$1" ]; then
    if ! [[ $1 =~ $MIN_DF_RE ]]; then
        OUT_DIR="$OUT_DIR"/freq/
        mkdir -p "$OUT_DIR"
        python "$PY_SCRIPT" --sqrt \
            --input_dir "$ORIGIN_DIR" \
            --second_dict "$DICT_PATH" \
            --output_dir "$OUT_DIR"
    else
        OUT_DIR="$OUT_DIR"/m_"$1"/
        mkdir -p "$OUT_DIR"
        python "$PY_SCRIPT" --min_df "${1}" \
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
