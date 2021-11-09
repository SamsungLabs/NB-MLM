#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi

ORIGIN_DIR="$DIR"/../DATA/filimdb-clf/
OUT_DIR="$DIR"/../SCORES
PY_SCRIPT="$DIR"/../../bayes_token_temperature_fix/make_inform_token_scores.py
DICT_DIR="$DIR"/../DATA/filimdb-mlm-bin
DICT_PATH="$DICT_DIR"/input0/dict.txt

if ! [[ -d "$ORIGIN_DIR" ]]; then
    bash "$DIR"/clf_data.sh
fi

if [ -n "$1" ]; then
    MIN_DF="$1"
else
    MIN_DF=50
fi

OUT_DIR="$OUT_DIR"/inform_m_"$MIN_DF"
mkdir -p "$OUT_DIR"
python "$PY_SCRIPT" --min_df "$MIN_DF" \
        --input_dir "$ORIGIN_DIR" \
        --second_dict "$DICT_PATH" \
        --output_dir "$OUT_DIR"
