#!/bin/bash

if [ -n "$1" ]; then
    mkdir "FILIMDB_INFORM_SCORES_${1}"
    python ./bayes_token_temperature_fix/make_inform_token_scores.py \
        --min_df "$1" \
        --input_dir ./FILIMDB/ \
        --second_dict ./FILIMDB_LM-bin/input0/dict.txt \
        --output_dir "./FILIMDB_INFORM_SCORES_$1/"
else
    mkdir "FILIMDB_INFORM_SCORES_50"
    python ./bayes_token_temperature_fix/make_inform_token_scores.py \
        --min_df 50 \
        --input_dir ./FILIMDB/ \
        --second_dict ./FILIMDB_LM-bin/input0/dict.txt \
        --output_dir ./FILIMDB_INFORM_SCORES_50/
fi
