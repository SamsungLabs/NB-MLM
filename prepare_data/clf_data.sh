#!/bin/bash
DATASET=$1

#if [ -n "$SLURM_JOB_ID" ] ; then
#    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
#else
DIR=$(dirname "$(realpath "$0") ")
#fi

ORIGIN_DIR="$(realpath "$DIR"/../"$DATASET"_experiments/DATASET)"
FAIRSEQ_DIR="$(realpath "$DIR"/../fairseq-mlm-pretrain)"

#MODE=$2

#if [ -z "$MODE" ]; then
#    DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-clf
#else
#    if [ "$MODE" = "all" ]; then
#        DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-clf-all
#    else
#        echo "Possible arguments:"
#        echo "  bash clf_data.sh dataset_name"
#        echo "  bash clf_data.sh dataset_name all"
#       exit
#    fi
#fi

DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-clf

mkdir -p "$DATA_DIR"

#if ! [[ -d "$ORIGIN_DIR" ]]; then
#    bash "$DIR"/download.sh "$DATASET"
#fi

# Copy raw .texts files
cp "$ORIGIN_DIR"/*.texts "$DATA_DIR"/
cp "$ORIGIN_DIR"/*.labels "$DATA_DIR"/

#if [ "$MODE" = "all" ]; then
#    cat "$DATA_DIR"/dev.texts >> "$DATA_DIR"/train.texts
#    cat "$DATA_DIR"/dev.labels >> "$DATA_DIR"/train.labels
#    cat "$DATA_DIR"/test.texts >> "$DATA_DIR"/train.texts
#    cat "$DATA_DIR"/test.labels >> "$DATA_DIR"/train.labels
#fi

for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$FAIRSEQ_DIR"/vocab/encoder.json \
        --vocab-bpe "$FAIRSEQ_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --workers 60
    python "$DIR"/truncate.py "$FILE".input0.bpe
done

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.truncate.bpe \
    --validpref "$DATA_DIR"/dev.texts.input0.truncate.bpe \
    --testpref "$DATA_DIR"/test.texts.input0.truncate.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$FAIRSEQ_DIR"/vocab/dict.txt \
    --workers 60

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.labels \
    --validpref "$DATA_DIR"/dev.labels \
    --testpref "$DATA_DIR"/test.labels \
    --destdir "$DATA_DIR"-bin/label \
    --workers 60
