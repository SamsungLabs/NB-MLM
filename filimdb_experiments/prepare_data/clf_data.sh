#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi

# Parse Name of the directory with processed CLF data
ORIGIN_DIR="$DIR"/../FILIMDB_DATASET
FAIRSEQ_DIR="$DIR"/../../fairseq-mlm-pretrain
DATA_DIR="$DIR/../DATA/filimdb-clf"

# Create directory for CLF data
mkdir -p "$DATA_DIR"
# Check data for exist and download if needed
if ! [[ -d "$ORIGIN_DIR" ]]; then
    bash "$DIR"/download.sh
fi

# Copy raw .texts files
cp "$ORIGIN_DIR"/*.texts "$DATA_DIR"/
cp "$ORIGIN_DIR"/*.labels "$DATA_DIR"/

for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$FAIRSEQ_DIR"/vocab/encoder.json \
        --vocab-bpe "$FAIRSEQ_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --workers 4
done

VALID_TEXTS="$DATA_DIR"/1dev.texts.input0.bpe,"$DATA_DIR"/1dev-b.texts.input0.bpe
VALID_TEXTS="$VALID_TEXTS","$DATA_DIR"/2dev.texts.input0.bpe,"$DATA_DIR"/2dev-b.texts.input0.bpe

TEST_TEXTS="$DATA_DIR"/test.texts.input0.bpe,"$DATA_DIR"/test-b.texts.input0.bpe

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.bpe \
    --validpref "$VALID_TEXTS" \
    --testpref "$TEST_TEXTS" \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$FAIRSEQ_DIR"/vocab/dict.txt \
    --workers 4

VALID_LABELS="$DATA_DIR"/1dev.labels,"$DATA_DIR"/1dev-b.labels
VALID_LABELS="$VALID_LABELS","$DATA_DIR"/2dev.labels,"$DATA_DIR"/2dev-b.labels

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.labels \
    --validpref "$VALID_LABELS" \
    --destdir "$DATA_DIR"-bin/label \
    --workers 4
