#!/bin/bash
DIR=$(dirname "$(realpath "$0") ")

FAIRSEQ_DIR="$DIR"/../../fairseq-mlm-pretrain

DATA_DIR="$DIR/../DATA/yelp-clf-dev2"

mkdir -p "$DATA_DIR"

python prepare_clf_data_dev2.py

for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$FAIRSEQ_DIR"/vocab/encoder.json \
        --vocab-bpe "$FAIRSEQ_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --workers 4
done


fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.bpe \
    --validpref "$DATA_DIR"/dev.texts.input0.bpe \
    --testpref "$DATA_DIR"/test.texts.input0.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$FAIRSEQ_DIR"/vocab/dict.txt \
    --workers 4

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.labels \
    --validpref "$DATA_DIR"/dev.labels \
    --testpref "$DATA_DIR"/test.labels \
    --destdir "$DATA_DIR"-bin/label \
    --workers 4
