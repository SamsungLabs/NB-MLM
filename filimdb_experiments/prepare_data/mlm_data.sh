#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi

ORIGIN_DIR="$DIR"/../FILIMDB_DATASET
FAIRSEQ_DIR="$DIR"/../../fairseq-mlm-pretrain
DATA_DIR="$DIR/../DATA/filimdb-mlm"

mkdir -p "$DATA_DIR"
if ! [[ -d "$ORIGIN_DIR" ]]; then
    bash "$DIR"/download.sh
fi

cat "$ORIGIN_DIR"/[t1]*.texts > "$DATA_DIR"/train.texts
cp "$ORIGIN_DIR"/2*.texts "$DATA_DIR"
cp "$ORIGIN_DIR"/*.labels "$DATA_DIR"/
sed -i 's!$!\n!' "$DATA_DIR"/*.texts

for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$FAIRSEQ_DIR"/vocab/encoder.json \
        --vocab-bpe "$FAIRSEQ_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --keep-empty --workers 4
done

fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.bpe \
    --validpref "$DATA_DIR"/2dev.texts.input0.bpe,"$DATA_DIR"/2dev-b.texts.input0.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$FAIRSEQ_DIR"/vocab/dict.txt \
    --workers 4
