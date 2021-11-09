DIR=$(dirname "$(realpath "$0") ")
PRETRAIN_MODEL_DIR="$DIR"/../../fairseq-mlm-pretrain

if [ -z "$1" ]; then
    DATA_DIR="$DIR/../DATA/yelp-mlm"
else
    
    DATA_DIR="$DIR/../DATA/yelp-mlm-all"
fi


mkdir -p "$DATA_DIR"


if [ -z "$1" ]; then
    python prepare_mlm_data.py
else
    python prepare_mlm_data.py --all
fi


for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$PRETRAIN_MODEL_DIR"/vocab/encoder.json \
        --vocab-bpe "$PRETRAIN_MODEL_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --keep-empty --workers 2
done


fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.bpe \
    --validpref "$DATA_DIR"/dev.texts.input0.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$PRETRAIN_MODEL_DIR"/vocab/dict.txt \
    --workers 2
