DIR=$(dirname "$(realpath "$0") ")
PRETRAIN_MODEL_DIR="$DIR"/../fairseq-mlm-pretrain


DATA_DIR="$DIR/amazon-mlm"


mkdir -p "$DATA_DIR"


for FILE in train.texts dev.texts; do
    python new_lines.py "$FILE" 
done;



for FILE in train.newlined.texts dev.newlined.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$PRETRAIN_MODEL_DIR"/vocab/encoder.json \
        --vocab-bpe "$PRETRAIN_MODEL_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$DATA_DIR/$FILE".input0.bpe \
        --keep-empty --workers 20
done


for FILE in ./amazon-mlm/*0.bpe; do
      python ./../prepare_data/truncate.py "$FILE"
done;



fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.newlined.texts.input0.truncate.bpe \
    --validpref "$DATA_DIR"/dev.newlined.texts.input0.truncate.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$PRETRAIN_MODEL_DIR"/vocab/dict.txt \
    --workers 20

