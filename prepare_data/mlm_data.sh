#!/bin/bash

DATASET=$1

#if [ -n "$SLURM_JOB_ID" ] ; then
#    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
#else
DIR=$(dirname "$(realpath "$0") ")
#fi
ORIGIN_DIR="$(realpath "$DIR"/../"$DATASET"_experiments/DATASET)"
FAIRSEQ_DIR="$(realpath "$DIR"/../fairseq-mlm-pretrain)"
MODE=$2

if [ -z "$MODE" ]; then
    DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-mlm
else
    if [ "$MODE" = "all" ]; then
	   DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-mlm-all
    elif [ "$MODE" = "curated" ]; then
        DATA_DIR="$DIR"/../"$DATASET"_experiments/DATA/"$DATASET"-mlm-curated
    else
        echo "Possible arguments:"
        echo "bash mlm_data.sh dataset_name"
        echo "bash mlm_data.sh dataset_name curated"
        echo "bash mlm_data.sh dataset_name all"
        exit
    fi
fi

mkdir -p "$DATA_DIR"

#if ! [[ -d "$ORIGIN_DIR" ]]; then
#    bash "$DIR"/download.sh "$DATASET"
#fi

if [ -z "$MODE" ]; then
    cp "$ORIGIN_DIR"/*.texts "$DATA_DIR"/

elif [ "$MODE" = "all" ]; then
    SEED=$3
    if [ -z "$SEED" ]; then
        SEED=42
    fi
    
    python "$DIR"/prepare_mlm_all.py \
       		    --origin_dir "$ORIGIN_DIR" \
                --data_dir "$DATA_DIR" \
                --seed "$SEED"

elif [ "$MODE" = "curated" ]; then
    cat "$ORIGIN_DIR"/train.texts "$ORIGIN_DIR"/train_unlabeled.texts > "$DATA_DIR"/train.texts
    cp "$ORIGIN_DIR"/dev.texts "$DATA_DIR" 
    
fi

#cp "$ORIGIN_DIR"/*.labels "$DATA_DIR"/

sed -i 's!$!\n!' "$DATA_DIR"/*.texts

for FILE in "$DATA_DIR"/*.texts; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$FAIRSEQ_DIR"/vocab/encoder.json \
        --vocab-bpe "$FAIRSEQ_DIR"/vocab/vocab.bpe \
        --inputs "$FILE" --outputs "$FILE".input0.bpe \
        --keep-empty --workers 60
    
    python "$DIR"/truncate.py "$FILE".input0.bpe 
done


fairseq-preprocess --only-source \
    --trainpref "$DATA_DIR"/train.texts.input0.truncate.bpe \
    --validpref "$DATA_DIR"/dev.texts.input0.truncate.bpe \
    --destdir "$DATA_DIR"-bin/input0 \
    --srcdict "$FAIRSEQ_DIR"/vocab/dict.txt \
    --workers 60
