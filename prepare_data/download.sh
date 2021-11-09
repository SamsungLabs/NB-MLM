#!/bin/bash

DATASET=$1

if [ "$DATASET" = "ag" ]; then
	DATA_PATH="ag"
elif [ "$DATASET" = "rct" ]; then
	DATA_PATH="rct-20k"
elif [ "$DATASET" = "rct500" ]; then
    DATA_PATH="rct-sample"
elif [ "$DATASET" = "chemprot" ]; then
	DATA_PATH="chemprot"
elif [ "$DATASET" = "aclarc" ]; then
	DATA_PATH="citation_intent"
elif [ "$DATASET" = "scierc" ]; then
	DATA_PATH="sciie"
elif [ "$DATASET" = "hyperp" ]; then
	DATA_PATH="hyperpartisan_news"
elif [ "$DATASET" = "helpful" ]; then
	DATA_PATH="amazon"
elif [ "$DATASET" = "imdb" ]; then
    DATA_PATH="imdb"
else
    echo "Possible datasets: ag, rct, rct500, chemprot, aclarc, scierc, hyperp, helpful, imdb"
    exit
fi

URL="https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/$DATA_PATH"

if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi


DATASET_DIR="$DIR"/../"$DATASET"_experiments/DATASET
echo $DATASET_DIR
if ! [[ -d "$DATASET_DIR" ]]; then
    mkdir -p "$DATASET_DIR"
fi

SPLITS=("train" "dev" "test")
for FILE in "${SPLITS[@]}";
do
    if ! [ -e  "$DATASET_DIR"/"$FILE".jsonl ]; then
        curl -Lo "$DATASET_DIR"/"$FILE".jsonl "$URL"/"$FILE".jsonl
        python "$DIR"/process.py \
                --name "$FILE" \
       		--dataset "$DATASET" \
                --path "$DATASET_DIR"/"$FILE".jsonl 
    fi
done
