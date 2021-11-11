#!/bin/bash
DIR=$(dirname "$(realpath "$0") ")

ORIG_DIR=$(realpath "$DIR"/../filimdb_experiments/FILIMDB_DATASET/)
DATASET_DIR="$DIR"/DATASET

if ! [[ -d "$DATASET_DIR" ]]; then
    mkdir -p "$DATASET_DIR"
fi

if ! [[ -d "$ORIG_DIR" ]]; then
    cd "$DIR"/../filimdb_experiments/prepare_data/
    bash download.sh
    cd "$DIR"
fi

cp ${ORIG_DIR}/train_unlabeled.texts ${DATASET_DIR}
