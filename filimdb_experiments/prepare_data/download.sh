#!/bin/bash
REPO="filimdb_evaluation"
URL=https://github.com/nvanva/"$REPO".git
if [ -n "$SLURM_JOB_ID" ] ; then
    DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP "Command=\K.*sh")")
else
    DIR=$(dirname "$(realpath "$0") ")
fi
#REPO="filimdb_evaluation"
#REPO_DIR="$DIR"/../../"$REPO"
REPO_DIR="$DIR"/"$REPO"


DATASET_DIR=$(realpath "$DIR"/../FILIMDB_DATASET)
if ! [[ -d "$DATASET_DIR" ]]; then
    mkdir -p "$DATASET_DIR"
fi

if ! [[ -d "$REPO_DIR" ]]; then
    git clone "$URL"
fi
tar xvzf "$REPO_DIR"/FILIMDB.tar.gz -C "$DATASET_DIR" --strip 1

sed -i 's!pos!1!' "$DATASET_DIR"/*labels
sed -i 's!neg!0!' "$DATASET_DIR"/*labels

for FILE in "$DATASET_DIR"/dev*;
do
    lines="$(wc -l "$FILE" | cut -f 1 -d ' ')"
    split -l "$(expr "$lines" / 2)" "$FILE"
    mv xaa "$(dirname "$FILE")"/1"$(basename "$FILE")"
    mv xab "$(dirname "$FILE")"/2"$(basename "$FILE")"
    rm "$FILE"
done
