

DIR=$(dirname "$(realpath "$0") ")

if [ -z "$1" ]; then
    DATA_DIR="$DIR/../amazon_experiments/DATA/"
else
    DATA_DIR="$DIR/../amazonYelp_experiments/DATA/"
fi


mkdir -p "$(dirname "$DATA_DIR")"

mkdir -p DATA_DIR

cp -r ./amazon-mlm "$DATA_DIR"

cp -r ./amazon-mlm-bin "$DATA_DIR"


if [ -z "$1" ]; then
    :
else
    mv "$DATA_DIR"/amazon-mlm "$DATA_DIR"amazonYelp-mlm
    mv "$DATA_DIR"/amazon-mlm-bin "$DATA_DIR"amazonYelp-mlm-bin
fi
