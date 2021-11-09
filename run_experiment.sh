#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00

DATASET=$1
TEMP=$2
MIN_DF=$3
MODE=$4

if [ -n "$MODE" ] ; then
    bash train-nb-mlm.sh ${DATASET}/${MODE} $TEMP $MIN_DF
    CKPT_DIR=${DATASET}_${MODE}_nb_mlm_temperature_${TEMP}_path_to_scores_m_${MIN_DF}
    for (( CKPT_NUM=0; CKPT_NUM<=100; CKPT_NUM+=10 )); do
        bash train-clf.sh $DATASET $CKPT_DIR $CKPT_NUM
        bash evaluate.sh $DATASET $CKPT_DIR dev $CKPT_NUM macro
        bash evaluate.sh $DATASET $CKPT_DIR test $CKPT_NUM macro
        bash evaluate.sh $DATASET $CKPT_DIR train $CKPT_NUM macro
    done
else
    bash train-nb-mlm.sh $DATASET $TEMP $MIN_DF
    CKPT_DIR=${DATASET}_nb_mlm_temperature_${TEMP}_path_to_scores_m_${MIN_DF}
    for (( CKPT_NUM=0; CKPT_NUM<=100; CKPT_NUM+=10 )); do
        bash train-clf.sh $DATASET $CKPT_DIR $CKPT_NUM
        bash evaluate.sh $DATASET $CKPT_DIR dev $CKPT_NUM macro
        bash evaluate.sh $DATASET $CKPT_DIR test $CKPT_NUM macro
        bash evaluate.sh $DATASET $CKPT_DIR train $CKPT_NUM macro
    done
fi
