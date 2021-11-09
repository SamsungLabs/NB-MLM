#!/bin/bash
#SBATCH --gres=gpu:1
TOTAL_NUM_UPDATES=7812 # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469     # 6 percent of the number of updates
LR=1e-05               # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=8 # Batch size.
#ROBERTA_PATH=/path/to/roberta/model.pt
#ROBERTA_PATH='/srv/home/arefyev/.cache/torch/pytorch_fairseq/83e3a689e28e5e4696ecb0bbb05a77355444a5c8a3437e0f736d8a564e80035e.c687083d14776c1979f3f71654febb42f2bb3d9a94ff7ebdfe1ac6748dba89d2/model.pt'
CKPT=$SLURM_ARRAY_TASK_ID
ROBERTA_DIR=$1
ROBERTA_PATH=${ROBERTA_DIR}/checkpoint${CKPT}.pt
for dropout in 0.1 0.3; do
  for att_do in 0.1 0.2; do
    for pooler_do in 0.0 0.3; do
      for wd in 0.1 0.3; do
        TB_DIR=${ROBERTA_DIR}/clf_tb_$(basename $1)_wd${wd}attdo${att_do}_poolerdo${pooler_do}/checkpoint${CKPT}
        SAVE_DIR=${ROBERTA_DIR}/clf_ckpts_$(basename $1)_wd${wd}attdo${att_do}_poolerdo${pooler_do}/checkpoint${CKPT}
        fairseq-train ../FILIMDB-bin/ --user-dir /home/narefyev/taskadapt/ROBERTA/temp_masked_lm \
          --valid-subset valid,valid1,valid2,valid3 \
          --restore-file $ROBERTA_PATH --tensorboard-logdir $TB_DIR --save-dir $SAVE_DIR \
          --max-positions 512 \
          --max-sentences $MAX_SENTENCES \
          --max-tokens 4400 \
          --task sentence_prediction \
          --reset-optimizer --reset-dataloader --reset-meters \
          --required-batch-size-multiple 1 \
          --init-token 0 --separator-token 2 \
          --arch roberta_base \
          --criterion sentence_prediction \
          --num-classes $NUM_CLASSES \
          --dropout $dropout --attention-dropout $att_do --pooler-dropout $pooler_do \
          --weight-decay $wd --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
          --clip-norm 0.0 \
          --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
          --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
          --max-epoch 10 \
          --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
          --truncate-sequence \
          --find-unused-parameters \
          --update-freq 4
      done
    done
  done
done
