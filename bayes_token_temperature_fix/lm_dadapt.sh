#!/bin/bash
#SBATCH --gres=gpu:2 --time 0-10:00:00

TOTAL_UPDATES=75000   # Total number of training steps
WARMUP_UPDATES=5000   # Warmup the learning rate over this many updates
PEAK_LR=1e-4          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_SENTENCES=16      # Number of sequences per batch (batch size)
UPDATE_FREQ=4         # Increase the batch size 16x
BREAK_MODE=complete_doc
MODEL=base
DATA_DIR=/home/mmcpavmironov/taskadapt/ROBERTA/FILIMDB_LM-bin/input0/
OUTPUT_DIR=filimdb_mlm_roberta_${MODEL}_${BREAK_MODE}
ROBERTA_PATH=/home/mmcpavmironov/taskadapt/ROBERTA/fairseq/roberta.${MODEL}/model.pt
USER_DIR=/home/mmcpavmironov/taskadapt/ROBERTA/custom

fairseq-train $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --tensorboard-logdir tb_${OUTPUT_DIR} \
    --save-dir checkpoints_${OUTPUT_DIR} --fp16 \
    --ddp-backend=no_c10d --skip-invalid-size-inputs-valid-test \
    --task "masked_lm" \
    --criterion "masked_lm" \
    --user-dir $USER_DIR \
    --arch roberta_$MODEL \
    --sample-break-mode $BREAK_MODE \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --log-format simple --log-interval 1
