#!/bin/bash
#SBATCH --gres=gpu:2 --time 0-60:00:00

TOTAL_UPDATES=75000   # Total number of training steps
WARMUP_UPDATES=5000   # Warmup the learning rate over this many updates
PEAK_LR=1e-4          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
MAX_SENTENCES=12      # Number of sequences per batch (batch size)
UPDATE_FREQ=5         # Increase the batch size 16x
BREAK_MODE=complete_doc
MODEL=large
DATA_DIR=../FILIMDB_LM-bin/input0/
TEMPERATURE=$1
MIN_DF=$2
OUTPUT_DIR=lm-roberta-${MODEL}_nbadapt-${BREAK_MODE}_FILIMDB-lr1e-4-75ksteps-bs128_temp${TEMPERATURE}_mindf${MIN_DF}

fairseq-train --restore-file ../fairseq/roberta.${MODEL}/model.pt --tensorboard-logdir tb_${OUTPUT_DIR} --save-dir checkpoints_${OUTPUT_DIR} --fp16 $DATA_DIR \
  --log_mask --log_dir_mask=logsmask_${OUTPUT_DIR} \
  --ddp-backend=no_c10d --skip-invalid-size-inputs-valid-test \
  --task temp_masked_lm --criterion masked_lm --user-dir /home/narefyev/taskadapt/ROBERTA/temp_masked_lm \
  --arch roberta_${MODEL} --sample-break-mode $BREAK_MODE --tokens-per-sample $TOKENS_PER_SAMPLE \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
  --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --temperature ${TEMPERATURE} --path_to_scores ../FILIMDB_SCORES_${MIN_DF}/
