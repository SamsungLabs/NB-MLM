#Run amazon DAPT
bash train-uniform-mlm.sh amazon --seed=103

#Run TAPT over DAPT
bash train-uniform-mlm.sh imdb/all --seed=103 --restore-file=./amazon_experiments/RUNS/amazon_uniform_mlm_seed_103/ckpt/checkpoint_1_20000.pt --reset-optimizer --reset-dataloader --reset-meters

#pop up results
cd ./imdb_experiments/RUNS
mv ./imdb_all_uniform_mlm_seed_103_restore_file_./amazon_experiments/RUNS/amazon_uniform_mlm_seed_103/ckpt/checkpoint_1_20000.pt_reset_optimizer_True_reset_dataloader_True_reset_meters_True/ckpt/ ./imdb_all_uniform_mlm_seed_103_restore_file_.
cd ./../../

#run classifier finetuning
bash train-clf.sh imdb imdb_all_uniform_mlm_seed_103_restore_file_. 90

#run evaluation
bash run_clf_eval.sh imdb test "./imdb_experiments/RUNS/imdb_all_uniform_mlm_seed_103_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt"
