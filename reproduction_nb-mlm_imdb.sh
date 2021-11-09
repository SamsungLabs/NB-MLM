#run amazon DAPT with imdb features
bash train-nb-mlm.sh amazon 0.1 10 --seed=101

#run TAPT over DAPT
bash train-nb-mlm.sh imdb/all 0.4 50 --seed=101 --restore-file=./amazon_experiments/RUNS/amazon_nb_mlm_temperature_0.1_path_to_scores_m_10_seed_101/ckpt/checkpoint_1_20000.pt --reset-optimizer --reset-dataloader --reset-meters

#pop up results
cd ./imdb_experiments/RUNS/
mv ./imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_./amazon_experiments/RUNS/amazon_nb_mlm_temperature_0.1_path_to_scores_m_10_seed_101/ckpt/checkpoint_1_20000.pt_reset_optimizer_True_reset_dataloader_True_reset_meters_True/ckpt/ ./imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_.
cd ./../../

#run classifier fine-tuning
bash train-clf.sh imdb imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_. 90

#run classifier evaluation
bash run_clf_eval.sh imdb test "./imdb_experiments/RUNS/imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt"
