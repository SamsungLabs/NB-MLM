#run amazon DAPT with yelp features
bash train-nb-mlm.sh amazonYelp 0.1 50 --seed=2020

#run TAPT over DAPT
bash train-nb-mlm.sh yelp/all 0.4 50 --seed=2020 --restore-file=./amazonYelp_experiments/RUNS/amazonYelp_nb_mlm_temperature_0.1_path_to_scores_m_50_seed_2020/ckpt/checkpoint_1_20000.pt --reset-optimizer --reset-dataloader --reset-meters

#pop up results
cd ./yelp_experiments/RUNS/
mv ./yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_./amazonYelp_experiments/RUNS/amazonYelp_nb_mlm_temperature_0.1_path_to_scores_m_50_seed_2020/ckpt/checkpoint_1_20000.pt_reset_optimizer_True_reset_dataloader_True_reset_meters_True/ckpt/ ./yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_.
cd ./../../

#run classifier fine-tuning
bash train-clf.sh yelp yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_. 27

#run classifier evaluation
bash run_clf_eval.sh yelp test "./yelp_experiments/RUNS/yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt"
