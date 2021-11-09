#run amazon DAPT
bash train-uniform-mlm.sh amazonYelp --seed=2021
#run TAPT over DAPT
bash train-uniform-mlm.sh yelp/all --seed=2021 --restore-file=./amazonYelp_experiments/RUNS/amazonYelp_uniform_mlm_seed_2021/ckpt/checkpoint_1_20000.pt --reset-optimizer --reset-dataloader --reset-meters

#pop up results
cd ./yelp_experiments/RUNS
mv ./yelp_all_uniform_mlm_seed_2021_restore_file_./amazonYelp_experiments/RUNS/amazonYelp_uniform_mlm_seed_2021/ckpt/checkpoint_1_20000.pt_reset_optimizer_True_reset_dataloader_True_reset_meters_True/ckpt/ ./yelp_all_uniform_mlm_seed_2021_restore_file_.
cd ./../../

#run classifier fine-tuning
bash train-clf.sh yelp yelp_all_uniform_mlm_seed_2021_restore_file_. 27

#run classifier evaluation
bash run_clf_eval.sh yelp test "./yelp_experiments/RUNS/yelp_all_uniform_mlm_seed_2021_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt"
