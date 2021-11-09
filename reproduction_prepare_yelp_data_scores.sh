cd ./yelp_experiments/prepare_data
#Convert data for fairseq MLM training
bash mlm_data.sh 
bash mlm_data.sh all
#Convert data for fairseq classifier training
bash clf_data.sh
#Create Naive bayes weighted token scores
bash token_scores.sh 50
