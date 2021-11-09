cd ./prepare_data
#1. Convert data for fairseq MLM training
bash mlm_data.sh imdb all 
#2. Convert data for fairseq classifier training
bash clf_data.sh imdb
#3. Create Naive bayes weighted token scores
bash token_scores.sh imdb 50 all
bash token_scores.sh imdb 10 all
