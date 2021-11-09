#Get reiwev's texts, split into train, dev and convert data for fairseq MLM training. The last step may take a lot of time.
bash prepare.sh 
bash train_dev.sh 
bash mlm_data.sh
