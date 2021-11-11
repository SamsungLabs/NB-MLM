#download imdb
cd ./prepare_data
bash download.sh imdb
#download unlabeled part of imdb
cd ./../imdb_experiments
bash download_unlabeled_data.sh
#download yelp
cd ./../yelp_experiments/prepare_data/
bash download.sh
#Download amazon. It may take a lot of time.
cd ./../../amazon_new/
bash ./download.sh
