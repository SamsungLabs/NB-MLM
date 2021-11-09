#download imdb
cd ./prepare_data
bash download.sh imdb
#download yelp
cd ./../yelp_experiments/prepare_data/
bash download.sh
#Download amazon. It may take a lot of time.
cd ./../../amazon_new/
bash ./download.sh
