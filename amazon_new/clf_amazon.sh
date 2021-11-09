if [ -z "$1" ]; then
    cp -r ./../imdb_experiments/DATA/imdb-clf ./../amazon_experiments/DATA/amazon-clf
    cp -r ./../imdb_experiments/DATA/imdb-clf-bin ./../amazon_experiments/DATA/amazon-clf-bin
else
     cp -r ./../yelp_experiments/DATA/yelp-clf ./../amazonYelp_experiments/DATA/amazonYelp-clf
     cp -r ./../yelp_experiments/DATA/yelp-clf-bin ./../amazonYelp_experiments/DATA/amazonYelp-clf-bin
fi
