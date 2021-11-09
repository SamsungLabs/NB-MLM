if [ -z "$1" ]; then
    cp -r ./../imdb_experiments/SCORES/ ./../amazon_experiments/
else
    cp -r ./../yelp_experiments/SCORES/ ./../amazonYelp_experiments/
fi
