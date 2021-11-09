#!/bin/bash/
mkdir -p ./../DATA/raw
wget -O ./../DATA/raw/yelp.tgz https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz
tar zxvf ./../DATA/raw/yelp.tgz -C ./../DATA/raw
