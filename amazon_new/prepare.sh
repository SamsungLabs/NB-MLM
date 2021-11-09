./jq-linux64 --raw-output '.reviewText | select(length >= 501)' < aggressive_dedup.json > reviews_min500chars.txt
