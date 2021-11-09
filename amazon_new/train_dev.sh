head -n 5000 reviews_min500chars.txt > dev_head.txt
tail -n 5000 reviews_min500chars.txt > dev_tail.txt
cat dev_head.txt dev_tail.txt > dev.texts

tail -n +5001 reviews_min500chars.txt > tmp_train.txt
head --lines=-5000 tmp_train.txt > train.texts
rm tmp_train.txt
