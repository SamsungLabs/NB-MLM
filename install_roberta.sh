#git clone https://github.com/nvanva/fairseq
#cd fairseq
#pip install --editable .
cd ./fairseq-mlm-pretrain
wget http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar xvzf roberta.base.tar.gz

mkdir vocab
cd vocab
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
