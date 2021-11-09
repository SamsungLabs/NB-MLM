import pandas as pd
import random

def process(texts):
    return [' '.join(txt.split('\n')).replace('\\n', ' ').replace('\r','').replace('\\','') for txt in texts]

def process_labels(labels):
    return [str(l) for l in labels]






df = pd.read_csv('./../DATA/raw/yelp_review_polarity_csv/train.csv', names=['label','text'], header=None)
train_texts = df["text"].tolist()
train_labels = df["label"].tolist()

random.seed(142)
random.shuffle(train_texts)

random.seed(142)
random.shuffle(train_labels)




dev_texts = []
dev_labels = []
pos_count = 0
neg_count = 0
inds = []


for i in range(len(train_texts)-1, -1, -1):
    if(train_labels[i] == 1):
        if(pos_count < 10000):
            dev_texts.append(train_texts[i])
            dev_labels.append(train_labels[i])
            pos_count += 1
            inds.append(i)
            
    if(train_labels[i] == 2):
        if(neg_count < 10000):
            dev_texts.append(train_texts[i])
            dev_labels.append(train_labels[i])
            neg_count += 1
            inds.append(i)
            
    if(pos_count==10000)and(neg_count==10000):
        break
        
inds = set(inds)

train_texts = [train_texts[i] for i in range(len(train_texts)) if i not in inds]
train_labels = [train_labels[i] for i in range(len(train_labels)) if i not in inds]


df = pd.read_csv('./../DATA/raw/yelp_review_polarity_csv/test.csv', names=['label','text'], header=None)
test_texts = df["text"].tolist()
test_labels = df["label"].tolist()

data_dir = 'yelp-clf-dev2'



with open("./../DATA/"+data_dir+"/train.texts", "w", encoding="utf8") as f:
    f.write('\n'.join(process(train_texts)))

with open("./../DATA/"+data_dir+"/train.labels", "w", encoding="utf8") as f:
    f.write('\n'.join(process_labels(train_labels)))




with open("./../DATA/"+data_dir+"/dev.texts", "w", encoding="utf8") as f:
    f.write('\n'.join(process(dev_texts)))

with open("./../DATA/"+data_dir+"/dev.labels", "w", encoding="utf8") as f:
    f.write('\n'.join(process_labels(dev_labels)))




with open("./../DATA/"+data_dir+"/test.texts", "w", encoding="utf8") as f:
    f.write('\n'.join(process(test_texts)))

with open("./../DATA/"+data_dir+"/test.labels", "w", encoding="utf8") as f:
    f.write('\n'.join(process_labels(test_labels)))


