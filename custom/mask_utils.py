import scipy.special
import numpy as np
def bayes_mask(d, temperature, item, sz, num_mask):
    token_list = np.copy(item).tolist()
    scores = [d[a] for a in token_list]
    #print(scores)            
    return np.random.choice(sz, num_mask, p = scipy.special.softmax(np.array(scores)/temperature), replace=False)

def right(lc, al, fw, sw):
    fw, sw = str(fw), str(sw)
    if(sw not in lc):
        return 0

    elif(fw not in lc[sw]):
        return 0

    else:
        acc_l = al[sw]

        eps = 1e-15

        return (np.log(lc[sw][fw]+eps) - np.log(acc_l+eps) - (np.log(lc[sw][fw].sum()) - np.log(acc_l.sum()))).max()

def left(rc, ar, fw, sw):
    fw, sw = str(fw), str(sw)
    if(fw not in rc):
        return 0

    elif(sw not in rc[fw]):
        return 0

    else:
        acc_r = ar[fw]

        eps = 1e-15

        return (np.log(rc[fw][sw]+eps) - np.log(acc_r+eps) - (np.log(rc[fw][sw].sum()) - np.log(acc_r.sum()))).max()
        
def ScoreTokens(d, tokens):
    rc, lc, ar, al = d
    rez = []
    words = tokens[:]
    for i in range(len(words)):
        if(i==0):
            rez.append(right(lc, al, words[i], words[i+1]))
        elif(i==len(words)-1):
            rez.append(left(rc, ar, words[i-1], words[i]))
        else:
            rez.append(abs(max(right(lc, al, words[i], words[i+1]), left(rc, ar, words[i-1], words[i]))))
    return rez  

def ScoreTokensMin(d, tokens):
    rc, lc, ar, al = d
    rez = []
    words = tokens[:]
    for i in range(len(words)):
        if(i==0):
            rez.append(right(lc, al, words[i], words[i+1]))
        elif(i==len(words)-1):
            rez.append(left(rc, ar, words[i-1], words[i]))
        else:
            rez.append(abs(min(right(lc, al, words[i], words[i+1]), left(rc, ar, words[i-1], words[i]))))
    return rez


def inform_mask(d, temperature, item, sz, num_mask):
    token_list = np.copy(item).tolist()
    scores = ScoreTokens(d, token_list)
    return np.random.choice(sz, num_mask, p = scipy.special.softmax(np.array(scores)/temperature), replace=False)

def inform_mask_min(d, temperature, item, sz, num_mask):
    token_list = np.copy(item).tolist()
    scores = ScoreTokensMin(d, token_list)
    #print(scores)
    return np.random.choice(sz, num_mask, p = scipy.special.softmax(np.array(scores)/temperature), replace=False)


def multi_bayes_mask(d, temperature, item, sz, num_mask):
    token_list = np.copy(item).tolist()
    
    indexes = list(range(len(token_list)))
    
    ngram_range = d[0]
    d = d[1]
    ngrams = []
    ngrams_intervals = []

    for k in range(ngram_range[0], ngram_range[1]+1):
        ngrams_intervals += list(zip(indexes, indexes[k-1:]))

    for interval in ngrams_intervals:
        ngrams.append(token_list[interval[0] : interval[1] + 1])

    ngrams = [' '.join(map(str, x)) for x in ngrams]
    scores = [d[a] for a in ngrams]
    #print(list(zip(scores, ngrams)))
    probs = scipy.special.softmax(np.array(scores)/temperature)

    masked = [False]*len(token_list)
    num_masked = 0
    current_number_ngrams = len(ngrams)

    while(num_masked < num_mask):
        ind = np.random.choice(current_number_ngrams, 1, p = probs, replace=False)[0]
        interval = ngrams_intervals[ind]

        for i in range(interval[0], interval[1]+1):
            if(not masked[i]):
                num_masked += 1
                masked[i] = True

        ngrams_intervals = ngrams_intervals[:ind] + ngrams_intervals[ind+1:]
        probs = np.hstack([probs[:ind], probs[ind+1:]])
        probs = probs/probs.sum()
        current_number_ngrams -= 1
        

    return np.arange(len(token_list))[masked]
