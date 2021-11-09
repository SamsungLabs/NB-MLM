from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict
import spacy
from scipy.special import softmax

from transformers import *


def calcNikFi(lp):
    fi = np.zeros((lp.shape[1],))
    for i in range(lp.shape[0]):
        for j in range(lp.shape[0]):
            if i != j:
                fi = np.maximum(fi, lp[i] - lp[j])
    return fi


class Masking:
    def __init__(self, NB, tokenizer, temperature, alpha=0.1, min_df=100, binary=True, save_rate=0, file=None,
                 max_seq_length=128, ngram_range=(1, 1) ):
        """ NB-naive bayes constructor
            tokenizer - tokenizer for NB
            temperature - temperature for softmax
            alpha - NB regularization
            min_df - token must occur at least min_df times to stay in the dictionary.
            binary - count all appearances in the text of the word or 0,1 (yes, no in text).
            save_rate - every save_rate calls of GenerateMask output examples from the batch (if 0 output absent).
            file - file for masking exaples. if None output goes to stdout"""
        self.max_seq_length = max_seq_length
        self.nb = NB(alpha=alpha)
        self.cv = CountVectorizer(min_df=min_df, binary=binary, tokenizer=tokenizer, ngram_range=ngram_range)
        self.temperature = temperature
        self.counter = 0
        self.save_rate = save_rate
        self.file = file

    def CalculateMetaInformation(self):
        lp = self.ob.feature_log_prob_
        # dif = lp[0] - lp[1]
        dif = np.abs(calcNikFi(lp))
        invert = {v: k for k, v in self.cv.vocabulary_.items()}
        l = {k: v for v, k in zip(dif, range(len(dif)))}
        self.d = defaultdict(float, {invert[k]: v for k, v in l.items()})

    def Initialize(self, path_to_texts, path_to_labels):
        """ 
        """
        texts = []
        labels = []
        i = 0
        with open(path_to_texts, 'r') as f:
            for text in f:
                texts.append(text)

        with open(path_to_labels, 'r') as f:
            for label in f:
                labels.append([int(label)])

        A = self.cv.fit_transform(texts)
        self.ob = self.nb.fit(A, np.array(labels).reshape(-1, ))
        self.CalculateMetaInformation()

    def Testing(self, path_to_texts, path_to_labels):
        """return accuracy score of nb on imdb. check how good your tokens like features"""
        texts = []
        labels = []
        i = 0

        with open(path_to_texts, 'r') as f:
            for text in f:
                texts.append(text)

        with open(path_to_labels, 'r') as f:
            for label in f:
                labels.append([int(label)])

        A = self.cv.transform(texts)
        y_pred = self.nb.predict(A)
        print(accuracy_score(y_pred, np.array(labels).reshape(-1, )))

    def ScoreTokens(self, tokens):
        return [abs(self.d[a]) for a in tokens]

    def GenerateMask(self, batch_of_tokens, tokens_borders, ids):
        """ generate mask by using:
        batch_of_tokens - list of lists.
            The external list contains examples from batches, the internal list contains example tokens
        tokens_borders - list of list of tuples.
            The external list contains examples from the batch,
            the internal list contains token boundaries in the ids matrix.
        ids - ids matrix. 
        return the matrix with the same dimensions as the ids matrix.
        """
        if self.save_rate != 0:
            self.counter = (self.counter + 1) % self.save_rate
        shape = ids.shape
        mask = np.zeros(shape)
        for i in range(shape[0]):
            tokens = batch_of_tokens[i]
            scores = self.ScoreTokens(tokens)  # [self.d[a] for a in tokens]
            num = int(0.15 * len(tokens))
            masked_tokens = np.random.choice(range(len(scores)), size=num,
                                             p=softmax(np.array(scores) / self.temperature),
                                             replace=False)
            sent = []
            for j in range(len(scores)):
                if j in masked_tokens:
                    mask[i][tokens_borders[i][j][0]:tokens_borders[i][j][1]] = 1
                    sent.append("<<" + batch_of_tokens[i][j] + ">>")
                else:
                    sent.append(batch_of_tokens[i][j])
            if (self.save_rate != 0) and (self.counter == 0):
                if self.file is not None:
                    print(" ".join(sent), file=self.file)
                else:
                    print(" ".join(sent))
        return mask


class IdsLevelMasking(Masking):
    def ToTokenizer(self, TextToIds):
        def tokenizer(text):
            return list(map(str, TextToIds(text)[1:-1]))

        return tokenizer

    def __init__(self, NB, TextToIds, TextToIdsForMask, temperature, alpha=0.1, min_df=50, binary=True, save_rate=0,
                 file=None, ngram_range=(1, 1)):
        """
        Every thing like in base class but instead of tokenizer
        take function which convert text to ids with extra tokens.
        ids must be on the list
        """
        self.tokenizer = self.ToTokenizer(TextToIds)
        self.tti = TextToIdsForMask
        self.tokenizer_mask = self.ToTokenizer(TextToIdsForMask)
        super().__init__(NB, self.tokenizer, temperature, alpha, min_df, binary, save_rate, file, ngram_range)

    def Mask(self, text):
        """take text return mask"""
        tokens_a = self.tokenizer_mask(text)
        return self.GenerateMask([tokens_a], [[(i + 1, i + 2) for i in range(len(tokens_a))]],
                                 np.array([self.tti(text)]))


def WordsLimits(words, ids, ids2str, sent_lims):
    ends = []
    currentword = 0
    prevlen = 0
    nakop = sent_lims[0]
    for i in range(1, len(ids) + 1):
        st = ids2str(ids[:i]).lower().replace(' ', '')
        # print(st,  nakop)
        while st.startswith(nakop + words[currentword].replace(' ', '')):
            ends.append(i)
            nakop += words[currentword].replace(' ', '')
            currentword += 1
            if currentword == len(words):
                break
        if currentword == len(words):
            break

    starts = []
    currentword = len(words) - 1
    nakop = sent_lims[1]
    for i in range(len(ids) - 1, -1, -1):
        st = ids2str(ids[i:]).lower().replace(' ', '')
        # print(st[:-prevend if prevend!=0 else len(st)],  words[currentword])
        while st.endswith(words[currentword].replace(' ', '') + nakop):
            starts.append(i)
            nakop = words[currentword].replace(' ', '') + nakop
            currentword -= 1
            if currentword == -1:
                break
        if currentword == -1:
            break

    lims = list(zip(starts[::-1], ends))

    assert len(words) == len(lims)
    for c in lims:
        assert c[0] < c[1]

    return lims


def PreprocessText(text):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # return tokenizer.decode(tokenizer.encode(text), clean_up_tokenization_spaces=False)
    max_seq_length = 128
    tokens_a = tokenizer.tokenize(text)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
        """fl=False
        while(tokens_a[-1].startswith('##')):
            fl=True
            tokens_a.pop(-1)
        if(fl):
            tokens_a.pop(-1)"""

    tokens_a = tokenizer.convert_tokens_to_ids(tokens_a)
    return tokenizer.decode(tokens_a, clean_up_tokenization_spaces=False)


def WordTokenizer(text):
    # return PreprocessText(text).split()
    return [a.text for a in nlp(text.lower()) if a.text.replace(' ', '') != '']


class WordLevelMasking(Masking):
    def __init__(self, NB, TextToIds, ids2str, temperature, alpha=0.1, min_df=100, binary=True, save_rate=0, file=None,
                 sentence_lims=('', ''), preprocess_text=False):
        """
        Every thing like in base class but instead of tokenizer
        take function which convert text to ids with extra tokens
        """
        self.tti = TextToIds
        self.i2s = ids2str
        self.sl = sentence_lims
        self.preprocess_text = preprocess_text
        super().__init__(NB, WordTokenizer, temperature, alpha, min_df, binary, save_rate, file)

    def Mask(self, text):
        """take text return mask"""
        if self.preprocess_text:
            text = PreprocessText(text)
        tokens_a = WordTokenizer(text.lower())

        return self.GenerateMask([tokens_a], [WordsLimits(tokens_a, self.tti(text), self.i2s, self.sl)],
                                 np.array([self.tti(text)]))


class MutualInformationMasking(Masking):

    def ToTokenizer(self, TextToIds):
        def tokenizer(text):
            return list(map(str, TextToIds(text)[1:-1]))

        return tokenizer

    def __init__(self, NB, TextToIds, TextToIdsForMask, temperature, alpha=0.1, min_df=50, binary=True, save_rate=0,
                 file=None):
        """Every thing like in base class but instead of tokenizer take function which convert text to ids with extra tokens.
        ids must be on the list"""

        self.tokenizer = self.ToTokenizer(TextToIds)
        self.tti = TextToIdsForMask
        self.tokenizer_mask = self.ToTokenizer(TextToIdsForMask)
        super().__init__(NB, self.tokenizer, temperature, alpha, min_df, binary, save_rate, file)
        self.cv = CountVectorizer(min_df=min_df, binary=binary, tokenizer=self.tokenizer, ngram_range=(1, 2))

    def CalculateMetaInformation(self):
        one_word = []
        two_word = []
        for k in self.cv.vocabulary_.keys():
            if ' ' not in k:
                one_word.append(k)
            else:
                two_word.append(k)
        self.right_context = {}

        for w in one_word:
            self.right_context[w] = {ww.split()[1]: self.ob.feature_count_[:, self.cv.vocabulary_[ww]] for ww in
                                     two_word if (ww.split()[0] == w)}

        self.left_context = {}
        for w in one_word:
            self.left_context[w] = {ww.split()[0]: self.ob.feature_count_[:, self.cv.vocabulary_[ww]] for ww in
                                    two_word if (ww.split()[1] == w)}

    def left(self, fw, sw):
        if fw not in self.right_context:
            return 0

        elif sw not in self.right_context[fw]:
            return 0

        else:
            acc_r = np.array([0., 0.])
            for v in (self.right_context[fw]).values():
                acc_r += v

            eps = 1e-15

            return (np.log(self.right_context[fw][sw] + eps) - np.log(acc_r + eps) - (
                    np.log(self.right_context[fw][sw].sum()) - np.log(acc_r.sum()))).max()

    def right(self, fw, sw):
        if sw not in self.left_context:
            return 0

        elif fw not in self.left_context[sw]:
            return 0

        else:
            acc_l = np.array([0., 0.])
            for v in self.left_context[sw].values():
                acc_l += v

            eps = 1e-15

            return (np.log(self.left_context[sw][fw] + eps) - np.log(acc_l + eps) - (
                    np.log(self.left_context[sw][fw].sum()) - np.log(acc_l.sum()))).max()

    def ScoreTokens(self, tokens):
        rez = []
        words = tokens[:]
        for i in range(len(words)):
            if i == 0:
                rez.append(self.right(words[i], words[i + 1]))
            elif i == len(words) - 1:
                rez.append(self.left(words[i - 1], words[i]))
            else:
                rez.append(abs(max(self.right(words[i], words[i + 1]), self.left(words[i - 1], words[i]))))
        return rez

    def Mask(self, text):
        """take text return mask"""
        tokens_a = self.tokenizer_mask(text)
        return self.GenerateMask([tokens_a], [[(i + 1, i + 2) for i in range(len(tokens_a))]],
                                 np.array([self.tti(text)]))


class SqrtMasking(Masking):
    def ToTokenizer(self, TextToIds):
        def tokenizer(text):
            return list(map(str, TextToIds(text)[1:-1]))

        return tokenizer

    def __init__(self, NB, TextToIds, TextToIdsForMask, save_rate=0, file=None):
        """Every thing like in base class but instead of tokenizer take function which convert text to ids with extra tokens.
        ids must be on the list"""

        self.tokenizer = self.ToTokenizer(TextToIds)
        self.tti = TextToIdsForMask
        self.tokenizer_mask = self.ToTokenizer(TextToIdsForMask)
        super().__init__(NB, self.tokenizer, 1, 0.1, 0, False, save_rate, file)

    def CalculateMetaInformation(self):
        lp = self.ob.feature_count_
        dif = -np.log(lp.sum(axis=0))
        invert = {v: k for k, v in self.cv.vocabulary_.items()}
        l = {k: v for v, k in zip(dif, range(len(dif)))}
        self.d = defaultdict(float, {invert[k]: v for k, v in l.items()})

    def ScoreTokens(self, tokens):
        return [0.5 * self.d[w] for w in tokens]

    def Mask(self, text):
        """take text return mask"""
        tokens_a = self.tokenizer_mask(text)
        return self.GenerateMask([tokens_a], [[(i + 1, i + 2) for i in range(len(tokens_a))]],
                                 np.array([self.tti(text)]))

#nlp = spacy.load('en_core_web_sm')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
