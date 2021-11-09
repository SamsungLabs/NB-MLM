# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary
from fairseq.data import BaseWrapperDataset, LRUCacheDataset

from collections import defaultdict
import pandas as pd
import time
import os
from os import path
import json

from custom.mask_utils import multi_bayes_mask, bayes_mask, inform_mask, inform_mask_min


class MaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            vocab: Dictionary,
            pad_idx: int,
            mask_idx: int,
            return_masked_tokens: bool = False,
            seed: int = 1,
            mask_prob: float = 0.15,
            leave_unmasked_prob: float = 0.1,
            random_token_prob: float = 0.1,
            freq_weighted_replacement: bool = False,
            mask_whole_words: torch.Tensor = None,
            temperature: float = 1.0,
            path_to_scores: str = './',
            log_mask: bool = False,
            log_dir: str = './',
            mask_type: str = 'bayes'
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        self.temperature = temperature
        self.mask_whole_words = mask_whole_words

        #if mask_type == "inform":
            #self.mask_func = inform_mask
        if(mask_type=='inform')or(mask_type=='inform_min'):
            print("INFORM MIN")
            if(mask_type=='inform'):
                self.mask_func = inform_mask
            else:
                self.mask_func = inform_mask_min

            with open(path.join(path_to_scores, 'ar.json')) as json_file:
                ar_j = json.load(json_file)
            ar = {k: np.array(v) for k, v in ar_j.items()}
            with open(path.join(path_to_scores, 'al.json')) as json_file:
                al_j = json.load(json_file)
            al = {k: np.array(v) for k, v in al_j.items()}
            with open(path.join(path_to_scores, 'rc.json')) as json_file:
                rc_j = json.load(json_file)
            rc = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in rc_j.items()}
            with open(path.join(path_to_scores, 'lc.json')) as json_file:
                lc_j = json.load(json_file)
            lc = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in lc_j.items()}
            self.token_scores = (rc, lc, ar, al)
        elif(mask_type=='multi_bayes'):
            self.mask_func = multi_bayes_mask
            if path_to_scores is not None:
                df = pd.read_csv(path.join(path_to_scores, 'token_scores.csv'))[["keys", "scores"]]
                d = defaultdict(float, {str(k): float(v) for k, v in zip(df["keys"], df["scores"])})
                ngram_lens = [len(x.split()) for x in d.keys()]
                self.token_scores = ((min(ngram_lens), max(ngram_lens)), d)
            else:
                self.token_scores = ((1, 1), defaultdict(float, {}))


        else:
            self.mask_func = bayes_mask
            if path_to_scores is not None:
                #print( path.join(path_to_scores, 'token_scores.csv') )
                #print( '---------------------------------------' )
                #print(  pd.read_csv(path.join(path_to_scores, 'token_scores.csv')).head() )
                #print('---------------------------------------')

                df = pd.read_csv(path.join(path_to_scores, 'token_scores.csv'))[["keys", "scores"]]
                self.token_scores = defaultdict(float, {k: float(v) for k, v in zip(df["keys"], df["scores"])})
                print(self.token_scores)
            else:
                self.token_scores = defaultdict(float, {})
        #if not path.exists(log_dir):
        #    os.mkdir(log_dir)

        self.filename = path.join(log_dir, str(time.time()) + '.log')
        # self.file = open(self.filename, 'w')
        self.log_mask = log_mask

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    #    def __del__(self):
    #        self.file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            # import os
            # print(os.listdir())
            item = self.dataset[index]
            # print(item)
            # exit(0)
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz + np.random.rand()
            )
            # token_list = np.copy(item).tolist()
            # scores = [self.token_scores[a] for a in token_list]
            # print(self.temperature, scipy.special.softmax(np.array(scores)/self.temperature) )
            # exit(0)

            # mask[np.random.choice(sz, num_mask, p=scipy.special.softmax(np.array(scores) / self.temperature),
            #                     replace=False)] = True
            mask[self.mask_func(self.token_scores, self.temperature, item, sz, num_mask)] = True
            #if self.log_mask:
            #    token_list = np.copy(item).tolist()
            #    with open(self.filename, 'a') as file:
            #        print(index, file=file)
            #        print(' '.join(map(str, token_list)), file=file)
            #        print(' '.join(map(str, mask.astype(np.int32).tolist())), file=file)
            # print(scores)
            # exit(0)

            # mask[np.random.choice(sz, num_mask, replace=False)] = True

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)
