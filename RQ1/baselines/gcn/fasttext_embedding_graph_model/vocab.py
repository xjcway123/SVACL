from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pathlib as Path
import sys
import os
import pandas as pd

from transformers import  AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn
from gcn.codebert.model import CodebertModel
import numpy as np
def gen_tok_pattern():
    single_toks = [
        "<=",
        ">=",
        "<",
        ">",
        "\\?",
        "\\/=",
        "\\+=",
        "\\-=",
        "\\+\\+",
        "--",
        "\\*=",
        "\\+",
        "-",
        "\\*",
        "\\/",
        "!=",
        "==",
        "=",
        "!",
        "&=",
        "&",
        "\\%",
        "\\|\\|",
        "\\|=",
        "\\|",
        "\\$",
        "\\:",
    ]
    single_toks = "(?:" + "|".join(single_toks) + ")"
    word_toks = "(?:[a-zA-Z0-9]+)"
    return single_toks + "|" + word_toks


# Extract features
def extract_features(start_n_gram, end_n_gram, token_pattern=None, vocabulary=None):
    return TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 1),
        use_idf=False,
        min_df=0.0,
        max_df=1.0,
        max_features=10000,
        norm=None,
        smooth_idf=False,
        lowercase=False,
        token_pattern=token_pattern,
        vocabulary=vocabulary,
    )


def get_tokenizer(vectorize=False):
    code_token_pattern = gen_tok_pattern()
    vectorizer = extract_features(
        start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern
    )
    if vectorize:
        return vectorizer
    return vectorizer.build_analyzer()
code_token_pattern = gen_tok_pattern()
vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
tokenizer = vectorizer.build_analyzer()

#feature_type = 'code'
#feature_scope = 'hc' #hunk and context
# token = 'word'

def count_corpus(tokens):  #@save
        return collections.Counter(tokens)

class Vocab:
    def __init__(self, data,tokenizer, min_freq=0,max_vocab=10000, max_len = 512,reserved_tokens=["<unk>","<pad>"]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        tokens = self.tokenizer(data)
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        self._token_freqs = self._token_freqs[:max_vocab-2]

        self.idx_to_token = reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < self.min_freq or len(self.idx_to_token)>=self.max_vocab:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices if index!=self.pad]
    def to_idxs(self,sentence):
        try:
            tokens = self.tokenizer(sentence)
        except:# input is empty
            tokens = []
        idxs = []
        for i in tokens:
            if i in self.token_to_idx.keys():
                idxs.append(self.token_to_idx[i])
            else:
                idxs.append(self.token_to_idx["<unk>"])
        idxs = idxs + [self.token_to_idx["<pad>"]]*(self.max_len-len(idxs))
        idxs = idxs[:self.max_len]
        return idxs
    @property
    def unk(self):  # unk is 0
        return 0
    @property
    def pad(self):  # pad is 1
        return 1
    @property
    def token_freqs(self):
        return self._token_freqs