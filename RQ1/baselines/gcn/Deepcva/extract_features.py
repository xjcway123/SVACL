from Deepcvahelper.tokenizer import gen_tok_pattern, extract_features
import collections
import pathlib as Path
import sys
import os
import pandas as pd

path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn

code_token_pattern = gen_tok_pattern()
vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
tokenizer = vectorizer.build_analyzer()


# feature_type = 'code'
# feature_scope = 'hc' #hunk and context
# token = 'word'

def count_corpus(tokens):  # @save
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, data, tokenizer, min_freq=0, max_vocab=10000, max_len=1024, reserved_tokens=["<pad>", "<unk>"]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        tokens = self.tokenizer(data)
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        self._token_freqs = self._token_freqs[:max_vocab - 2]
        # unk token index 1 , pad token index 0
        self.idx_to_token = reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < self.min_freq or len(self.idx_to_token) >= self.max_vocab:
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
        return [self.idx_to_token[index] for index in indices if index != self.pad]

    def to_idxs(self, sentence):
        try:
            tokens = self.tokenizer(sentence)
        except:  # input is empty
            tokens = []
        idxs = []
        for i in tokens:
            if i in self.token_to_idx.keys():
                idxs.append(self.token_to_idx[i])
            else:
                idxs.append(self.token_to_idx["<unk>"])
        idxs = idxs + [self.token_to_idx["<pad>"]] * (self.max_len - len(idxs))
        idxs = idxs[:self.max_len]
        return idxs

    @property
    def unk(self):  # unk token
        return 1

    @property
    def pad(self):  # pad token
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def extract_code(code: str, lines):
    codelines = code.splitlines()
    # print(codelines)
    # print(lines)
    try:
        lines = [int(i) - 1 for i in lines.split()]
    except:  # no lines
        lines = []
    return "\n".join([codelines[i] for i in lines])


# read the data just from train
data = pd.read_csv("E:\\xjc\\SOTitlePlus\\baselines\\FVA-master\\FVA-master\\Data\\new_data.xlsx")
data.fillna("", inplace=True)  # change nan to ""
print(data.count())
train_data = data[data["partition"] == "train"]
blaming_func_after = train_data["blaming_func_after"].tolist()
blaming_func_before = train_data["blaming_func_before"].tolist()
train_data = "\n".join(blaming_func_after + blaming_func_before)
# print(train_data)
vocab = Vocab(train_data, tokenizer, min_freq=0, max_vocab=10000, max_len=1024)
# print(list(vocab.token_to_idx.items())[:10])
# print(vocab.token_freqs)
# print(len(vocab.token_freqs))

# print(vocab.to_idxs(s))

# print(data["blaming_add_lines"])
data["pre_code_feature"] = data[["blaming_delete_lines", "blaming_func_before"]].apply(
    lambda r: vocab.to_idxs(
        extract_code(r.blaming_func_before, r.blaming_delete_lines)
    ), axis=1
)
data["pre_context_feature"] = data[["pre_context"]].apply(
    lambda r: vocab.to_idxs(
        r.pre_context
    ), axis=1
)
data["cur_code_feature"] = data[["blaming_add_lines", "blaming_func_after"]].apply(
    lambda r: vocab.to_idxs(
        extract_code(r.blaming_func_after, r.blaming_add_lines)
    ), axis=1
)
data["cur_context_feature"] = data[["cur_context"]].apply(
    lambda r: vocab.to_idxs(
        r.cur_context
    ), axis=1
)
data["pre_code_feature"] = data["pre_code_feature"].apply(
    lambda r: " ".join([str(i) for i in r])
)
data["pre_context_feature"] = data["pre_context_feature"].apply(
    lambda r: " ".join([str(i) for i in r])
)
data["cur_code_feature"] = data["cur_code_feature"].apply(
    lambda r: " ".join([str(i) for i in r])
)
data["cur_context_feature"] = data["cur_context_feature"].apply(
    lambda r: " ".join([str(i) for i in r])
)
feature_path = gcn.get_dir(gcn.data_dir() / "deepcva") / ("deepcva_feature.csv")
data.to_csv(feature_path, index=False)
