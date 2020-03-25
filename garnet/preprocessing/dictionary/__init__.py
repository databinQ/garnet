# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/1/15 9:47
"""

import codecs
from typing import Iterable
from collections import Counter

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


class Dictionary(object):
    def __init__(self, corpus=None, min_count=1, special_tokens=[]):
        self.itot = dict()
        self.ttoi = None
        self.special_init()
        for token in special_tokens:
            self.itot[len(self.itot)] = token
        if corpus is not None:
            self.init_with_corpus(corpus=corpus, min_count=min_count)
        self.refresh()

    def __len__(self):
        return len(self.itot)

    def __getitem__(self, key):
        return self.ttoi[key]

    def __contains__(self, key):
        return key in self.ttoi

    def id2token(self, index):
        if isinstance(index, Iterable):
            return [self.itot[id] for id in index]
        else:
            return self.itot[index]

    def token2id(self, token):
        if isinstance(token, Iterable):
            return [self.ttoi[t] for t in token]
        else:
            return self.ttoi[token]

    def special_init(self):
        assert len(self.itot) == 0, "Special initialization must be processed before any other initial function"
        self.itot[0] = PAD
        self.itot[1] = BOS
        self.itot[2] = EOS
        self.itot[3] = UNK

    def init_with_corpus(self, corpus, min_count=1):
        """
        Corpus is comprised of sentences, and each sentence must be segmented.
        """
        total = []
        for s in corpus:
            total.extend(s)
        counter = Counter(total)
        for token, count in counter.items():
            if count >= min_count:
                self.itot[len(self.itot)] = token

    def refresh(self):
        self.ttoi = {index: token for token, index in self.itot.items()}

    def extend(self, tokens):
        for token in tokens:
            self.itot[len(self.itot)] = token
        self.refresh()

    @classmethod
    def load(cls, dict_path, sep=None):
        with codecs.open(dict_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if sep is None:
            corpus = [t.strip() for t in lines if t.strip()]
        else:
            corpus = [t.split(sep)[0].strip() for t in lines if t.strip()]
        dic = cls()
        dic.itot = {i: token for i, token in enumerate(corpus)}
        dic.refresh()
        return dic

    def save(self, dict_path):
        with codecs.open(dict_path, "w", encoding="utf-8") as f:
            token_index = sorted(list(self.itot.items()), key=lambda x: x[0], reverse=False)
            plain_text = "\n".join(list(zip(*token_index))[1])
            f.write(plain_text)
