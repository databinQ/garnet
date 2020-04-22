# coding: utf-8

"""
@File   : vocabulary.py
@Author : garnet
@Time   : 2020/4/14 20:48
"""

import json
import codecs
import pathlib

from . import StateUnit
from ...utils.io import safe_save
from ...utils.io import safe_save_json
from ...utils.path import check_suffix

PAD = '[PAD]'
UNK = '[UNK]'
SOS = '[SOS]'  # start of sentence
EOS = '[EOS]'  # end of sentence
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


class Vocabulary(StateUnit):
    def __init__(self, special_tokens=None, ignore_case=False):
        """
        :param special_tokens: list. e.g. ['<spt1>', '<spt2>', '<spt3>']
        """
        super().__init__()
        self.ignore_case = ignore_case

        self._vocab = {
            PAD: 0,
            UNK: 1,
            SOS: 2,
            EOS: 3,
        }
        if special_tokens is not None:
            for token in special_tokens:
                self._vocab[token] = len(self._vocab)

        self._vocab_rev = dict()
        self._update_vocab_rev()

    def _update_vocab_rev(self):
        vocab_rev = {v: k for k, v in self._vocab.items()}
        self._vocab_rev = vocab_rev
        return vocab_rev

    def fit(self, input_: list):
        """
        :param input_: list of tokens, or list of lists containing tokens, inner list stands for single sentence
        """
        assert len(input_) > 0, "Input must can not be empty"
        if not isinstance(input_[0], str):
            input_ = [token.lower() if self.ignore_case else token for l in input_ for token in l]
        token_set = set(input_)

        init_map_len, token_num = len(self._vocab), len(token_set)
        new_vocab = dict(list(zip(list(token_set), range(init_map_len, init_map_len + len(token_set)))))
        self._vocab.update(new_vocab)
        self._update_vocab_rev()

        super().fit(input_)
        return self

    def __getitem__(self, item):
        return self._vocab.get(item) or self._vocab[UNK]

    def __contains__(self, item):
        return True if item in self._vocab else False

    def word2id(self, word):
        return self[word]

    def id2word(self, id_):
        return self._vocab_rev.get(id_) if id_ in self._vocab_rev else UNK

    def transform(self, input_: list):
        if self.ignore_case:
            input_ = [token.lower() for token in input_]
        return [self[token] for token in input_]

    def reverse_transform(self, input_: list):
        return [self.id2word(index) for index in input_]

    @property
    def vocab(self):
        return self._vocab

    def to_txt(self, file_path):
        sequential_tokens = list(zip(*sorted(self._vocab.items(), key=lambda x: x[1], reverse=False)))[0]
        safe_save(file_path, '\n'.join(sequential_tokens), suffix='txt')

    def to_json(self, file_path):
        safe_save_json(file_path, self._vocab)

    @classmethod
    def read_txt(cls, file_path, encoding='utf-8', ignore_case=False):
        path = pathlib.Path(file_path)
        with codecs.open(path, 'r', encoding=encoding) as f:
            tokens = [token.strip() for token in f.readlines()]
            vocab = dict([(token, i) for i, token in enumerate(tokens)])

        instance = cls(ignore_case=ignore_case)
        instance._vocab = vocab
        instance._update_vocab_rev()
        return instance

    @classmethod
    def read_json(cls, file_path, encoding='utf-8', ignore_case=False):
        path = pathlib.Path(file_path)
        check_suffix(path, suffix='json')
        with codecs.open(path, 'r', encoding=encoding) as f:
            vocab = json.load(f)

        instance = cls(ignore_case=ignore_case)
        instance._vocab = vocab
        instance._update_vocab_rev()
        return instance


class BertVocabulary(Vocabulary):
    def __init__(self, dict_path, ignore_case=False):
        self.ignore_case = ignore_case

        self._vocab = dict()
        self._vocab_rev = dict()
        self.fit(dict_path)

    def fit(self, file_path):
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            tokens = [token.strip() for token in f.readlines()]
            self._vocab = dict([(token, i) for i, token in enumerate(tokens)])
            self._update_vocab_rev()
        super().fit(file_path)
        return self
