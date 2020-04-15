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
from ...utils.io import safe_save, safe_save_json
from ...utils.path import check_suffix

PAD = '[PAD]'
UNK = '[UNK]'
SOS = '[SOS]'  # start of sentence
EOS = '[EOS]'  # end of sentence


class Vocabulary(StateUnit):
    def __init__(self, special_tokens=None, with_sos=True, with_eos=True):
        """
        :param special_tokens: list. e.g. ['<spt1>', '<spt2>', '<spt3>']
        """
        super().__init__()
        self.with_sos = with_sos
        self.with_eos = with_eos

        self._vocab = {
            PAD: 0,
            UNK: 1,
            SOS: 2,
            EOS: 3,
        }
        if special_tokens is not None:
            for token in special_tokens:
                self._vocab[token] = len(self._vocab)

        self._index2token = dict()
        self._update_index2token()

        self._context['vocab'] = self._vocab
        self._context['index2token'] = self._index2token
        self._context['with_sos'] = with_sos
        self._context['with_eos'] = with_eos

    def _update_index2token(self):
        index2token = {v: k for k, v in self._vocab.items()}
        self._index2token = index2token
        return index2token

    def fit(self, input_: list):
        """
        :param input_: list of tokens, or list of lists containing tokens, inner list stands for single sentence
        """
        assert len(input_) > 0, "Input must can not be empty"
        if not isinstance(input_[0], str):
            input_ = [token for l in input_ for token in l]
        token_set = set(input_)

        init_map_len, token_num = len(self._vocab), len(token_set)
        new_vocab = dict(list(zip(list(token_set), range(init_map_len, init_map_len + len(token_set)))))
        self._vocab.update(new_vocab)
        self._update_index2token()

        super().fit(input_)
        return self

    def transform(self, input_: list):
        indexes = [self._vocab.get(token) or self._vocab.get(token)[UNK] for token in input_]
        indexes = [self._vocab[SOS]] + indexes if self.with_sos else indexes
        indexes = indexes + [self._vocab[EOS]] if self.with_eos else indexes
        return indexes

    def reverse_transform(self, input_: list):
        return [self._index2token.get(index) or UNK for index in input_]

    @property
    def vocab(self):
        return self.vocab

    def to_txt(self, file_path):
        sequential_tokens = list(zip(*sorted(self._vocab.items(), key=lambda x: x[1], reverse=False)))[0]
        safe_save(file_path, '\n'.join(sequential_tokens), suffix='txt')

    def to_json(self, file_path):
        safe_save_json(file_path, self._vocab)

    @classmethod
    def read_txt(cls, file_path, encoding='utf-8', with_sos=True, with_eos=True):
        path = pathlib.Path(file_path)
        with codecs.open(path, 'r', encoding=encoding) as f:
            tokens = [token.strip() for token in f.readlines() if token.strip()]
            vocab = dict([(token, i) for i, token in enumerate(tokens)])

        instance = cls(with_sos=with_sos, with_eos=with_eos)
        instance._vocab = vocab
        instance._update_index2token()
        return instance

    @classmethod
    def read_json(cls, file_path, encoding='utf-8', with_sos=True, with_eos=True):
        path = pathlib.Path(file_path)
        check_suffix(path, suffix='json')
        with codecs.open(path, 'r', encoding=encoding) as f:
            vocab = json.load(f)

        instance = cls(with_sos=with_sos, with_eos=with_eos)
        instance._vocab = vocab
        instance._update_index2token()
        return instance
