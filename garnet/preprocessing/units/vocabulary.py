# coding: utf-8

"""
@File   : vocabulary.py
@Author : garnet
@Time   : 2020/4/14 20:48
"""

from . import StateUnit


class Vocabulary(StateUnit):
    def __init__(self, special_tokens=None):
        """
        :param special_tokens: list. e.g. ['<spt1>', '<spt2>', '<spt3>']
        """
        super().__init__()
        self._vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[SOS]': 2,
            '[EOS]': 3,
        }
        if special_tokens is not None:
            for token in special_tokens:
                self._vocab[token] = len(self._vocab)
        self._index2token = {v: k for k, v in self._vocab.items()}
        self._context['vocab'] = self._vocab
        self._context['index2token'] = self._index2token

    def fit(self, input_):
        """
        :param input_: list of tokens
        """
