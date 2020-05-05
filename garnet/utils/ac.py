# coding: utf-8

"""
@File   : ac.py
@Author : garnet
@Time   : 2020/5/5 15:32
"""

import ahocorasick


class ACUnicode(object):
    def __init__(self):
        self._ac = ahocorasick.Automaton()

    def _encode(self, s: str):
        return s

    def add_word(self, k: str, v=None):
        k = self._encode(k)
        return self._ac.add_word(k, v)

    def iter(self, s):
        s = self._encode(s)
        return self._ac.iter(s)

    def get(self, k, default=None):
        k = self._encode(k)
        return self._ac.get(k, default)

    def exists(self, k):
        k = self._encode(k)
        return self._ac.exists(k)

    def make_automaton(self):
        return self._ac.make_automaton()
