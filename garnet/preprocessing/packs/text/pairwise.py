# coding: utf-8

"""
@File   : pairwise.py
@Author : garnet
@Time   : 2020/3/25 23:25
"""

from . import TextDataPack
from .. import ClassifyDataPackMixin

from ..constant import *


class PairwiseTextDataPack(TextDataPack, ClassifyDataPackMixin):
    def __init__(self, left, relation, right=None):
        super(PairwiseTextDataPack, self).__init__()
        self._left = left
        self._relation = relation
        self._right = right or self._left
        self.shared = True if right is None or left is right else False

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, data):
        self._relation = data

    def apply(self, func, mode='both', name=None, *args, **kwargs):
        if self.shared or mode == 'both':
            self._left = self.apply_on_text(
                data=self._left, func=func, text_column=COLUMN_TEXT, name=name, *args, **kwargs
            )
            self._right = self.apply_on_text(
                data=self._right, func=func, text_column=COLUMN_TEXT, name=name, *args, **kwargs
            )
        elif mode == 'left':
            self._left = self.apply_on_text(
                data=self._left, func=func, text_column=COLUMN_TEXT, name=name, *args, **kwargs
            )
        elif mode == 'right':
            self._right = self.apply_on_text(
                data=self._right, func=func, text_column=COLUMN_TEXT, name=name, *args, **kwargs
            )
        else:
            raise ValueError("Parameter `mode` must be one of ('both', 'left', 'right')")

    def shuffle(self):
        self._relation = self._shuffle(self._relation)

    def unpack(self):
        left_X = self._relation.merge(self._left, on=COLUMN_TEXT_ID)[COLUMN_TEXT].tolist()
        right_X = self._relation.merge(self._right, on=COLUMN_TEXT_ID)[COLUMN_TEXT].tolist()
        return (left_X, right_X), self._relation[COLUMN_LABEL] if COLUMN_LABEL in self._relation.columns else None
