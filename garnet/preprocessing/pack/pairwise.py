# coding: utf-8
# @Author: garnet
# @Time: 2019/12/5 23:19

import pandas as pd

from ..pack import ModelDataPack


class PairwiseDataPack(ModelDataPack):
    MAJOR_NON_TEXT_COLUMNS = ["tid"]
    
    def __init__(self, left: pd.DataFrame, right: pd.DataFrame, relation: pd.DataFrame):
        self._left = left
        self._right = right
        self._relation = relation

    @property
    def major_data(self):
        return self._left

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
    def relation(self, data: pd.DataFrame):
        self._relation = data

    def apply(self, func, mode="normal", name="applied_text", verbose=1):
        if mode == "left":
            self._apply_on_text(self._left, func=func, text_col="text", name=name, verbose=verbose)
        if mode == "right":
            self._apply_on_text(self._right, func=func, text_col="text", name=name, verbose=verbose)
        if mode == "normal":
            self._apply_on_text(self._left, func=func, text_col="text", name=name, verbose=verbose)
            self._apply_on_text(self._right, func=func, text_col="text", name=name, verbose=verbose)

    def has_label(self) -> bool:
        return "label" in self._relation.columns

    def __len__(self):
        return self._relation.shape[0]

    def shuffle(self):
        self._relation.sample(frac=1.)
        self._relation.reset_index(drop=True, inplace=True)
