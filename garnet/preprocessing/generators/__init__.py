# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/3 17:49
"""

import keras
import typing
import numpy as np
import pandas as pd

from ..packs import DataPack


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_pack: DataPack, batch_size: int, shuffle: bool = True):
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.data_pack = data_pack
        self._X, self._y = self.data_pack.unpack()

        self.step = len(data_pack) // self.batch_size + 1 if len(data_pack) % self.batch_size != 0 else 0
        self._batch_indices = None
        self.reset_index()

    def __len__(self):
        return self.step

    def __getitem__(self, item):
        indices = self._batch_indices[item]
        return self._get_item(self._X, indices), self._get_item(self._y, indices)

    def on_epoch_end(self):
        if self.shuffle:
            self.reset_index()

    def make_chunk(self):
        """
        Return total data, used by `fit` method of keras models, which need full data for an epoch.
        """
        return self._X, self._y

    def reset_index(self):
        indices = list(range(len(self)))
        if self.shuffle:
            np.random.shuffle(indices)

        self._batch_indices = []
        for i in range(self.step):
            lower, upper = self.batch_size * i, self.batch_size * (i + 1)
            batch = indices[lower: upper]
            self._batch_indices.append(batch)

    def _get_item(self, data, indices):
        if data is None:
            return None

        if isinstance(data, list):
            return [data[index] for index in indices]
        elif isinstance(data, pd.DataFrame):
            return data.iloc[indices]
        else:
            return data[indices]
