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
from collections import Iterable

from ..packs import DataPack
from ...utils.get_item import get_item


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_pack: DataPack,
                 batch_size: int,
                 shuffle: bool = True,
                 preload: bool = False,
                 *args,
                 **kwargs):
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.data_pack = data_pack
        self.num_per_epoch = len(self.data_pack)
        self.steps = self.get_steps(self.num_per_epoch)

        self._batch_indices = None
        self.reset_index()

        # Load and store data before training, only used when dataset is small
        self.loaded, self._data = False, None
        if preload:
            self.preload()
            self.loaded = True

    def __len__(self):
        return self.steps

    def __getitem__(self, item: int):
        indices = self._batch_indices[item]
        if self.loaded:
            if isinstance(self._data, tuple):
                return tuple(get_item(t, indices) for t in self._data)
            else:
                return get_item(self._data, indices)
        else:
            return self.transform(self.data_pack[indices])

    def whole(self):
        if not self.loaded:
            self.preload()
        return self._data

    def preload(self):
        assert hasattr(self.data_pack, 'unpack'), "Custom `DataPack` class must have `unpack` method"
        tmp_data = self.data_pack.unpack()
        self._data = self.transform(tmp_data)

    def transform(self, data):
        return data

    def get_steps(self, num_samples):
        return num_samples // self.batch_size + 1 if num_samples % self.batch_size != 0 else 0

    def on_epoch_end(self):
        if self.shuffle:
            self.reset_index()

    def reset_index(self):
        indices = list(range(len(self)))
        if self.shuffle:
            np.random.shuffle(indices)

        self._batch_indices = []
        for i in range(self.steps):
            lower, upper = self.batch_size * i, self.batch_size * (i + 1)
            batch = indices[lower: upper]
            self._batch_indices.append(batch)


class LazyDataGenerator(object):
    def __init__(self,
                 data: typing.Union[DataPack, Iterable],
                 batch_size,
                 buffer_size=None,
                 shuffle: bool = True,
                 *args,
                 **kwargs):
        self.data_pack = data
        self.batch_size = batch_size
        self.num_per_epoch = len(self.data_pack) if hasattr(data, '__len__') and hasattr(data, '__getitem__') else None
        self.steps = self.get_steps(self.num_per_epoch) if self.num_per_epoch else None
        self.buffer_size = buffer_size or self.batch_size * 1000
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def __iter__(self):
        batch_data = []
        for sample in self.sample():
            batch_data.append(self.transform(sample))
            if len(batch_data) == self.batch_size:
                yield batch_data
                batch_data = []

        if batch_data:
            yield batch_data

    def transform(self, data):
        return data

    def get_steps(self, num_samples):
        return num_samples // self.batch_size + 1 if num_samples % self.batch_size != 0 else 0

    def generator_fixed_length(self):
        indices = list(range(self.num_per_epoch))
        np.random.shuffle(indices)
        for i in indices:
            yield self.data_pack[i]

    def generator_unfixed_length(self):
        caches = []
        for sample in self.data_pack:
            caches.append(sample)
            if len(caches) == self.buffer_size:
                index = np.random.randint(len(caches))
                yield caches.pop(index)

        while caches:
            index = np.random.randint(len(caches))
            yield caches.pop(index)

    def sample(self):
        if self.shuffle:
            generator = self.generator_fixed_length() if self.steps else self.generator_unfixed_length()
        else:
            generator = iter(self.data_pack)

        for s in generator:
            yield s
