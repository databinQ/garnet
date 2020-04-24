# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/3 17:49
"""

import keras
import typing

from ..packs import DataPack


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_pack: DataPack, batch_size: int, shuffle: bool = True):
        ...

    def __len__(self):
        ...

    def __getitem__(self, item):
        ...

    def on_epoch_end(self):
        ...
