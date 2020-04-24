# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/3 17:47
"""

import dill
import typing
import codecs
import numpy as np
import pandas as pd
from pathlib import Path


class DataPack(object):
    """
    Train or test data are respectively stored in `DataPack` object. `DataPack` also provides common processing methods,
    and specific method for specific data structure.
    """
    DATA_FILENAME = "data.dill"

    def save(self, directory_path: typing.Union[str, Path]):
        """
        Save the `DataPack` object to specified directory with fixed file name.
        """
        directory_path = Path(directory_path)
        file_path = directory_path.joinpath(self.DATA_FILENAME)

        if file_path.exists():
            print("{} already exist, now covering old version data...".format(file_path))
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)

        dill.dump(self, codecs.open(file_path, "wb"))

    @classmethod
    def load(cls, directory_path: typing.Union[str, Path]):
        """
        Reload data from specified directory, and return a `DataPack` object
        """
        directory_path = Path(directory_path)
        file_path = directory_path.joinpath(cls.DATA_FILENAME)
        dp = dill.load(codecs.open(file_path, "rb"))
        return dp

    def apply(self, func: typing.Callable, verbose: int = 1, *args, **kwargs):
        raise NotImplementedError


class ClassifyDataPackMixin(object):
    """
    Mixin class for data used in classification task
    """

    def unpack(self):
        """
        Unpack the data for training.
        The return value can be directly feed to `model.fit` or `model.fit_generator`.

        :return: A tuple of (X, y). `y` is `None` if `self` has no label.
        """

    @property
    def X(self):
        return self.unpack()[0]

    @property
    def y(self):
        return self.unpack()[1]

    @property
    def has_label(self) -> bool:
        return False if self.y is None else True

    @staticmethod
    def _shuffle(data: typing.Optional[list, tuple, np.ndarray, pd.DataFrame, pd.Series]):
        if data is None:
            return data

        num_samples = len(data[0]) if isinstance(data, tuple) else len(data)
        random_index = np.random.permutation(num_samples)

        packed_data = data if isinstance(data, tuple) else (data,)
        new_data = []
        for d in packed_data:
            if isinstance(d, list):
                new_d = [d[i] for i in random_index]
            elif isinstance(d, (pd.DataFrame, pd.Series)):
                new_d = d.iloc[random_index]
            else:
                new_d = d[random_index]
            new_data.append(new_d)
        return tuple(new_data) if isinstance(data, tuple) else new_data[0]

    def shuffle(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


def load_data_pack(directory_path: typing.Union[str, Path]):
    return DataPack.load(directory_path)
