# coding: utf-8
"""
@File   : __init__.py.py
@Author : garnet
@Time   : 2020/3/3 17:47
"""

import dill
import typing
import codecs
from pathlib import Path


class DataPack(object):
    """
    Train and test data are stored in `DataPack` object respectively, and in the of `pandas.DataFrame` in general.
    `DataPack` also provides common processing methods, and specific method for specific data structure.
    """
    DATA_FILENAME = "data.dill"
    INDEX_COLS = ["index"]

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

    @property
    def has_label(self):
        raise NotImplementedError
