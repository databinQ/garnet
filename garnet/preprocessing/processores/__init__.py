# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/25 16:20
"""

import abc
import dill
import codecs
import typing
from pathlib import Path

from ..packs import DataPack


class DataPreprocessor(metaclass=abc.ABCMeta):
    """
    A preprocessor should be used in two steps. First, `fit`, then, `transform`.

    `fit` collects information into `context`, which includes everything the preprocessor needs to `transform` together
    with other useful information for later use. `fit` will only change the preprocessor's inner state but not the
    input data.

    In contrast, `transform` returns a modified copy of the input data without changing the preprocessor's inner state.
    """
    DATA_FILENAME = "preprocessor.dill"

    def __init__(self, *args, **kwargs):
        self._context = dict()

    @property
    def context(self):
        return self._context

    @abc.abstractmethod
    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be implemented in the child class, and is expected to return
        itself as a callable object.

        :param data_pack: :class:`DataPack` object to be fitted.
        :param verbose: Verbosity.
        """

    @abc.abstractmethod
    def transform(self, data_pack: DataPack, verbose: int = 1):
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be implemented in the child class.

        :param data_pack: :class:`DataPack` object to be transformed.
        :param verbose: Verbosity.
        """

    def fit_transform(self, data_pack: DataPack, verbose: int = 1):
        return self.fit(data_pack, verbose=verbose).transform(data_pack, verbose=verbose)

    def save(self, directory_path: typing.Union[str, Path]):
        """
        Save the `DataPreprocessor` object to specified directory with fixed file name.
        """
        directory_path = Path(directory_path)
        file_path = directory_path.joinpath(self.DATA_FILENAME)

        if file_path.exists():
            print("{} already exist, now covering old version data...".format(file_path))
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)

        dill.dump(self, codecs.open(file_path, "wb"))
