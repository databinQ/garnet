# coding: utf-8

"""
@File   : mixin.py
@Author : garnet
@Time   : 2020/4/24 16:15
"""

import typing
import numpy as np
import pandas as pd
from tqdm import tqdm

from .constant import COLUMN_TEXT


class BaseMixin(object):
    pass


class PandasMixin(BaseMixin):
    pass


class TextMixin(BaseMixin):
    @staticmethod
    def apply_on_text(
            data: typing.Union[pd.DataFrame, list],
            func: typing.Callable,
            text_column: str = COLUMN_TEXT,
            name: typing.Optional[str] = None,
            verbose: int = 1,
            *args,
            **kwargs,
    ):
        """
        Apply preprocess function on text stored in `pandas.DataFrame`.

        data: `list` or `pandas.DataFrame` which contains text samples
        func: The function to apply
        text_column: Name of text column
        name: Name of data column obtained with function. Default is `None`, no new column is create, and the
        result of function replaces `text_column`
        verbose: Verbosity
        """
        new_column = name or text_column
        if verbose:
            if isinstance(data, pd.DataFrame):
                tqdm.pandas(desc="Processing " + name + " with " + func.__name__)
                data[new_column] = data[text_column].progress_apply(func, args=args, **kwargs)
            else:
                data = [func(sample) for sample in tqdm(data)]
        else:
            if isinstance(data, pd.DataFrame):
                data[new_column] = data[text_column].apply(func, args=args, **kwargs)
            else:
                data = [func(sample) for sample in data]
        return data


class ClassificationMixin(BaseMixin):
    def __iter__(self):
        """
        Make `DataPack` as a generator which producing batch data.
        """
        raise NotImplementedError

    def unpack(self):
        """
        Unpack the data for training.
        The return value can be directly feed to `model.fit` or `model.fit_generator`.

        :return: A tuple of (X, y). `y` is `None` if `self` has no label.
        """
        raise NotImplementedError

    def shuffle(self):
        raise NotImplementedError

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
    def _shuffle(data: typing.Union[list, tuple, np.ndarray, pd.DataFrame, pd.Series, None]):
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
