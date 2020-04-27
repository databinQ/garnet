# coding: utf-8

"""
@File   : get_item.py
@Author : garnet
@Time   : 2020/4/27 15:00
"""

import typing
import numpy as np
import pandas as pd


def slice_indices(indices: slice, total_length):
    for index in range(*indices.indices(total_length)):
        yield index


def get_item(data,
             indices: typing.Union[int, slice, list, np.array, pd.Series],
             keep_dim: bool = False):
    if data is None:
        return None

    if isinstance(data, list):
        return get_item_list(data, indices, keep_dim)
    elif isinstance(data, np.ndarray):
        return get_item_array(data, indices, keep_dim)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return get_item_pandas(data, indices, keep_dim)
    else:
        raise TypeError("Data class must be one of (list, np.ndarray, pd.DataFrame, pd.Series, None), "
                        "got {} instead".format(type(data)))


def get_item_list(data: list,
                  indices: typing.Union[int, slice, list, np.array, pd.Series],
                  keep_dim: bool = False):
    if isinstance(indices, int):
        return data[indices: indices + 1] if keep_dim else data[indices]
    elif isinstance(indices, slice):
        return [data[index] for index in slice_indices(indices, len(data))]
    else:
        return [data[index] for index in indices]


def get_item_array(data: np.ndarray,
                   indices: typing.Union[int, slice, list, np.array, pd.Series],
                   keep_dim: bool = False):
    if isinstance(indices, int):
        return data[indices: indices + 1] if keep_dim else data[indices]
    elif isinstance(indices, slice):
        return data[list(slice_indices(indices, len(data)))]
    else:
        return data[list(indices)]


def get_item_pandas(data: typing.Union[pd.DataFrame, pd.Series],
                    indices: typing.Union[int, slice, list, np.array, pd.Series],
                    keep_dim: bool = False):
    if isinstance(indices, int):
        return data.iloc[indices: indices + 1] if keep_dim else data.iloc[indices]
    elif isinstance(indices, slice):
        return data.iloc[list(slice_indices(indices, len(data)))]
    else:
        return data.iloc[list(indices)]
