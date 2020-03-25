# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/3 18:18
"""

import typing
import pandas as pd
from tqdm import tqdm

from .. import DataPack


class TextDataPack(DataPack):
    DEFAULT_TEXT_COLUMNS = "text"

    @staticmethod
    def _apply_on_text(
            data: pd.DataFrame,
            func: typing.Callable,
            text_column: str = "text",
            name: str = None,
            verbose: int = 1
    ):
        """
        Apply preprocess function on text stored in `pandas.DataFrame`.

        data: Text data in `pandas.DataFrame` format
        func: The function to apply
        text_column: Name of text column
        name: Name of data column obtained with function. Default is `None`, no new column is create, and the
        result of function replaces `text_column`
        verbose: Verbosity
        """
        new_column = name or text_column
        if verbose:
            tqdm.pandas(desc="Processing " + name + " with " + func.__name__)
            data[new_column] = data[text_column].progress_apply(func)
        else:
            data[new_column] = data[text_column].apply(func)
        return data
