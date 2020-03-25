# coding: utf-8
"""
@File   : group
@Author : garnet
@Time   : 2020/3/25 17:13
"""

import typing
import pandas as pd

from . import TextDataPack


class GroupTextDataPack(TextDataPack):
    def __init__(
            self,
            chunk_data: pd.DataFrame = None,
            chunk_json: typing.Optional[dict, str] = None,
            group_data: pd.DataFrame = None,
            text_data: pd.DataFrame = None,
            group_index_column=None,
            group_name_column=None,
            group_info_columns=None,
            text_columns=None,
            *args,
            **kwargs
    ):
        assert chunk_data is not None or chunk_json is not None or (group_data is not None and text_data is not None), \
            "One of [`chunk_data`, `chunk_json`, (`group_data`, `text_data`)] must be assigned"

        super(GroupTextDataPack, self).__init__(*args, **kwargs)

        self._group_index_column = group_index_column
        self._group_name_column = group_name_column
        self._group_info_columns = group_info_columns or []
        self._text_columns = text_columns or self.DEFAULT_TEXT_COLUMNS

        self.group_data = None
        self.text_data = None

        if chunk_data is not None:
            self.initial_chunk_data(chunk_data)

    def initial_chunk_data(self, chunk_data):
        group_cols = [self._group_index_column]
        if self._group_name_column:
            group_cols += [self._group_name_column]
        group_cols += self._group_info_columns

        self.group_data = chunk_data[group_cols].drop_duplicates()
        self.text_data = chunk_data[[self._group_index_column, self._text_columns]].set_index(self._group_index_column)

    def has_label(self):
        return False

    @property
    def num_group(self):
        return 0 if self.group_data is None else len(self.group_data)
