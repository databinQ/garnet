# coding: utf-8
"""
@File   : group
@Author : garnet
@Time   : 2020/3/25 17:13
"""

import typing
import numpy as np
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
        self._text_columns = text_columns or self.DEFAULT_TEXT_COLUMN

        self.group_data = None
        self.text_data = None

        if chunk_data is not None:
            self.initial_chunk_data(chunk_data)
        elif chunk_json is not None:
            self.initial_json_data(chunk_json)
        else:
            self.group_data = group_data
            self.text_data = text_data

    def initial_chunk_data(self, chunk_data):
        group_cols = [self._group_index_column]
        if self._group_name_column:
            group_cols += [self._group_name_column]
        group_cols += self._group_info_columns

        self.group_data = chunk_data[group_cols].drop_duplicates()
        self.text_data = chunk_data[[self._group_index_column, self._text_columns]]
        self.text_data[self.DEFAULT_TEXT_ID_COLUMN] = range(len(self.text_data))
        self.text_data = self.text_data[[self.DEFAULT_TEXT_ID_COLUMN, self._group_index_column, self._text_columns]]

    def initial_json_data(self, json_data):
        # TODO: Finish this method
        pass

    def has_label(self):
        return False

    @property
    def num_group(self):
        return 0 if self.group_data is None else len(self.group_data)

    @property
    def num_sentence(self):
        return 0 if self.text_data is None else len(self.text_data)

    def apply(self, func: typing.Callable, name=None, verbose=1):
        self.text_data = self.apply_on_text(
            self.text_data, func=func, text_column=self._text_columns, name=name, verbose=verbose
        )

    def make_pairs(self, mode='point', num_pos=1, num_neg=1):
        pass

    def _make_pair_point_mode(self, num_pos=1, num_neg=1):
        group_text_map = self.text_data[[self.DEFAULT_TEXT_ID_COLUMN, self._group_index_column]]

        pairs = []
        for group_index, group_data in group_text_map.groupby(self._group_index_column):
            negatives = group_text_map[group_text_map[self._group_index_column] != group_index]
            for lid in group_data[self.DEFAULT_TEXT_ID_COLUMN]:
                # Gather positives
                for rid in group_data[self.DEFAULT_TEXT_ID_COLUMN].sample(n=num_pos):
                    if lid != rid:
                        pairs.append((lid, rid, 1))
                # Gather negatives
                for rid in negatives[self.DEFAULT_TEXT_ID_COLUMN].sample(n=num_neg):
                    pairs.append((lid, rid, 0))
        relation = pd.DataFrame(pairs, columns=["left_id", "right_id", "label"])

    class GroupView(object):
        """
        Internal class used for `text2group` function.
        """
        def __init__(self, data_pack: 'GroupTextDataPack', gid_column: str, tid_column: str):
            self.data_pack = data_pack
            self.gid_column = gid_column
            self.tid_column = tid_column

        def __getitem__(self, index: typing.Optional[int, list, tuple, np.array]):
            if isinstance(index, int):
                res = self.data_pack.text_data.loc[self.data_pack.text_data[self.tid_column] == index, self.gid_column]
                return None if len(res) == 0 else res.iloc[0]
            else:
                if isinstance(index, np.array):
                    assert index.ndim == 1, "Numpy array containing text index must be 1 dimension"
