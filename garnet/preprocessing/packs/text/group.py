# coding: utf-8
"""
@File   : group
@Author : garnet
@Time   : 2020/3/25 17:13
"""

import math
import typing
import itertools
import numpy as np
import pandas as pd

from . import TextDataPack
from .pairwise import PairwiseTextDataPack
from ..constant import *


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
            text_id_column=None,
            text_column=None,
            *args,
            **kwargs
    ):
        assert chunk_data is not None or chunk_json is not None or (group_data is not None and text_data is not None), \
            "One of [`chunk_data`, `chunk_json`, (`group_data`, `text_data`)] must be assigned"

        super(GroupTextDataPack, self).__init__(*args, **kwargs)

        self._group_index_column = group_index_column
        self._group_name_column = group_name_column
        self._group_info_columns = group_info_columns or []
        self._text_id_column = text_id_column
        self._text_columns = text_column

        self.group_data = None
        self.text_data = None

        if chunk_data is not None:
            self.initial_chunk_data(chunk_data)
        elif chunk_json is not None:
            self.initial_json_data(chunk_json)
        else:
            self.group_data = group_data
            self.text_data = text_data
        self._rename()

    def _rename(self):
        self.group_data.rename(columns={
            self._group_index_column: COLUMN_GROUP_ID,
            self._group_name_column: COLUMN_GROUP_NAME,
        })
        self.text_data.rename(columns={
            self._group_index_column: COLUMN_GROUP_ID,
            self._text_id_column: COLUMN_TEXT_ID,
            self._text_columns: COLUMN_TEXT,
        })
        self._group_index_column = COLUMN_GROUP_ID
        self._group_name_column = COLUMN_GROUP_NAME
        self._text_id_column = COLUMN_TEXT_ID
        self._text_columns = COLUMN_TEXT

    def initial_chunk_data(self, chunk_data):
        group_cols = [self._group_index_column]
        if self._group_name_column:
            group_cols += [self._group_name_column]
        group_cols += self._group_info_columns

        self.group_data = chunk_data[group_cols].drop_duplicates()
        self.text_data = chunk_data[[self._group_index_column, self._text_columns]]
        self.text_data[COLUMN_TEXT_ID] = range(len(self.text_data))
        self.text_data = self.text_data[[COLUMN_TEXT_ID, self._group_index_column, self._text_columns]]

    def initial_json_data(self, json_data):
        # TODO: Finish this method
        pass

    @property
    def num_group(self):
        return 0 if self.group_data is None else len(self.group_data)

    @property
    def num_sentence(self):
        return 0 if self.text_data is None else len(self.text_data)

    def apply(self, func: typing.Callable, name=None, verbose=1, *args, **kwargs):
        self.text_data = self.apply_on_text(
            self.text_data, func=func, text_column=self._text_columns, name=name, verbose=verbose, *args, **kwargs
        )

    def make_pairs(self, mode='point', num_pos=1, num_neg=1, scale_pos=1., scale_neg=1.):
        """
        Generate pairwise data from grouped data.
        :param mode: Method of generating pairwise data `point` or 'group'
        :param num_pos: Number of positive samples made from one text, used for `point` method
        :param num_neg: Number of negative samples made from one text, used for `point` method
        :param scale_pos: Scale rate of positive text pairs, less or equal than 1.0, used for `group` method
        :param scale_neg: Scale rate of negative text pairs, greater or equal than 1.0, used for `group` method
        :return: Pairwise data in `PairwiseTextDataPack` instance
        """
        if mode == 'point':
            return self._make_pair_point_mode(num_pos=num_pos, num_neg=num_neg)
        elif mode == 'group':
            return self._make_pair_group_mode(scale_pos=scale_pos, scale_neg=scale_neg)
        else:
            raise ValueError("`mode` must be one of ('point', 'group'), got{} instead".format(mode))

    def _make_pair_point_mode(self, num_pos=1, num_neg=1):
        group_text_map = self.text_data[[COLUMN_TEXT_ID, self._group_index_column]]

        pairs = []
        for group_index, group_data in group_text_map.groupby(self._group_index_column):
            negatives = group_text_map[group_text_map[self._group_index_column] != group_index]
            for lid in group_data[COLUMN_TEXT_ID]:
                # Gather positives
                for rid in group_data[COLUMN_TEXT_ID].sample(n=num_pos):
                    if lid != rid:
                        pairs.append((lid, rid, 1))
                # Gather negatives
                for rid in negatives[COLUMN_TEXT_ID].sample(n=num_neg):
                    pairs.append((lid, rid, 0))

        np.random.shuffle(pairs)
        relation = pd.DataFrame(pairs, columns=[COLUMN_PAIRWISE_LEFT_ID, COLUMN_PAIRWISE_RIGHT_ID, COLUMN_LABEL])
        share_data = self.text_data[[COLUMN_TEXT_ID, COLUMN_TEXT]]
        return PairwiseTextDataPack(left=share_data, right=share_data, relation=relation)

    def _make_pair_group_mode(self, scale_pos=1., scale_neg=1.):
        group_text_map = self.text_data[[COLUMN_TEXT_ID, COLUMN_GROUP_ID]]

        pairs = []
        for group_index, group_data in group_text_map.groupby(COLUMN_GROUP_ID):
            pos_pairs = list(itertools.combinations(group_data[COLUMN_TEXT_ID].tolist(), 2))
            pos_pairs = pos_pairs[:math.ceil(len(pos_pairs) * scale_pos)]
            pos_pairs = [tuple(list(tp) + [1]) for tp in pos_pairs]

            # If this group only contains single one sentence, then positive pair can not be generated,
            # and continue to next group
            if len(pos_pairs) == 0:
                continue

            negatives = group_text_map[group_text_map[COLUMN_GROUP_ID] != group_index]
            neg_pairs = []
            for i in range(math.ceil(len(pos_pairs) * scale_neg)):
                t_pos_id = group_data[COLUMN_TEXT_ID].sample(n=1).iloc[0]
                t_neg_id = negatives[COLUMN_TEXT_ID].sample(n=1).iloc[0]
                neg_pairs.append((t_pos_id, t_neg_id, 0))

            group_pairs = pos_pairs + neg_pairs
            pairs.extend(group_pairs)

        np.random.shuffle(pairs)
        relation = pd.DataFrame(pairs, columns=[COLUMN_PAIRWISE_LEFT_ID, COLUMN_PAIRWISE_RIGHT_ID, COLUMN_LABEL])
        share_data = self.text_data[[COLUMN_TEXT_ID, COLUMN_TEXT]]
        return PairwiseTextDataPack(left=share_data, right=share_data, relation=relation)

    class GroupView(object):
        """
        Internal class used for `t2g` function.
        """

        def __init__(self, data_pack: 'GroupTextDataPack'):
            self.data_pack = data_pack
            self.gid_column = self.data_pack._group_index_column
            self.tid_column = COLUMN_TEXT_ID

        def __getitem__(self, index: typing.Optional[int, list, tuple, np.array]):
            if isinstance(index, int):
                res = self.data_pack.text_data.loc[self.data_pack.text_data[self.tid_column] == index, self.gid_column]
                return None if len(res) == 0 else res.iloc[0]
            else:
                if isinstance(index, np.array):
                    assert index.ndim == 1, "Numpy array containing text index must be 1 dimension"
                return [t if t == t else None for t in pd.DataFrame(index, columns=[self.tid_column]).merge(
                    self.data_pack.text_data, how="left", on=self.tid_column)[self.gid_column]]

    @property
    def t2g(self):
        """
        Get corresponding group index with text/sentence index

        Example:
            >>> gtdp = GroupTextDataPack(...)
            >>> gtdp.t2g[7]  # 7 is text/sentence id
            89  # 89 is the group index that text/sentence with id 7 belonging to
            >>> gtdp.t2g[[7, 0, 45, 356, 7, 1053]]
            [89, 1, 0, 333, 89, 123]  # text/sentence id can be organized in `list`, `tuple`, `numpy.array`, and return
            is a `list`
        """
        return self.GroupView(self)
