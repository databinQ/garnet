# coding: utf-8
# @Author: garnet
# @Time: 2019/12/5 07:31

import math
import typing
import itertools
import numpy as np
import pandas as pd

from ..pack import DataPack
from ..pack.pairwise import PairwiseDataPack


class GroupDataPack(DataPack):
    AVAILABLE_PAIR_MODE = []

    def __init__(self, *args, **kwargs):
        self._group_index = None
        self._group_text = None

    @property
    def major_data(self):
        return self._group_text

    @property
    def has_label(self) -> bool:
        return False

    @property
    def group_index(self):
        return self._group_index

    @property
    def group_text(self):
        return self._group_text

    @group_text.setter
    def group_text(self, data: pd.DataFrame):
        self._group_text = data

    def apply(self, func: typing.Callable, mode="normal", name="applied_text", verbose=1):
        self._group_text = self._apply_on_text(self._group_text, func=func, text_col="text", name=name, verbose=verbose)

    def make_pairs(self, *args, **kwargs):
        raise NotImplementedError

    def get_pair_mode(self):
        return self.AVAILABLE_PAIR_MODE

    @staticmethod
    def check_text_format(data):
        """
        Input DataFrame `data` must contains column `text` and `group_index`.
        """
        t_columns = data.columns
        try:
            assert "text" in t_columns and "group_index" in t_columns
        except AssertionError as e:
            raise ValueError(
                "Text DataFrame for `GroupDataPack` instance must contains column `text` and `group_index`"
            ) from e

    def _copy_and_reset_data(self, raw_data):
        data = raw_data.copy()
        data = data.reset_index().rename(columns={"index": "tid"})
        return data

    def _make_pairs_point_mode(self, data, num_pos=1, num_neg=1):
        self.check_text_format(data)

        one_side = self._copy_and_reset_data(data)

        total_pairs = []
        for group, group_data in one_side.groupby("group_index"):
            neg_text = one_side[one_side["group_index"] != group]
            for tid in group_data["tid"]:
                for rtid in group_data["tid"].sample(n=num_pos):
                    # Sampling positive samples
                    if tid != rtid:
                        total_pairs.append((tid, rtid, 1))
                for rtid in neg_text["tid"].sample(n=num_neg):
                    # Sampling negative samples
                    total_pairs.append((tid, rtid, 0))
        relation = pd.DataFrame(total_pairs, columns=["left_id", "right_id", "label"])
        one_side.drop("group_index", axis=1, inplace=True)
        return PairwiseDataPack(left=one_side, right=one_side.copy(), relation=relation)

    def _make_pairs_pair_mode(self, data, frac_pos=1., scale_neg=1.):
        self.check_text_format(data)

        one_side = self._copy_and_reset_data(data)

        total_pairs = []
        for group, group_data in one_side.groupby("group_index"):
            pos_pairs = list(itertools.combinations(group_data["tid"].tolist(), 2))
            np.random.shuffle(pos_pairs)
            pos_pairs = pos_pairs[:math.ceil(len(pos_pairs) * frac_pos)]
            pos_pairs = [tuple(list(tp) + [1]) for tp in pos_pairs]

            # If only one sentence was contained in this group, then continue to next
            if len(pos_pairs) == 0:
                continue

            neg_text = one_side[one_side["group_index"] != group]
            neg_pairs = []
            for i in range(math.ceil(len(pos_pairs) * scale_neg)):
                t_pos_id = group_data["tid"].sample(n=1).iloc[0]
                t_neg_id = neg_text["tid"].sample(n=1).iloc[0]
                neg_pairs.append((t_pos_id, t_neg_id, 0))

            group_pairs = pos_pairs + neg_pairs
            group_pairs = pd.DataFrame(group_pairs, columns=["left_id", "right_id", "label"])
            total_pairs.append(group_pairs)
        relation = pd.concat(total_pairs).reset_index(drop=True)
        one_side.drop("group_index", axis=1, inplace=True)
        return PairwiseDataPack(left=one_side, right=one_side.copy(), relation=relation)


class SimpleGroupDataPack(GroupDataPack):
    """
    Texts(sentence, paragraph or document) are divided into groups, texts contained in the single group are similar or
    have the similar semantic meaning.
    To initial a `GroupDataPack` instance, data should be organized in below format:

    ```
    {
        "group_name1": [g1_s1, g1_s2, g1_s3, ...],
        "group_name2": [g2_s1, g2_s2, g2_s3, ...],
        ...
    }
    ```

    in which `group_nameX` is the name of the group or just the index of the group, and `gX_sY` means the `Yth` sentence
    of group `X`.
    ```
    """
    AVAILABLE_PAIR_MODE = ["point", "pair"]
    MAJOR_NON_TEXT_COLUMNS = ["group_index"]

    def __init__(self, group_data):
        super(SimpleGroupDataPack, self).__init__()
        self._group_index = pd.DataFrame(
            [(i, tk) for i, tk in enumerate(group_data.keys())],
            columns=["index", "group_name"]
        )
        self._group_text = pd.DataFrame(
            [(i, ts) for i, (_, tl) in enumerate(group_data.items()) for ts in tl],
            columns=self.MAJOR_COLUMNS,
        )

    def make_pairs(self, mode="point", num_pos=1, num_neg=1, frac_pos=1., scale_neg=1.):
        return self._make_pairs_point_mode(self._group_text, num_pos=num_pos, num_neg=num_neg) if mode == "point" else \
            self._make_pairs_pair_mode(self._group_text, frac_pos=frac_pos, scale_neg=scale_neg)


class SubGroupDataPack(GroupDataPack):
    """
    Sentences in single group are separated into several sub-groups, and relation are set up between sub-groups within
    internal group, sub-groups from different groups do not have relationships in general.
    """
    AVAILABLE_PAIR_MODE = ["internal", "external_point", "external_pair"]
    MAJOR_NON_TEXT_COLUMNS = ["group_index", "sub_group_index"]

    def __init__(self, group_data):
        super(SubGroupDataPack, self).__init__()
        self._group_index = pd.DataFrame(
            [(i, tk) for i, tk in enumerate(group_data.keys())],
            columns=["index", "group_name"]
        )
        self._group_text = pd.DataFrame(
            [(i, j, ts) for i, (_, tg) in enumerate(group_data.items()) for j, tl in enumerate(tg) for ts in tl],
            columns=self.MAJOR_COLUMNS,
        )

    def make_pairs(self, mode="internal", **kwargs):
        if mode.startswith("internal"):
            return self._make_pairs_internal_mode(self._group_index, mode=mode, **kwargs)
        else:
            raise NotImplementedError

    def _make_pairs_internal_mode(self, data, mode, **kwargs):
        t_columns = data.columns
        try:
            assert "text" in t_columns and "group_index" in t_columns and "sub_group_index" in t_columns
        except AssertionError as e:
            raise ValueError(
                "Text DataFrame for `GroupDataPack` instance must contains column `text`, `group_index`, "
                "and `sub_group_index`"
            ) from e

        if mode == "internal":
            def _full_mode(ldata, rdata):
                lid = ldata["tid"].tolist()
                rid = rdata["tid"].tolist()

                pos = [t + (1,) for t in itertools.combinations(lid, 2)]
                neg = [t + (0,) for t in itertools.product(lid, rid)]
                return pd.DataFrame(pos + neg, columns=["left_id", "right_id", "label"])

            data = data.reset_index(name="tid")

            lefts, rights, relations = [], [], []
            for group, group_data in data.groupby("group_index"):
                subs = [sub_data for _, sub_data in group_data.groupby("sub_group_index")]
                if len(subs) == 1:
                    continue

                left_data = subs[0]
                right_data = pd.concat(subs[1:])
                lefts.append(left_data)
                rights.append(right_data)

                relations.append(_full_mode(left_data, right_data))

            left = pd.concat(lefts).reset_index(drop=True).drop(self.MAJOR_NON_TEXT_COLUMNS, axis=1)
            right = pd.concat(rights).reset_index(drop=True).drop(self.MAJOR_NON_TEXT_COLUMNS, axis=1)
            relation = pd.concat(relations).reset_index(drop=True)
            return PairwiseDataPack(left=left, right=right.copy(), relation=relation)
        elif mode == "external_point":
            self._make_pairs_point_mode(
                self._group_text,
                num_pos=kwargs.get("num_pos", 1),
                num_neg=kwargs.get("num_neg", 1)
            )
        elif mode == "external_pair":
            self._make_pairs_pair_mode(
                self._group_text,
                frac_pos=kwargs.get("frac_pos", 1),
                scale_neg=kwargs.get("scale_neg", 1)
            )
        else:
            raise NotImplementedError(
                "`{}` mode hasn't been implement, must be one of {}".format(mode, ", ".join(self.AVAILABLE_PAIR_MODE))
            )
