# coding: utf-8
# @Author: garnet
# @Time: 2019/12/5 07:31

import typing
import pandas as pd

from ..pack import DataPack
from ..pack.pairwise import PairwiseDataPack


class GroupDataPack(DataPack):
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
    def __init__(self, group_data):
        self._group_index = pd.DataFrame([(i, tk) for i, tk in enumerate(group_data.keys())], columns=["index", "group_name"])
        self._group_text = pd.DataFrame(
            [(i, ts) for i, (_, tl) in enumerate(group_data.items()) for ts in tl],
            columns=["group_index", "text"]
        )

    @property
    def has_label(self) -> bool:
        return False

    @property
    def group_index(self):
        return self._group_index

    @property
    def group_text(self):
        return self._group_text

    def apply(self, func: typing.Callable, mode="normal", name="applied_text", verbose=1):
        self._group_text = self._apply_on_text(self._group_text, func=func, text_col="text", name=name, verbose=verbose)

    def make_pairs(self):
        pass
