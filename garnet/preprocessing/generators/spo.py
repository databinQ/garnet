# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/4/25 21:17
"""

import numpy as np

from . import DataGenerator
from ..packs.text.spo import SpoDataPack
from ..units.tokenizer import BertTokenizer


class SpoBertDataGenerator(DataGenerator):
    def __init__(self,
                 data_pack: SpoDataPack,
                 tokenizer: BertTokenizer,
                 batch_size: int,
                 shuffle: bool = True,
                 *args,
                 **kwargs):
        super().__init__(data_pack=data_pack, batch_size=batch_size, shuffle=shuffle)

        self._tokenizer = tokenizer
        assert self._tokenizer.fitted is True, "Need a fitted Tokenizer"

    def initialize(self):
        assert hasattr(self.data_pack, 'unpack'), "Custom `DataPack` class must have `unpack` method"
        texts, spoes = self.data_pack.unpack()
        self._X, self._y = self.unpack(texts, spoes)
        self.cal_steps(len(self._X))
        self.reset_index()

    def unpack(self, texts, spoes=None):
        spoes = [None] * len(texts) if spoes is None else spoes
        for text, spo in zip(texts, spoes):
            token_ids, segment_ids = self._tokenizer.transform(text)

            sample_spoes = dict()
            if spo is not None:
                for s, p, o in spo:
                    s_token_ids = self._tokenizer.transform(s)[0][1:-1]
                    o_token_ids = self._tokenizer.transform(o)[0][1:-1]
                    pid = self.data_pack.schema2id[p]

                    s_idx = self._search(s_token_ids, token_ids)
                    o_idx = self._search(o_token_ids, token_ids)
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + len(s_token_ids) - 1)
                        o = (o_idx, o_idx + len(o_token_ids) - 1, pid)
                        if s not in sample_spoes:
                            sample_spoes[s] = []
                        sample_spoes[s].append(o)

            if sample_spoes:
                subject_labels = np.zeros(shape=(len(token_ids), 2))  # (seq_len, 2)
                s_start_ids, s_end_ids = list(zip(*sample_spoes.keys()))
                subject_labels[s_start_ids, 0] = 1
                subject_labels[s_end_ids, 1] = 1

                """
                Add negative samples.
                If only positive samples exist, given a subject, there must be an object. So negative samples are wrong
                subject, thus they have no objects. 
                """
                start, end = np.array(sample_spoes.keys()).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)







    @staticmethod
    def _search(pattern_ids, text_ids):
        n = len(pattern_ids)
        for i in range(len(text_ids)):
            if text_ids[i: i + n] == pattern_ids:
                return i
        return -1
