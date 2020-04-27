# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/4/25 21:17
"""

import typing
import numpy as np

from . import LazyDataGenerator
from ..packs.text.spo import SpoDataPack
from ..units.tokenizer import BertTokenizer
from ...utils.sequence import sequence_padding


class SpoBertDataGenerator(LazyDataGenerator):
    def __init__(self,
                 data_pack: SpoDataPack,
                 tokenizer: BertTokenizer,
                 batch_size: int,
                 shuffle: bool = True,
                 buffer_size: typing.Optional[int] = None,
                 *args,
                 **kwargs):
        super().__init__(data=data_pack, batch_size=batch_size, shuffle=shuffle, buffer_size=buffer_size)

        assert tokenizer.fitted is True, "Need a fitted Tokenizer"
        self._tokenizer = tokenizer

    def __iter__(self):
        batch_token_ids = []
        batch_segment_ids = []
        batch_subject_labels = []
        batch_subject_ids = []
        batch_object_labels = []

        for sample in self.sample():
            text, spo = sample


    def transform(self, data):
        self._tokenizer.transform()

    def initialize(self):
        assert hasattr(self.data_pack, 'unpack'), "Custom `DataPack` class must have `unpack` method"
        texts, spoes = self.data_pack.unpack()
        max_length = self._tokenizer.max_length

        total_token_ids, total_segment_ids, total_subject_labels, total_subject_ids, total_object_labels = \
            self.unpack(texts, spoes)
        max_length = max_length or max(len(x) for x in total_token_ids)

        pos_padding, pos_truncate = 'post', 'pre'
        self._total_token_ids = sequence_padding(
            total_token_ids, max_length=max_length, padding=pos_padding, truncate=pos_truncate
        )
        self._total_segment_ids = sequence_padding(
            total_segment_ids, max_length=max_length, padding=pos_padding, truncate=pos_truncate
        )
        self._total_subject_labels = sequence_padding(
            total_subject_labels, max_length=max_length, padding=pos_padding, truncate=pos_truncate,
            padding_index=np.zeros(2)
        )
        self._total_subject_ids = np.array(total_subject_ids)
        self._total_object_labels = sequence_padding(
            total_object_labels, max_length=max_length, padding=pos_padding, truncate=pos_truncate,
            padding_index=np.zeros(shape=(len(self.data_pack.schema2id), 2))
        )

        self.steps = self.cal_steps(len(self._total_token_ids))
        self.reset_index()

    def unpack(self, texts, spoes=None):
        spo_list = [None] * len(texts) if spoes is None else spoes

        total_token_ids = []
        total_segment_ids = []
        total_subject_labels = []
        total_subject_ids = []
        total_object_labels = []

        for text, spo in zip(texts, spo_list):
            token_ids, segment_ids = self._tokenizer.transform(text)
            if spoes is None:
                total_token_ids.append(token_ids)
                total_segment_ids.append(segment_ids)
                continue

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
                If only positive samples exist, given a subject, there must be an object. So negative samples are fake
                subject, thus they have no objects. 
                """
                start, end = np.array(list(sample_spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)

                # get object corresponds with subject
                # attention, subject can be fake, and object is empty in this case
                object_labels = np.zeros(shape=(len(token_ids), len(self.data_pack.schema2id), 2))
                for o in sample_spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1

                total_token_ids.append(token_ids)
                total_segment_ids.append(segment_ids)
                total_subject_labels.append(subject_labels)
                total_subject_ids.append(subject_ids)
                total_object_labels.append(object_labels)

        return total_token_ids, total_segment_ids, total_subject_labels, total_subject_ids, total_object_labels

    @staticmethod
    def _search(pattern_ids, text_ids):
        n = len(pattern_ids)
        for i in range(len(text_ids)):
            if text_ids[i: i + n] == pattern_ids:
                return i
        return -1

    def make_chunk(self):
        return self._total_token_ids, \
               self._total_segment_ids, \
               self._total_subject_labels, \
               self._total_subject_ids, \
               self._total_object_labels
