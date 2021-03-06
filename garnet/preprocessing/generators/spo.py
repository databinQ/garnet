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
from ..units.task.spo import SpoSearcher
from ...utils.sequence import sequence_padding


class SpoBertDataGenerator(LazyDataGenerator):
    def __init__(self,
                 data_pack: SpoDataPack,
                 tokenizer: BertTokenizer,
                 batch_size: int,
                 searcher: SpoSearcher = None,
                 shuffle: bool = True,
                 buffer_size: typing.Optional[int] = None,
                 *args,
                 **kwargs):
        super().__init__(data=data_pack, batch_size=batch_size, shuffle=shuffle, buffer_size=buffer_size)

        assert tokenizer.fitted is True, "Need a fitted Tokenizer"
        self._tokenizer = tokenizer
        self._searcher = searcher
        self.pos_padding, self.pos_truncate = 'post', 'pre'

    def __iter__(self):
        batch_token_ids = []
        batch_segment_ids = []
        batch_subject_labels = []
        batch_subject_ids = []
        batch_object_labels = []
        batch_other_subjects = []
        batch_other_objects = []

        for i, sample in enumerate(self.sample()):
            text, spo = sample
            token_ids, segment_ids = self._tokenizer.transform(text)

            prior_spoes = dict()
            if self._searcher is not None:
                for prior_spo in self._searcher.extract(text, excludes=spo):
                    prior_s, prior_p, prior_o = prior_spo
                    pid = self.data_pack.schema2id[prior_p]
                    s_idx, s_len = self._search_start_index(prior_s, token_ids)
                    o_idx, o_len = self._search_start_index(prior_o, token_ids)
                    if s_idx != -1 and o_idx != -1:
                        s_key = (s_idx, s_idx + s_len - 1)
                        o_key = (o_idx, o_idx + o_len - 1, pid)
                        if s_key not in prior_spoes:
                            prior_spoes[s_key] = []
                        prior_spoes[s_key].append(o_key)

            if not self.data_pack.with_label:
                # test data
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                if len(batch_token_ids) == self.batch_size:
                    yield self.gen_test_batch(batch_token_ids, batch_segment_ids)
                    batch_token_ids = []
                    batch_segment_ids = []

                continue

            sample_spoes = dict()
            if self.data_pack.with_label:
                for s, p, o in spo:
                    pid = self.data_pack.schema2id[p]
                    s_idx, s_len = self._search_start_index(s, token_ids)
                    o_idx, o_len = self._search_start_index(o, token_ids)
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + s_len - 1)
                        o = (o_idx, o_idx + o_len - 1, pid)
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
                object_labels = np.zeros(shape=(len(token_ids), self.data_pack.num_predicate, 2))
                for o in sample_spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1

                if self._searcher is not None:
                    prior_subjects_ids = np.zeros(shape=(len(token_ids), 2))
                    prior_objects_ids = np.zeros(shape=(len(token_ids), self.data_pack.num_predicate, 2))
                    if prior_spoes:
                        s_start_ids, s_end_ids = list(zip(*prior_spoes.keys()))
                        prior_subjects_ids[s_start_ids, 0] = 1
                        prior_subjects_ids[s_end_ids, 1] = 1
                    for o in prior_spoes.get(subject_ids, []):
                        prior_objects_ids[o[0], o[2], 0] = 1
                        prior_objects_ids[o[1], o[2], 1] = 1
                    prior_objects_ids = prior_objects_ids.reshape((len(token_ids), -1))

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if self._searcher is not None:
                    batch_other_subjects.append(prior_subjects_ids)
                    batch_other_objects.append(prior_objects_ids)

                if len(batch_token_ids) == self.batch_size:
                    yield self.gen_train_batch(batch_token_ids, batch_segment_ids,
                                               batch_subject_labels, batch_subject_ids, batch_object_labels,
                                               batch_other_subjects, batch_other_objects)
                    batch_token_ids = []
                    batch_segment_ids = []
                    batch_subject_labels = []
                    batch_subject_ids = []
                    batch_object_labels = []
                    batch_other_subjects = []
                    batch_other_objects = []
        if len(batch_token_ids) != 0:
            if self.data_pack.with_label:
                yield self.gen_train_batch(batch_token_ids, batch_segment_ids,
                                           batch_subject_labels, batch_subject_ids, batch_object_labels,
                                           batch_other_subjects, batch_other_objects)
            else:
                yield self.gen_test_batch(batch_token_ids, batch_segment_ids)

    def gen_train_batch(self,
                        batch_token_ids,
                        batch_segment_ids,
                        batch_subject_labels,
                        batch_subject_ids,
                        batch_object_labels,
                        batch_other_subjects,
                        batch_other_objects):
        batch_token_ids = sequence_padding(
            batch_token_ids,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate
        )
        batch_segment_ids = sequence_padding(
            batch_segment_ids,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate
        )
        batch_subject_labels = sequence_padding(
            batch_subject_labels,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate,
            padding_index=np.zeros(2)
        )
        batch_subject_ids = np.array(batch_subject_ids)
        batch_object_labels = sequence_padding(
            batch_object_labels,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate,
            padding_index=np.zeros(shape=(self.data_pack.num_predicate, 2))
        )

        if len(batch_other_subjects) != 0 and len(batch_other_objects) != 0:
            batch_other_subjects = sequence_padding(
                batch_other_subjects,
                max_length=self._tokenizer.max_length,
                padding=self.pos_padding,
                truncate=self.pos_truncate,
                padding_index=np.zeros(2)
            )
            batch_other_objects = sequence_padding(
                batch_other_objects,
                max_length=self._tokenizer.max_length,
                padding=self.pos_padding,
                truncate=self.pos_truncate,
                padding_index=np.zeros(shape=(self.data_pack.num_predicate * 2))
            )
            return [batch_token_ids,
                    batch_segment_ids,
                    batch_subject_labels,
                    batch_subject_ids,
                    batch_object_labels,
                    batch_other_subjects,
                    batch_other_objects], \
                   None
        else:
            return [batch_token_ids,
                    batch_segment_ids,
                    batch_subject_labels,
                    batch_subject_ids,
                    batch_object_labels], \
                   None

    def gen_test_batch(self, batch_token_ids, batch_segment_ids):
        batch_token_ids = sequence_padding(
            batch_token_ids,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate
        )
        batch_segment_ids = sequence_padding(
            batch_segment_ids,
            max_length=self._tokenizer.max_length,
            padding=self.pos_padding,
            truncate=self.pos_truncate
        )
        return [batch_token_ids, batch_segment_ids]

    @staticmethod
    def _search(pattern_ids, text_ids):
        n = len(pattern_ids)
        for i in range(len(text_ids)):
            if text_ids[i: i + n] == pattern_ids:
                return i
        return -1

    def _search_start_index(self, token, text_token_ids):
        """
        :param token: token string
        :param text_token_ids: list containing each text token index
        """
        token_ids = self._tokenizer.transform(token)[0][1:-1]
        return self._search(token_ids, text_token_ids), len(token_ids)
