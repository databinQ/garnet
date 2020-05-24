# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/5/10 8:52
"""

import numpy as np
from tqdm import tqdm
from keras.models import Model

from . import Evaluator
from ...preprocessing.packs.text.spo import SpoDataPack
from ...preprocessing.units.tokenizer import BertTokenizer
from ...preprocessing.units.task.spo import SpoSearcher


class SPOTriplet(object):
    """
    Store simple spo triplet. Overwrite `__hash__` and `__eq__` method, to satisfy robust spo triple comparison
    """

    def __init__(self, subject, predicate, object_):
        """
        `subject`, `predicate` and `object_` are all strings
        """
        self.spo = (subject, predicate, object_)
        # self.spox = (
        #     spo[0],
        #     spo[1],
        #     tuple(sorted([(k, v) for k, v in spo[2].items()])),
        # )

    def __hash__(self):
        return self.spo.__hash__()

    def __eq__(self, spo):
        return self.spo == spo.spo

    def to_triplet(self):
        return self.spo

    def to_dict(self):
        return {
            'subject': self.spo[0],
            'predicate': self.spo[1],
            'object': self.spo[2]
        }

    def __str__(self):
        return '{}'.format(self.to_dict())

    def __repr__(self):
        return self.__str__()


class SPOComplexTriplet(SPOTriplet):
    """
    Spo triplet with complex object should be stored in this class
    """

    def __init__(self, subject, predicate, complex_object: dict):
        self.spo = (
            subject,
            predicate,
            tuple(sorted([(k, v) for k, v in complex_object.items()])),
        )

    @staticmethod
    def obj2tuple(dict_obj):
        return tuple(sorted([(k, v) for k, v in dict_obj.items()]))

    @staticmethod
    def obj2dict(tuple_obj):
        return {k: v for k, v in tuple_obj}

    def to_triplet(self):
        return self.spo[0], self.spo[1], self.obj2dict(self.spo[2])

    def to_dict(self):
        return {
            'subject': self.spo[0],
            'predicate': self.spo[1],
            'object': self.obj2dict(self.spo[2])
        }


class SpoPointEvaluator(Evaluator):
    def __init__(self,
                 subject_model: Model,
                 object_model: Model,
                 dev_data: SpoDataPack,
                 tokenizer: BertTokenizer,
                 threshold_sub_start=0.5,
                 threshold_sub_end=0.5,
                 threshold_obj_start=0.5,
                 threshold_obj_end=0.5,
                 save_path=None,
                 polarity='upper'):
        super().__init__(save_path=save_path, polarity=polarity)
        self._subject_model = subject_model
        self._object_model = object_model
        self._dev_data = dev_data
        self._tokenizer = tokenizer

        self._threshold_sub_start = threshold_sub_start
        self._threshold_sub_end = threshold_sub_end
        self._threshold_obj_start = threshold_obj_start
        self._threshold_obj_end = threshold_obj_end

    def on_epoch_end(self, epoch, logs=None):
        tp, pcp, cp = 1e-10, 1e-10, 1e-10

        pbar = tqdm()
        for sample in self._dev_data:
            text, spo_list = sample

            predict_spoes = self.extract(text, mode='object')
            real_spoes = self._schema_restore(spo_list)
            p_set = set(predict_spoes)
            r_set = set(real_spoes)

            tp += len(p_set & r_set)
            pcp += len(p_set)
            cp += len(r_set)
            f1, precision, recall = 2 * tp / (pcp + cp), tp / pcp, tp / cp

            pbar.update()
            pbar.set_description("f1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(f1, precision, recall))

        pbar.close()

        if self.better(f1):
            self.update_metric(f1)
            self.save_model()

        print('f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, best f1: {:.4f}\n'.format(
            f1,
            precision,
            recall,
            self.best_metric_value
        ))

    def _schema_restore(self, spo_list):
        spo_map = dict()
        for s, p, o in spo_list:
            p1, p2 = p.split('|')
            sp1 = (s, p1)
            if sp1 not in spo_map:
                spo_map[sp1] = dict()
            spo_map[sp1][p2] = o
        return list(set([self.SPO((k[0], k[1], v)) for k, v in spo_map.items()]))

    def _extract_fragment(self, text, start_token_id, end_token_id, mapping=None):
        if mapping is None:
            _, mapping = self._tokenizer.match_tokenize(text)

        if start_token_id < 0 or end_token_id >= len(mapping):
            return None

        start_char_ids = mapping[start_token_id]
        end_token_id = mapping[end_token_id]
        if len(start_char_ids) == 0 or len(end_token_id) == 0:
            print("Wrong token index for text: {}".format(text))
            return None

        start_char_index, end_char_index = start_char_ids[0], end_token_id[-1]
        return text[start_char_index: end_char_index + 1]

    def _subject_proba_parse(self, proba):
        """
        :param proba: 2D proba matrix of single sentence text with shape (sequence_length, 2)
        """
        start = np.where(proba[:, 0] > self._threshold_sub_start)[0]
        end = np.where(proba[:, 1] > self._threshold_sub_end)[0]

        sample_subject_ids = []
        for start_index in start:
            end_indices = end[end >= start_index]
            if len(end_indices):
                end_index = end_indices[0]
                sample_subject_ids.append((start_index, end_index))
        return sample_subject_ids

    def _object_proba_parse(self, proba):
        """
        :param proba:
        :return:
        """

    def predict_subjects(self, text):
        b_single = isinstance(text, str)
        if b_single:
            text = [text]

        total_subjects = []
        subject_preds = self.predict_subjects_proba(text)
        for i in range(len(text)):
            sample_subject_ids = self._subject_proba_parse(subject_preds[i, :, :])

            sample_subjects = []
            if sample_subject_ids:
                _, mapping = self._tokenizer.match_tokenize(text[i])
                for start_index, end_index in sample_subject_ids:
                    subject = self._extract_fragment(text[i], start_index, end_index, mapping=mapping)
                    if subject:
                        sample_subjects.append(subject)

            total_subjects.append(sample_subjects)
        return total_subjects

    def predict_subjects_proba(self, text):
        """
        :param text: single text string or list of texts
        """
        if isinstance(text, str):
            text = [text]

        id_list, segment_list = [], []
        for t in text:
            token_ids, segment_ids = self._tokenizer.transform(t)
            id_list.append(token_ids)
            segment_list.append(segment_ids)

        subject_preds = self._subject_model.predict([id_list, segment_list])
        return subject_preds

    def extract_spoes(self, text):
        # tokens, mapping = self._tokenizer.match_tokenize(text)
        token_ids, segment_ids = self._tokenizer.transform(text)
        subject_preds = self._subject_model.predict([[token_ids], [segment_ids]])
        subjects = self._subject_proba_parse(subject_preds[0, :, :])

        if subjects:
            spoes = []
            num_subjects = len(subjects)

            token_ids = np.repeat([token_ids], repeats=num_subjects, axis=0)
            segment_ids = np.repeat([segment_ids], repeats=num_subjects, axis=0)
            subjects = np.array(subjects)
            object_preds = self._object_model.predict([token_ids, segment_ids, subjects])

            for subject, object_pred in zip(subjects, object_preds):
                sub_start, sub_end = subject
                start = np.where(object_pred[:, :, 0] > self._threshold_obj_start)
                end = np.where(object_pred[:, :, 1] > self._threshold_obj_end)

                for start_idx, p_idx1 in zip(*start):
                    for end_idx, p_idx2 in zip(*end):
                        if p_idx1 == p_idx2 and 0 < start_idx <= end_idx:
                            spoes.append((
                                (mapping[sub_start][0], mapping[sub_end][-1]),
                                p_idx1,
                                (mapping[start_idx][0], mapping[end_idx][-1])
                            ))

            # list of (subject_text, predicate_name, object_text)
            res = [(text[s[0]: s[1] + 1], self._dev_data.id2schema[p], text[o[0]: o[1] + 1]) for s, p, o in spoes]
            return res
        return []

    def extract(self, text, mode='triple'):
        spo_list = self.extract_spoes(text)
        hie_spoes = self._schema_restore(spo_list)
        return [spo.spo for spo in hie_spoes] if mode == 'triple' else hie_spoes

    class SPO(tuple):
        """
        用来存三元组的类
        表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
        使得在判断两个三元组是否等价时容错性更好。
        """

        def __init__(self, spo):
            self.spox = (
                spo[0],
                spo[1],
                tuple(sorted([(k, v) for k, v in spo[2].items()])),
            )
            self._spo_raw = spo

        def __hash__(self):
            return self.spox.__hash__()

        def __eq__(self, spo):
            return self.spox == spo.spox

        @property
        def spo(self):
            return self._spo_raw

        def to_dict(self):
            return {
                'subject': self._spo_raw[0],
                'predicate': self._spo_raw[1],
                'object': self._spo_raw[2]
            }

        def __str__(self):
            return '{}'.format(self.to_dict())

        def __repr__(self):
            return self.__str__()


class SpoPointPriorEvaluator(SpoPointEvaluator):
    def __init__(self,
                 subject_model: Model,
                 object_model: Model,
                 dev_data: SpoDataPack,
                 tokenizer: BertTokenizer,
                 searcher: SpoSearcher,
                 threshold_sub_start=0.5,
                 threshold_sub_end=0.5,
                 threshold_obj_start=0.5,
                 threshold_obj_end=0.5,
                 save_path=None,
                 polarity='upper'):
        super().__init__(
            subject_model=subject_model,
            object_model=object_model,
            dev_data=dev_data,
            tokenizer=tokenizer,
            threshold_sub_start=threshold_sub_start,
            threshold_sub_end=threshold_sub_end,
            threshold_obj_start=threshold_obj_start,
            threshold_obj_end=threshold_obj_end,
            save_path=save_path,
            polarity=polarity,
        )
        self._searcher = searcher

    def extract_spoes(self, text):
        tokens, mapping = self._tokenizer.match_tokenize(text)
        token_ids, segment_ids = self._tokenizer.transform(text)

        prior_spoes = dict()
        for prior_s, prior_p, prior_o in self._searcher.extract(text):
            s_idx, s_len = self._search_start_index(prior_s, token_ids)
            o_idx, o_len = self._search_start_index(prior_o, token_ids)
            pid = self._dev_data.schema2id[prior_p]
            if s_idx != -1 and o_idx != -1:
                s_key = (s_idx, s_idx + s_len - 1)
                o_key = (o_idx, o_idx + o_len - 1, pid)
                if s_key not in prior_spoes:
                    prior_spoes[s_key] = []
                prior_spoes[s_key].append(o_key)

        prior_subjects_ids = np.zeros(shape=(len(token_ids), 2))
        if prior_spoes:
            s_start_ids, s_end_ids = list(zip(*prior_spoes.keys()))
            prior_subjects_ids[s_start_ids, 0] = 1
            prior_subjects_ids[s_end_ids, 1] = 1

        subject_preds = self._subject_model.predict([[token_ids], [segment_ids], [prior_subjects_ids]])

        # get subjects
        start = np.where(subject_preds[0, :, 0] > self._threshold_sub_start)[0]  # index of the start token of subject
        end = np.where(subject_preds[0, :, 1] > self._threshold_sub_end)[0]
        subjects, prior_objs = [], []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                sub = (i, j[0])
                subjects.append(sub)
                pri_obj = np.zeros(shape=(len(token_ids), self._dev_data.num_predicate, 2))
                for o in prior_spoes.get(sub, []):
                    pri_obj[o[0], o[2], 0] = 1
                    pri_obj[o[1], o[2], 0] = 1
                pri_obj = pri_obj.reshape((len(token_ids), -1))
                prior_objs.append(pri_obj)

        if subjects:
            spoes = []
            num_subjects = len(subjects)

            token_ids = np.repeat([token_ids], repeats=num_subjects, axis=0)
            segment_ids = np.repeat([segment_ids], repeats=num_subjects, axis=0)
            subjects = np.array(subjects)
            prior_object_ids = np.array(prior_objs)

            object_preds = self._object_model.predict([token_ids, segment_ids, subjects, prior_object_ids])

            for subject, object_pred in zip(subjects, object_preds):
                sub_start, sub_end = subject
                start = np.where(object_pred[:, :, 0] > self._threshold_obj_start)
                end = np.where(object_pred[:, :, 1] > self._threshold_obj_end)

                for start_idx, p_idx1 in zip(*start):
                    for end_idx, p_idx2 in zip(*end):
                        if p_idx1 == p_idx2 and 0 < start_idx <= end_idx:
                            spoes.append((
                                (mapping[sub_start][0], mapping[sub_end][-1]),
                                p_idx1,
                                (mapping[start_idx][0], mapping[end_idx][-1])
                            ))

            # list of (subject_text, predicate_name, object_text)
            res = [(text[s[0]: s[1] + 1], self._dev_data.id2schema[p], text[o[0]: o[1] + 1]) for s, p, o in spoes]
            return res
        return []

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
