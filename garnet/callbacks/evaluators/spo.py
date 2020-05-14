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

    def extract_spoes(self, text):
        tokens = self._tokenizer.tokenize(text)
        mapping = self._tokenizer.rematch(text, tokens)
        token_ids, segment_ids = self._tokenizer.transform(text)

        subject_preds = self._subject_model.predict([[token_ids], [segment_ids]])
        start = np.where(subject_preds[0, :, 0] > self._threshold_sub_start)[0]  # index of the start token of subject
        end = np.where(subject_preds[0, :, 1] > self._threshold_sub_end)[0]

        # get subjects
        subjects = []
        for i in start:
            j = end[end > i]
            if len(j) > 0:
                subjects.append((i, j[0]))

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
                        if p_idx1 == p_idx2 and start_idx <= end_idx:
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

    def evaluate(self):
        pass

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
    def extract_spoes(self, text):
        pass
