# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/4/23 21:18
"""

import typing
import numpy as np

from .. import DataPack
from ..mixin import TextMixin, ClassificationMixin


class SpoDataPack(ClassificationMixin, TextMixin, DataPack):
    def __init__(self, data, schema):
        """
        SPO means (subject, predicate, object) triples, which is usually the output of relation extraction or
        information extraction task.

        :param data: list. Each element is a sample in `dict` format, which contains key `text` and `spo_list`.
            `spo_list` is the list of all spo triples in the text of this sample, element of `spo_list` is a `tuple`
            in (subject, predicate, object) format.
            1. For training data, if text doesn't contain SPO triple, the value of `spo_list` should be empty list.
            2. Object in SPO triple can be a complex dict, which has multi keys.
        :param schema: schema of spo triples. Schema may contain complex relation, which the object of particular
            predicate has more than one attribute.
        """
        self.schema = schema
        self.schema2id, self.id2schema, self.flatten2complex = self.parse_schema(schema)
        self.data, self.is_train = self.parse_data(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            yield sample.get('text'), sample.get('spo_list')

    def __getitem__(self, item: typing.Union[int, slice, list, np.array]):
        if isinstance(item, int):
            return self.data[item].get('text'), self.data[item].get('spo_list')
        elif isinstance(item, slice):
            texts, spoes = [], []
            for index in list(range(*item.indices(len(self)))):
                texts.append(self.data[index].get('text'))
                spoes.append(self.data[index].get('spo_list'))
            return texts, spoes if self.with_label else None
        else:
            texts, spoes = [], []
            for index in item:
                texts.append(self.data[index].get('text'))
                spoes.append(self.data[index].get('spo_list'))
            return texts, spoes if self.with_label else None

    @property
    def with_label(self) -> bool:
        return self.is_train

    def shuffle(self):
        self.data = self._shuffle(self.data)

    def unpack(self):
        X, y = list(zip(*[(sample.get('text'), sample.get('spo_list')) for sample in self.data]))
        y = None if None in y else y
        return X, y

    def apply(self, func: typing.Callable, *args, **kwargs):
        new_texts = self.apply_on_text([sample['text'] for sample in self.data], func=func)
        for i in range(len(new_texts)):
            self.data[i]['text'] = new_texts[i]

    @staticmethod
    def parse_data(data):
        assert data is not None and len(data) > 0, "Input data is empty"
        assert 'text' in data[0], "Each sample must contain key `text`"

        if 'spo_list' not in data[0]:  # test data
            return [{'text': sample['text']} for sample in data], False

        # train data
        train_data = []
        for sample in data:
            spo_list = []
            for spo in sample['spo_list']:
                subject, predicate, object_ = spo['subject'], spo['predicate'], spo['object']
                if isinstance(object_, str):
                    spo_list.append((subject, predicate, object_))
                elif isinstance(object_, dict):
                    for k, v in object_.items():
                        spo_list.append((subject, '|'.join([predicate, k]), v))
                else:
                    raise ValueError("Object of SPO triple must be a `str` or `dict`")
            train_data.append({
                'text': sample['text'],
                'spo_list': spo_list,
            })
        return train_data, True

    @staticmethod
    def parse_schema(schema):
        schema2id = dict()
        flatten2complex = dict()
        for s in schema:
            predicate = s['predicate']
            object = s['object_type']
            if isinstance(object, str):
                schema2id[predicate] = len(schema2id)
                flatten2complex[predicate] = (predicate,)
            else:
                for sub_p in object.keys():
                    new_predicate = predicate + "|" + sub_p
                    schema2id[new_predicate] = len(schema2id)
                    flatten2complex[new_predicate] = (predicate, sub_p)
        id2schema = {v: k for k, v in schema2id.items()}
        return schema2id, id2schema, flatten2complex

    @staticmethod
    def schema_restore(spoes):
        spo_map = dict()
        for s, p, o in spoes:
            p1, p2 = spoes.split('|')
            sp1 = (s, p1)
            if sp1 in spo_map:
                spo_map[sp1][p2] = o
            else:
                spo_map[sp1] = {p2: o}
        return [(k[0], k[1], v) for k, v in spo_map.items()]
