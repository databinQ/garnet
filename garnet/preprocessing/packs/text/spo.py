# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/4/23 21:18
"""

from . import TextDataPack
from .. import ClassifyDataPackMixin


class SpoDataPack(TextDataPack, ClassifyDataPackMixin):
    def __init__(self, data, schema):
        """
        SPO means (subject, predicate, object) triples, which is usually the output of relation extraction or
        information extraction task.

        :param data: list. Each element is a sample in `dict` format, which contains key `text` and `spo_list`.
            `spo_list` is the list of all spo triples in the text of this sample, element of `spo_list` is a `tuple`
            in (subject, predicate, object) format.
            Noting when spo triple is a complex
        :param schema: schema of spo triples. Schema may contain complex relation, which the object of particular
            predicate has more than one attribute.
        """
        self.schema = schema
        self.schema2id, self.id2schema, self.flatten2complex = self.parse_schema(schema)
        self.data = self.parse_data(data)

    def parse_data(self, data):
        assert data is not None and len(data) > 0, "Input data is empty"
        assert 'text' in data[0], "Each sample must contain key `text`"

        if 'spo_list' not in data[0]:  # test data
            return [{'text': sample['text']} for sample in data]

        for sample in data:
            spo_list = sample['spo_list']



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
