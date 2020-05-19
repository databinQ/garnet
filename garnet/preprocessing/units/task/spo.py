# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/5/5 15:28
"""

from tqdm import tqdm

from .. import StateUnit
from ....utils.ac import ACUnicode
from ...packs.text.spo import SpoDataPack


class SpoSearcher(StateUnit):
    def __init__(self):
        super().__init__()
        self._data_pack = None
        self.s_ac = ACUnicode()
        self.o_ac = ACUnicode()
        self.so2p = dict()

    def fit(self, train_data: SpoDataPack):
        self._data_pack = train_data

        for _, spo_list in tqdm(self._data_pack):
            for s, p, o in spo_list:
                if not self.s_ac.exists(s):
                    self.s_ac.add_word(s, s)
                if not self.o_ac.exists(o):
                    self.o_ac.add_word(o, o)

                so = (s, o)
                if so not in self.so2p:
                    self.so2p[so] = set()
                self.so2p[so].add(p)

        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

        super().fit(train_data)

    def extract(self, text, excludes=None):
        result = set()
        for s in self.s_ac.iter(text):
            s = s[1]
            for o in self.o_ac.iter(text):
                o = o[1]
                so = (s, o)
                if so in self.so2p:
                    for p in self.so2p[so]:
                        spo = (s, p, o)
                        if excludes is None:
                            result.add(spo)
                        elif spo not in excludes:
                            result.add(spo)
        return list(result)

    def transform(self, input_, excludes=None):
        return self.extract(input_, excludes=excludes)
