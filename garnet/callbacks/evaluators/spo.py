# coding: utf-8

"""
@File   : spo.py
@Author : garnet
@Time   : 2020/5/10 8:52
"""

from tqdm import tqdm
from keras.models import Model

from . import Evaluator
from ...preprocessing.packs.text.spo import SpoDataPack



class SpoEvaluator(Evaluator):
    def __init__(self, model: Model, dev_data: SpoDataPack, polarity='upper'):
        super().__init__(polarity=polarity)
        self.set_model(model)
        self.dev_data = dev_data

    def on_epoch_end(self, epoch, logs=None):
        X, Y, Z = 1e-10, 1e-10, 1e-10

        pbar = tqdm()
        for sample in self.dev_data:
            pass

    @staticmethod
    def schema_restore(spo_list):
        spo_map = dict()
        for s, p, o in spo_list:
            p1, p2 = p.split('|')
            sp1 = (s, p1)
            if sp1 not in spo_map:
                spo_map[sp1] = dict()
            spo_map[sp1][p2] = o
        return [(k[0], k[1], v) for k, v in spo_map.items()]

    def extract_spoes(self, text):


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

        def __hash__(self):
            return self.spox.__hash__()

        def __eq__(self, spo):
            return self.spox == spo.spox
