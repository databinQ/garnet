# coding: utf-8

"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/5/10 8:50
"""

import keras


class Evaluator(keras.callbacks.Callback):
    def __init__(self, polarity='upper', *args, **kwargs):
        super().__init__()
        self._polarity = polarity
        self.best_metric_value = 0.

    @property
    def polarity(self):
        return True if self._polarity == 'upper' else False

    def evaluate(self):
        raise NotImplementedError()
