# coding: utf-8

"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/5/10 8:50
"""

import keras
import pathlib


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path=None, polarity='upper', *args, **kwargs):
        super().__init__()
        self._polarity = polarity
        self.best_metric_value = 0. if self.polarity else 1e10
        self._save_path = pathlib.Path(save_path) if save_path is not None else None

    @property
    def polarity(self):
        return True if self._polarity == 'upper' else False

    def better(self, value):
        if self.polarity:
            return True if value > self.best_metric_value else False
        else:
            return True if value < self.best_metric_value else False

    def update_metric(self, value):
        self.best_metric_value = value

    def auto_update_metric(self, value):
        if self.better(value):
            self.update_metric(value)

    def save_model(self):
        if self._save_path is not None:
            self.model.save_weights(self._save_path)
        else:
            raise ValueError("A directory must be assigned before saving model weights")
