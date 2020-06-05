# coding: utf-8

"""
@File   : mask.py
@Author : garnet
@Time   : 2020/6/5 11:00
"""

import keras
from keras.layers import Layer


class MaskRemove(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        return None


custom_objects = {
    'MaskRemove': MaskRemove,
}

keras.utils.get_custom_objects().update(custom_objects)
