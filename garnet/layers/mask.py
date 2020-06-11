# coding: utf-8

"""
@File   : mask.py
@Author : garnet
@Time   : 2020/6/5 11:00
"""

import keras
import keras.backend as K
from keras.layers import Layer


class MaskRemove(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        return None


class MaskExtract(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            raise ValueError("Last layer doesn't contain mask tensor")
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return [None for _ in input_shape]
        else:
            return None


custom_objects = {
    'MaskRemove': MaskRemove,
    'MaskExtract': MaskExtract,
}

keras.utils.get_custom_objects().update(custom_objects)
