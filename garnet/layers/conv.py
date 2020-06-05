# coding: utf-8

"""
@File   : conv.py
@Author : garnet
@Time   : 2020/6/5 11:05
"""

import keras
import keras.backend as K


class MaskedConv1D(keras.layers.Conv1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]
            )
        else:
            mask = K.expand_dims(mask, axis=-1) if K.ndim(mask) == 2 else mask
            inputs *= mask
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask
