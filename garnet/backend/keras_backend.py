# coding: utf-8

"""
@File   : keras_backend.py
@Author : garnet
@Time   : 2020/4/13 16:08
"""

import keras.backend as K


def sequence_masking(x, mask, mode='mul', axis=1):
    """
    Mask sequence tensor.

    :param x: Sequence tensor. Usually axis `1` stands for sequence index
    :param mask: Mask of sequence, with shape (batch_size, seq_len) or (batch_size, seq_len, 1)
    :param mode: `mul` or `add`
    :param axis: Axis of sequence index
    """
    if mask is None:
        return x

    if axis == -1:
        axis = K.ndim(x) - 1
    for _ in range(axis - 1):
        mask = K.expand_dims(mask, 1)
    for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
        mask = K.expand_dims(mask, K.ndim(mask))
    if mode == 'mul':
        return x * mask
    else:
        return x - (1 - mask) * 1e12
