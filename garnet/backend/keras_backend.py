# coding: utf-8

"""
@File   : keras_backend.py
@Author : garnet
@Time   : 2020/4/13 16:08
"""

import tensorflow as tf
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


def batch_gather(params, indices):
    return tf.gather(params, indices, batch_dims=-1)


def batch_range(seq_input, dtype='int32'):
    """
    Create a 2D tensor containing a sequence of integers, from `0` to `sequence_length - 1`. In the same batch,
    each sample has the same sequence vector.

    :param seq_input: 2D or 3D sequence tensor
    :return: 2D tensor with shape (batch_size, sequence_length)
    """
    tensor_shape = K.shape(seq_input)
    batch_dim, seq_dim = tensor_shape[0], tensor_shape[1]

    index = K.expand_dims(K.arange(seq_dim), 0)
    index = tf.tile(index, K.stack([batch_dim, K.constant(1, dtype='int32')]))

    if dtype != 'int32':
        index = K.cast(index, dtype)
    return index


def sequence_extract(sequence, starts, ends):
    """
    Extract fragments of sequences with start and end index. Different samples have different fragment lengths.

    :param sequence: 2D or 3D sequence tensor
    :param starts: tensor of start index
    :param ends: tensor of end index
    :return:
        1. fragment tensor with the same shape of input sequence tensor, the values of steps which are out of fragment
          are zeros
        2. 2D mask tensor, values belonging to fragments are `1`, others are `0`
    """
    assert K.ndim(starts) == 1 and K.ndim(ends) == 1
    starts, ends = K.expand_dims(K.cast(starts, 'int32'), axis=1), K.expand_dims(K.cast(ends, 'int32'), axis=1)

    sequence_index = batch_range(sequence)
    mask = tf.where(
        sequence_index < starts,
        K.zeros_like(sequence_index, dtype=K.floatx()),
        K.ones_like(sequence_index, dtype=K.floatx())
    )
    mask = tf.where(
        sequence_index > ends,
        K.zeros_like(sequence_index, dtype=K.floatx()),
        mask
    )

    output = sequence * K.expand_dims(mask, axis=-1) if K.ndim(sequence) == 3 else sequence * mask
    return output, mask


def sequence_extract_mean(sequence, starts, ends):
    assert K.ndim(sequence) == 3
    output, mask = sequence_extract(sequence, starts, ends)
    return K.sum(output, axis=1) / K.sum(mask, axis=-1, keepdims=True)
