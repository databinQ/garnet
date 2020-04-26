# coding: utf-8

"""
@File   : sequence.py
@Author : garnet
@Time   : 2020/4/26 16:04
"""

import numpy as np


def sequence_padding(inputs, max_length=None, padding='post', truncate='pre', padding_index=0):
    """
    Padding or truncate sequence samples into same length.

    :param inputs: sequence samples, usually `list`
    :param max_length: max length of sequence. If this is `None`, it will be inferred from `inputs` as the length of
        longest sequence
    :param padding: 'pre' or 'post'
    :param truncate: 'pre' or 'post'
    :param padding_index: index of padding token [PAD], or a vector that represent pad token
    """

    max_length = max_length or max(len(sample) for sample in inputs)

    outputs = np.array(
        [
            (
                np.concatenate([[padding_index] * (max_length - len(sample)), sample])
                if padding == 'pre' else
                np.concatenate([sample, [padding_index] * (max_length - len(sample))])
            )
            if len(sample) < max_length else
            (
                sample[:max_length]
                if truncate == 'pre' else
                sample[-max_length:]
            )
            for sample in inputs
        ],
    )
    return outputs
