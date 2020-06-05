# coding: utf-8

"""
@File   : attention.py
@Author : garnet
@Time   : 2020/4/13 14:41
"""

import keras
import tensorflow as tf
import keras.backend as K

from ..backend import sequence_masking


class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-head attention layer. See [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 num_heads,
                 head_size,
                 key_size=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        """
        :param num_heads: Number of attention heads
        :param head_size: Size of single head output
        :param key_size: Size of query and key vector, which is used when the length of key vector and value vector
            are different. See [Low-Rank Bottleneck in Multi-head Attention Models](https://arxiv.org/abs/2002.07028)
            for more information.
        :param kernel_initializer: Regularizer of the projection matrix weight
        :param use_bias: Whether use bias term
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True

        self.num_heads = num_heads
        self.head_size = head_size
        self.key_size = key_size or head_size
        self.out_dim = num_heads * head_size
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.q_dense = keras.layers.Dense(
            units=self.num_heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        self.k_dense = keras.layers.Dense(
            units=self.num_heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        self.v_dense = keras.layers.Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        self.o_dense = keras.layers.Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, a_mask=None, **kwargs):
        """
        :param mask: tuple with 3 elements, which represent masks of queries, keys and values
            - q_mask: mask of query input, set the padding part of the output to 0
            - k_mask: mask of key input
            - v_mask: mask of value input, prevent attention calculation from including input padding information
        :param a_mask: mask of attention score tensor
        """

        """
        q: (batch_size, query_len, hidden_query)
        k: (batch_size, key_len, hidden_key)
        v: (batch_size, key_len, hidden_value)
        """
        q, k, v = inputs[:3]

        q_mask, k_mask, v_mask = None, None, None
        if mask is not None:
            q_mask, k_mask, v_mask = mask[0], mask[1], mask[2]

        if q_mask is not None:
            q_mask = K.cast(q_mask, K.floatx())  # (batch_size, query_len)
        if k_mask is not None:
            k_mask = K.cast(k_mask, K.floatx())  # (batch_size, key_len)
        if v_mask is not None:
            v_mask = K.cast(v_mask, K.floatx())  # (batch_size, key_len)
        if a_mask is not None:
            a_mask = K.cast(a_mask, K.floatx())  # (batch_size, query_len, key_len)

        qw = self.q_dense(q)  # (batch_size, query_len, num_heads * key_size)
        kw = self.k_dense(k)  # (batch_size, key_len, num_heads * key_size)
        vw = self.v_dense(v)  # (batch_size, key_len, num_heads * head_size)
        qw = K.reshape(qw, (
            -1, K.shape(q)[1], self.num_heads, self.key_size))  # (batch_size, query_len, num_heads, key_size)
        kw = K.reshape(kw,
                       (-1, K.shape(k)[1], self.num_heads, self.key_size))  # (batch_size, key_len, num_heads, key_size)
        vw = K.reshape(vw, (
            -1, K.shape(v)[1], self.num_heads, self.head_size))  # (batch_size, key_len, num_heads, head_size)

        # Attention score(scale-dot method)
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)  # (batch_size, num_heads, query_len, key_len)
        a = a / self.key_size ** 0.5  # scale
        a = sequence_masking(a, v_mask, mode='add', axis=-1)
        if a_mask is not None:
            a -= (1 - a_mask) * 1e12
        a = K.softmax(a, axis=-1)

        # Output
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)  # (batch_size, query_len, num_heads, head_size)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))  # (batch_size, query_len, num_heads * head_size)
        o = self.o_dense(o)
        o = sequence_masking(o, q_mask, mode='mul', axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.out_dim

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'out_dim': self.out_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'MultiHeadAttention': MultiHeadAttention,
}

keras.utils.get_custom_objects().update(custom_objects)
