# coding: utf-8

"""
@File   : embedding.py
@Author : garnet
@Time   : 2020/5/28 11:27
"""

import keras
import keras.backend as K


class RelativePositionEmbedding(keras.layers.Layer):
    """
    Relative position encoding of two sequences with embedding matrix.

    See [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) for details.
    """

    def __init__(self, input_dim, output_dim, embedding_initializer='zeros', **kwargs):
        """
        :param input_dim: max relative index distance
        :param output_dim: output size
        :param embedding_initializer: embedding initializer
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_initializer = keras.initializers.get(embedding_initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='relative_position_embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embedding_initializer,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pos_idx = self.compute_mask(*inputs)
        return K.gather(self.embeddings, pos_idx)

    def compute_relative_position(self, seq1, seq2):
        # Compute relative position
        idx1 = K.expand_dims(K.arange(0, K.shape(seq1)[1], dtype='int32'), axis=1)  # (seq_length1, 1)
        idx2 = K.expand_dims(K.arange(0, K.shape(seq2)[1], dtype='int32'), axis=0)  # (1, seq_length2)
        pos_idx = idx2 - idx1  # (seq_length1, seq_length2)

        # Adjust negative relative index to positive, meanwhile move 0 to the middle of relative position index
        max_position = (self.input_dim - 1) // 2
        pos_idx = K.clip(pos_idx, -max_position, max_position)
        pos_idx += max_position
        return pos_idx

    def compute_output_shape(self, input_shape):
        return input_shape[0][1], input_shape[1][1], self.output_dim  # seq_length1, seq_length2, output_size

    def compute_mask(self, inputs, mask=None):
        return mask[0] if mask else mask

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embedding_initializer': keras.initializers.serialize(self.embedding_initializer)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class


custom_objects = {
    'RelativePositionEmbedding': RelativePositionEmbedding,
}

keras.utils.get_custom_objects().update(custom_objects)
