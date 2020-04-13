# coding: utf-8

"""
@File   : feed_forward.py
@Author : garnet
@Time   : 2020/4/13 21:53
"""

import keras
import keras.backend as K


class FeedForwardLayer(keras.layers.Layer):
    """
    Feed forward layer. Equivalent to double dense layers
    """

    def __init__(self,
                 hidden_size,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        """
        :param hidden_size: Internal projection size
        :param activation: Activation of first dense layer. The second dense layer has no activation function
        :param use_bias: Whether the dense layer use bias
        :param kernel_initializer: Kernel initializer of both layers
        :param kwargs:
        """
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.hidden_size = hidden_size
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.first_dense = keras.layers.Dense(
            units=self.hidden_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        self.second_dense = keras.layers.Dense(
            units=input_shape[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        super(FeedForwardLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.first_dense(inputs)
        x = self.second_dense(x)
        return x

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForwardLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'FeedForwardLayer': FeedForwardLayer,
}

keras.utils.get_custom_objects().update(custom_objects)
