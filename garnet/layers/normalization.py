# coding: utf-8

"""
@File   : normalization.py
@Author : garnet
@Time   : 2020/4/11 11:25
"""

import keras
import keras.backend as K
from keras.layers import Dense


class LayerNormalization(keras.layers.Layer):
    """
    Layer normalization layer. See [Layer Normalization](https://arxiv.org/abs/1607.06450)
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True

        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)

    def build(self, input_shape):
        shape = input_shape[-1:]

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask


class CondLayerNormalization(keras.layers.Layer):
    """
    Conditional layer normalization
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param hidden_units: Hidden units of dense layer which transform addition condition inputs
        :param hidden_activation: Activation of hidden dense
        :param hidden_initializer: Initializer of hidden dense
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.supports_masking = True

        self.center = center
        self.scale = scale
        self.hidden_units = hidden_units
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.hidden_initializer = keras.initializers.get(hidden_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)

    def build(self, input_shape):
        shape = input_shape[0][-1:]

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )

        if self.hidden_units is not None:
            self.hidden_dense = Dense(
                units=self.hidden_units,
                activation=self.hidden_activation,
                use_bias=False,
                kernel_initializer=self.hidden_initializer,
            )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros',
                )

            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros',
                )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: list, (sequence_input, condition_input)
        """
        inputs, cond = inputs

        if self.hidden_units is not None:
            cond = self.hidden_dense(cond)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            cond = K.expand_dims(cond, axis=1)

        if self.center:
            beta = self.beta_dense(cond) + self.beta
        if self.scale:
            gamma = self.gamma_dense(cond) + self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs *= gamma
        if self.center:
            outputs += beta
        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'hidden_units': self.hidden_units,
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'hidden_initializer': keras.initializers.serialize(self.hidden_initializer),
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask


custom_objects = {
    'LayerNormalization': LayerNormalization,
    'CondLayerNormalization': CondLayerNormalization,
}

keras.utils.get_custom_objects().update(custom_objects)
