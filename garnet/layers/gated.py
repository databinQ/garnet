# coding: utf-8

"""
@File   : gated.py
@Author : garnet
@Time   : 2020/6/3 22:43
"""

import copy
import keras
import keras.backend as K
from keras.layers.wrappers import Wrapper


class GatedConvBlock(Wrapper):
    def __init__(self, conv_layer, num_layers=1, gate_activation='sigmoid', **kwargs):
        if conv_layer.padding != 'same':
            raise ValueError("The padding mode of this layer must be `same`, but got {}".format(conv_layer.padding))

        super().__init__(conv_layer, **kwargs)
        self.supports_masking = True

        self.num_layers = num_layers
        self.gate_activation = keras.activations.get(gate_activation)

        self.rank = conv_layer.rank  # Convolution dimension rank
        self.input_spec = conv_layer.input_spec
        self.padding = conv_layer.padding

        # Number of input convolution layer's channel must be double size comparing with output channel.
        self.filters = conv_layer.filters // 2

        self.conv_layers = []
        for i in range(self.num_layers):
            new_conv_layer = copy.deepcopy(conv_layer)
            new_conv_layer.name = 'GatedConvBlock_{}_{}'.format(conv_layer.name, i)
            self.conv_layers.append(new_conv_layer)

    def build(self, input_shape=None):
        current_input_shape = input_shape
        for layer in self.conv_layers:
            with K.name_scope(layer.name):
                layer.build(current_input_shape)
            current_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, mask=None):
        current_inputs = inputs
        for i, layer in enumerate(self.conv_layers):
            layer_output = layer(current_inputs)
            linear_output, gated_output = self.half_slice(layer_output)
            current_inputs = linear_output * self.gate_activation(gated_output)
            current_inputs._keras_shape = K.int_shape(linear_output)

        # residual connection
        return inputs + current_inputs

    def half_slice(self, x):
        if self.rank == 1:
            linear_output, gated_output = x[:, :, :self.filters], x[:, :, self.filters:]
        elif self.rank == 2:
            linear_output, gated_output = x[:, :, :, :self.filters], x[:, :, :, self.filters:]
        elif self.rank == 3:
            linear_output, gated_output = x[:, :, :, :, :self.filters], x[:, :, :, :, self.filters:]
        else:
            raise ValueError("Only 1D, 2D, 3D convolutions are supported, got {}".format(self.rank))
        return linear_output, gated_output

    def compute_output_shape(self, input_shape):
        current_shape = input_shape
        for layer in self.conv_layers:
            layer_output = layer.compute_output_shape(current_shape)
            layer_output = list(layer_output)
            layer_output[-1] = layer_output[-1] // 2
            current_shape = layer_output
        return tuple(current_shape)

    def get_weights(self):
        weights = None
        for layer in self.conv_layers:
            if weights is None:
                weights = layer.get_weights()
            else:
                weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for layer in self.conv_layers:
            layer.set_weights(weights)

    @property
    def trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'trainable_weights'):
                weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'non_trainable_weights'):
                weights += layer.non_trainable_weights
        return weights

    @property
    def updates(self):
        updates_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'updates'):
                updates_ += layer.upates
        return updates_

    @property
    def losses(self):
        losses_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'losses'):
                losses_ += layer.losses
        return losses_

    @property
    def constraints(self):
        constraints_ = {}
        for layer in self.conv_layers:
            if hasattr(layer, 'constraints'):
                constraints_.update(layer.constraints)
        return constraints_
