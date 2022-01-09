from src_to_implement.Layers import Base
import numpy as np
import copy
from src_to_implement.Layers import Helpers
import sys


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.batch_size = 0
        self.channel_size = 0
        self.image_H = 0
        self.image_W = 0
        self.mean = 0
        self.var = 0
        self.trainable = True
        # weights
        self.gamma = np.ones((1, channels))
        # bias
        self.beta = np.zeros((1, channels))
        # Moving average decay
        self.alpha = 0.8
        self.epsilon = np.finfo(float).eps
        self.first_step = True
        self.input_tensor = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.normed_input = None
        self.weight_optimizer = None
        self.bias_optimizer = None
        self.mean_vector = None
        self.var_vector = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if not self.testing_phase:
            # Convolutional layer
            if np.size(input_tensor.shape) > 2:
                self.batch_size, self.channel_size, self.image_H, self.image_W = input_tensor.shape
                reformat_image_tensor = self.reformat(input_tensor)
                normed_input = self.var_mean_calculation(reformat_image_tensor)
                reformat_normed_input_tensor = self.reformat(normed_input)
                return reformat_normed_input_tensor
            # fully connected layer
            else:
                normed_input = self.var_mean_calculation(input_tensor)
                return normed_input
        # Test Phase --------------------------------
        else:
            if np.size(input_tensor.shape) > 2:
                self.batch_size, self.channel_size, self.image_H, self.image_W = input_tensor.shape
                reformat_image_tensor = self.reformat(input_tensor)
                normed_input = (reformat_image_tensor - self.mean)/np.sqrt(self.var + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                reformat_normed_input_tensor = self.reformat(normed_input)
                return reformat_normed_input_tensor
            # fully connected layer
            else:
                normed_input = (input_tensor - self.mean)/np.sqrt(self.var + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                return normed_input

    def reformat(self, image_tensor):
        if np.size(image_tensor.shape) > 2:
            reformat_image_tensor = image_tensor.reshape((image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]*image_tensor.shape[3]))
            reformat_image_tensor = reformat_image_tensor.transpose(1, 0, 2)
            reformat_image_tensor = reformat_image_tensor.reshape(reformat_image_tensor.shape[0], reformat_image_tensor.shape[1]*reformat_image_tensor.shape[2])
            reformat_image_tensor = reformat_image_tensor.transpose()
            return reformat_image_tensor
        else:
            reformat_vector_tensor = image_tensor.transpose()
            reformat_vector_tensor = reformat_vector_tensor.reshape(reformat_vector_tensor.shape[0], self.batch_size, self.image_H * self.image_W)
            reformat_vector_tensor = reformat_vector_tensor.transpose(1, 0, 2)
            reformat_vector_tensor = reformat_vector_tensor.reshape(self.batch_size, self.channel_size, self.image_H, self.image_W)
            return reformat_vector_tensor

    def var_mean_calculation(self, input_vector):
        mean = np.mean(input_vector, axis=0)
        mean_vector = mean.reshape(1, input_vector.shape[1])
        self.mean_vector = mean_vector
        var = np.var(input_vector, axis=0)
        var_vector = var.reshape(1, input_vector.shape[1])
        self.var_vector = var_vector
        # Moving average estimation
        if self.first_step:
            self.first_step = False
            self.mean = mean_vector
            self.var = var_vector
        self.mean = (1 - self.alpha) * mean_vector + self.alpha * self.mean
        self.var = (1 - self.alpha) * var_vector + self.alpha * self.var
        # -----------------------------
        normed_input = (input_vector - mean_vector) / np.sqrt(var_vector + self.epsilon)
        self.normed_input = normed_input
        normed_input = self.gamma * normed_input + self.beta
        return normed_input

    def backward(self, error_tensor):
        # Convolutional layer
        if np.size(self.input_tensor.shape) > 2:
            reformat_error_tensor = self.reformat(error_tensor)
            reformat_input_tensor = self.reformat(self.input_tensor)
            bp_error_tensor = Helpers.compute_bn_gradients(reformat_error_tensor, reformat_input_tensor, self.gamma, self.mean_vector, self.var_vector)
            reformat_bp_error_tensor = self.reformat(bp_error_tensor)
            self._gradient_weights = np.sum(reformat_error_tensor * self.normed_input, axis=0).reshape(1, error_tensor.shape[1])
            self._gradient_bias = np.sum(reformat_error_tensor, axis=0).reshape(1, error_tensor.shape[1])
            if self.weight_optimizer is not None:
                self.gamma = self.weight_optimizer.calculate_update(self.gamma, self._gradient_weights)
                self.beta = self.bias_optimizer.calculate_update(self.beta, self._gradient_bias)
            return reformat_bp_error_tensor
            # fully connected layer
        else:
            bp_error_tensor = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.gamma, self.mean_vector, self.var_vector)
            self._gradient_weights = np.sum(error_tensor * self.normed_input, axis=0).reshape(1, error_tensor.shape[1])
            self._gradient_bias = np.sum(error_tensor, axis=0).reshape(1, error_tensor.shape[1])
            if self.weight_optimizer is not None:
                self.gamma = self.weight_optimizer.calculate_update(self.gamma, self._gradient_weights)
                self.beta = self.bias_optimizer.calculate_update(self.beta, self._gradient_bias)
            return bp_error_tensor

    def get_weights(self):
        return self.gamma

    def set_weights(self, value):
        self.gamma = value

    weights = property(get_weights, set_weights)

    def get_bias(self):
        return self.beta

    def set_bias(self, value):
        self.beta = value

    bias = property(get_bias, set_bias)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, value):
        self._gradient_weights = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_gradient_bias(self, value):
        self._gradient_bias = value

    gradient_bias = property(get_gradient_bias, set_gradient_bias)

    def get_optimizer(self):
        return self.weight_optimizer

    def set_optimizer(self, value):
        self.weight_optimizer = value
        self.bias_optimizer = copy.deepcopy(value)

    optimizer = property(get_optimizer, set_optimizer)

    def initialize(self, weight_initializer, bias_initializer):
        pass


