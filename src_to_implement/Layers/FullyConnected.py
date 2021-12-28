# from Base import BaseLayer
from src_to_implement.Layers import Base
from src_to_implement.Optimization import Optimizers
import numpy as np


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = None
        self.weights = np.random.rand(input_size + 1, output_size)
        self.gradient_weights = 0
        self.input_tensor = 0

    def forward(self, input_tensor):
        ones = np.ones([np.size(input_tensor[:, 0]), 1])
        input_tensor = np.append(input_tensor, ones, 1)
        output_tensor = np.matmul(input_tensor, self.weights)
        self.input_tensor = input_tensor
        return output_tensor

    def backward(self, error_tensor):
        input_tensor = np.matmul(error_tensor, np.transpose(self.weights))
        input_tensor_reduced = input_tensor[:, :-1]
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return input_tensor_reduced

    def getter_optimizer(self):
        return self.optimizer

    def setter_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        weights_initializer.fan_in = self.input_size
        weights_initializer.fan_out = self.output_size
        bias_initializer.fan_in = self.input_size
        bias_initializer.fan_out = self.output_size
        weights = weights_initializer.initialize((self.input_size, self.output_size), weights_initializer.fan_in,
                                                 weights_initializer.fan_out)
        bias = bias_initializer.initialize((1, bias_initializer.fan_out), bias_initializer.fan_in,
                                           bias_initializer.fan_out)
        self.weights = np.append(weights, bias, 0)

