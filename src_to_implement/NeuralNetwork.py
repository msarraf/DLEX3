from Layers import *
from Optimization import *
import copy


class NeuralNetwork:
    def __init__(self, optimizer: Optimizers.Sgd, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            output_tensor = layer.forward(self.input_tensor)
            self.input_tensor = output_tensor
        final_loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        self.loss.append(final_loss)
        return final_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()
        print(self.label_tensor)

    def test(self, input_tensor):
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
        return input_tensor

