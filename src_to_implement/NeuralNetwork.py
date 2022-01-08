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
        reg_loss = 0
        for layer in self.layers:
            # calculate regularization loss for each trainable layer
            if self.optimizer.regularizer is not None:
                if layer.trainable is True:
                    if isinstance(layer, FullyConnected.FullyConnected):
                        layer_weights = layer.weights
                    else:
                        layer_weights = layer.weights[:, :-1]
                    reg_loss += layer.optimizer.regularizer.norm(layer_weights)
            # ---------------------------------------------------------
            output_tensor = layer.forward(self.input_tensor)
            self.input_tensor = output_tensor
            # add regularization effect for the final loss calculation
        prediction_tensor = self.input_tensor + reg_loss
        final_loss = self.loss_layer.forward(prediction_tensor, self.label_tensor)
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
        self.phase = False
        for i in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
        return input_tensor

    def set_phase(self, value):
        for layer in self.layers:
            layer.testing_phase = value

    def get_phase(self):
        pass
    phase = property(get_phase, set_phase)
