import copy

from src_to_implement.Layers import Base
import numpy as np
from src_to_implement.Layers import TanH, Sigmoid, FullyConnected


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = None
        self.belong_to_long_sequence = False
        self.first_state = np.zeros((1, hidden_size))
        self.hidden_state = None
        self.fully_connected_states_tanh = None
        self.gradient_weights_fc_state = None
        self.fullyConnected_input_tensor = None
        self._optimizer = None
        self.tanH = TanH.TanH()
        self.Sigmoid = Sigmoid.Sigmoid()
        self.fullyConnected_output = FullyConnected.FullyConnected(hidden_size, output_size)
        self.fullyConnected_states = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.hidden_state = np.zeros((self.batch_size, self.hidden_size))
        self.fully_connected_states_tanh = np.zeros((self.batch_size, self.hidden_size))
        self.fullyConnected_input_tensor = np.zeros((self.batch_size, self.hidden_size+self.input_size+1))
        if self.belong_to_long_sequence:
            self.hidden_state[0, :] = self.first_state
        else:
            self.first_state = np.zeros((1, self.hidden_size))
        for batch in range(self.batch_size):
            state_input = np.append(self.first_state.reshape(1, self.hidden_size), input_tensor[batch, :].reshape(1, self.input_size), 1)
            fully_connected_state_output = self.fullyConnected_states.forward(state_input)
            self.fullyConnected_input_tensor[batch, :] = self.fullyConnected_states.get_input_tensor()
            fully_connected_states = self.tanH.forward(fully_connected_state_output)
            self.fully_connected_states_tanh[batch, :] = fully_connected_states
            self.hidden_state[batch, :] = fully_connected_states
            self.first_state = fully_connected_states
        output = self.fullyConnected_output.forward(self.hidden_state)
        output = self.Sigmoid.forward(output)
        return output

    def backward(self, error_tensor):
        self.gradient_weights_fc_state = 0
        bp_output = np.zeros((self.batch_size, self.input_size))
        bp_out_tensor = self.Sigmoid.backward(error_tensor)
        bp_out_tensor = self.fullyConnected_output.backward(bp_out_tensor)
        bp_tensor_tanh_befor = np.zeros((1, self.hidden_size))
        for batch in reversed(range(self.batch_size)):
            # because we call forward several times we must again initialize tanH out put for each forward
            self.tanH.set_out_tanh(self.fully_connected_states_tanh[batch, :])
            # every state output relate to the previous inputs
            bp_tensor_tanh = bp_out_tensor[batch, :].reshape(1, self.hidden_size) + bp_tensor_tanh_befor.reshape(1, self.hidden_size)
            bp_states = self.tanH.backward(bp_tensor_tanh)
            # because we call forward several times we must again initialize input tensor for each backward
            self.fullyConnected_states.set_input_tensor(self.fullyConnected_input_tensor[batch, :].reshape(1, self.input_size+self.hidden_size+1))
            bp_states_and_input = self.fullyConnected_states.backward(bp_states)
            bp_tensor_tanh_befor = bp_states_and_input[0, 0:self.hidden_size]
            bp_output[batch, :] = bp_states_and_input[0, self.hidden_size:]
            self.gradient_weights_fc_state += self.fullyConnected_states.get_gradient_weights()
        if self._optimizer is not None:
            weights_fullyconnected_states = self.fullyConnected_states.get_weights()
            weights_fullyconnected_states = self._optimizer.calculate_update(weights_fullyconnected_states, self.gradient_weights_fc_state)
            self.fullyConnected_states.set_weights(weights_fullyconnected_states)
        return bp_output

    def set_belong_to_long_sequence(self, value):
        self.belong_to_long_sequence = value

    def get_belong_to_long_sequence(self):
        return self.belong_to_long_sequence

    memorize = property(get_belong_to_long_sequence, set_belong_to_long_sequence)

    def initialize(self, weights_initializer, bias_initializer):
        self.fullyConnected_states.initialize(weights_initializer, bias_initializer)
        self.fullyConnected_output.initialize(weights_initializer, bias_initializer)

    def get_weights(self):
        return self.fullyConnected_states.get_weights()

    def set_weights(self, values):
        self.fullyConnected_states.set_weights(values)

    weights = property(get_weights, set_weights)

    def get_gradient_weights(self):
        return self.gradient_weights_fc_state

    def set_gradient_weights(self, value):
        self.gradient_weights_fc_state = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, value):
        self._optimizer = value
        self.fullyConnected_output.optimizer = copy.deepcopy(value)

    optimizer = property(get_optimizer, set_optimizer)
