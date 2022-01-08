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
        self.tanH = TanH.TanH()
        self.Sigmoid = Sigmoid.Sigmoid()
        self.fullyConnected_output = FullyConnected.FullyConnected(hidden_size, output_size)
        self.fullyConnected_states = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.hidden_state = np.zeros((self.batch_size, self.hidden_size))
        if self.belong_to_long_sequence:
            self.hidden_state[0, :] = self.first_state
        for batch in range(self.batch_size):
            state_input = np.append(self.first_state, input_tensor[batch, :].reshape(1, self.input_size), 1)
            fully_connected_states = self.fullyConnected_states.forward(state_input)
            fully_connected_states = self.tanH.forward(fully_connected_states)
            self.hidden_state[batch, :] = fully_connected_states
            self.first_state = fully_connected_states
        output = self.fullyConnected_output.forward(self.hidden_state)
        output = self.Sigmoid.forward(output)
        return output

    def backward(self, error_tensor):
        bp_output = np.zeros((self.batch_size, self.input_size))
        bp_out_tensor = self.Sigmoid.backward(error_tensor)
        bp_out_tensor = self.fullyConnected_output.backward(bp_out_tensor)
        bp_tensor_tanh_befor = np.zeros((1, self.hidden_size))
        for batch in reversed(range(self.batch_size)):
            self.tanH.out_TanH = self.hidden_state[batch, :]
            # every state output relate to the previous inputs
            bp_tensor_tanh = bp_out_tensor[batch, :] + bp_tensor_tanh_befor
            bp_states = self.tanH.backward(bp_tensor_tanh)
            bp_states = self.fullyConnected_states.backward(bp_states)
            bp_tensor_tanh_befor = bp_states
            bp_output[batch, :] = bp_states

    def set_belong_to_long_sequence(self, value):
        self.belong_to_long_sequence = value

    def get_belong_to_long_sequence(self):
        return self.belong_to_long_sequence

    memorize = property(get_belong_to_long_sequence, set_belong_to_long_sequence)