import numpy as np
from CNN.src_to_implement.Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = 0

    def forward(self, input_tensor):
        max_output_size = np.max(np.transpose(input_tensor), 0)
        shift_input_tensor = np.add(np.transpose(input_tensor), -1*max_output_size)
        exp_shift_input_tensor = np.exp(shift_input_tensor)
        sum_shift_input_tensor = np.sum(exp_shift_input_tensor, 0)
        output = np.divide(exp_shift_input_tensor, sum_shift_input_tensor)
        output = np.transpose(output)
        self.output = output
        return output

    def backward(self, error_tensor):
        error_tensor_prediction = np.multiply(error_tensor, self.output)
        error_tensor_prediction = np.transpose(error_tensor_prediction)
        sum_error_tensor_prediction = np.sum(error_tensor_prediction, 0)
        result = np.add(np.transpose(error_tensor), -1*sum_error_tensor_prediction)
        result = np.transpose(result)
        output = np.multiply(result, self.output)
        return output



