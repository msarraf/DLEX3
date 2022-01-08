import numpy as np
from src_to_implement.Layers import Base

class Flatten(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_tensor = 0
        self.error_tensor = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.input_tensor.shape) > 2:
            return self.input_tensor.reshape(self.input_tensor.shape[0], self.input_tensor.shape[1]*self.input_tensor.shape[2]*self.input_tensor.shape[3])
        else:
            return input_tensor

    def backward(self, error_tensor):
        if len(self.input_tensor.shape) > 2:
            self.error_tensor = error_tensor
            reshaped_flatten = self.error_tensor.reshape(self.input_tensor.shape)
            return reshaped_flatten
        else:
            self.error_tensor = error_tensor
            return self.error_tensor
