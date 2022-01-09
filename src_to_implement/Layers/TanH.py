from src_to_implement.Layers import Base
import numpy as np


class TanH(Base.BaseLayer):
    def __init__(self):
        self.out_TanH = None
        super().__init__()

    def forward(self, input_tensor):
        self.out_TanH = np.tanh(input_tensor)
        return self.out_TanH

    def backward(self, error_tensor):
        return (1 - self.out_TanH ** 2) * error_tensor

    def set_out_tanh(self, value):
        self.out_TanH = value


