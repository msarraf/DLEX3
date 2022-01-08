import numpy as np
from src_to_implement.Layers import Base


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        self.out_sigmoid = None
        super().__init__()

    def forward(self, input_tensor):
        self.out_sigmoid = 1 / (1 + np.exp(-1*input_tensor))
        return self.out_sigmoid

    def backward(self, error_tensor):
        return self.out_sigmoid * (1 - self.out_sigmoid) * error_tensor
