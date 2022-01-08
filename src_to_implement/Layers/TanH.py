
import numpy as np
class TanH:
    def __init__(self):
        pass

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        self.out_TanH = np.tanh(self.input_tensor)
        return self.out_TanH

    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        return (1 - self.out_TanH**2) * self.error_tensor