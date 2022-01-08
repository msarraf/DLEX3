import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        self.out_sigmoid = 1/(1 + np.exp(-self.input_tensor))
        return self.out_sigmoid

    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        return self.out_sigmoid * (1 - self.out_sigmoid) * self.error_tensor