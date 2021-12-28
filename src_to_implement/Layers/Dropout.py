from src_to_implement.Layers import Base
import numpy as np
class Dropout(Base.BaseLayer):

    def __init__(self,probability):
        super().__init__()
        self.probability = probability
        self.trainable = False


    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        if self.testing_phase == False:
            self.zero_mapping = np.random.rand(self.input_tensor.shape[0], self.input_tensor.shape[1]) < self.probability
            remained_input_tensor = np.multiply(self.input_tensor, self.zero_mapping)
            remained_input_tensor /= self.probability  # this line is called inverted dropout technique
        else:
            remained_input_tensor =  self.input_tensor
        return(remained_input_tensor)
    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        output = np.multiply(self.zero_mapping ,self.error_tensor)
        output /= self.probability
        return(output)



