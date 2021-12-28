from src_to_implement.Layers import Base
import numpy as np
import sys
class BatchNormalization(Base.BaseLayer):

    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.trainable = True

        self.gamma = 1
        self.beta = 0
        self.it_index = 0








    def forward(self,input_tensor):

        self.input_tensor = input_tensor
        epsilon = sys.float_info.epsilon
        if self.testing_phase == False:
            self.mean_input_tensor = np.mean(self.input_tensor,axis = 0)
            self.var_input_tensor = np.var(self.input_tensor,axis =0)
            if self.it_index == 0:
                self.mean_mov = self.mean_input_tensor
                self.var_mov = self.var_input_tensor
            alpha = 0.8
            self.mean_mov = alpha*self.mean_mov + (1-alpha)*self.mean_input_tensor
            self.var_mov = alpha*self.var_mov + (1-alpha)*self.var_input_tensor
            self.normalized_input_tensor = np.divide((self.input_tensor - self.mean_input_tensor), np.sqrt(self.var_input_tensor + epsilon))
            self.output = np.multiply(self.gamma,self.normalized_input_tensor) + self.beta
            return self.output

        else:

            #self.normalized_input_tensor = np.divide((self.input_tensor - self.mean_mov),np.sqrt(self.var_mov + epsilon))
            self.normalized_input_tensor = np.divide((self.input_tensor - self.mean_mov),
                                                     np.sqrt(self.var_mov))
            self.output = np.multiply(self.gamma, self.normalized_input_tensor) + self.beta

            return self.output









