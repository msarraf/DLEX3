import numpy as np


class Flatten:

    def __init__(self):
        self.trainable = False
        self.batch_size = 0
        self.feature_map = 0
        self.feature_map_dim = 0
        self.flatten_output = 0

    def forward(self,input_tensor):
       self.input_tensor = input_tensor
       self.batch_size = self.input_tensor.shape[0]
       self.feature_map = self.input_tensor.shape[1]
       self.feature_map_dim = [self.input_tensor.shape[2], self.input_tensor.shape[3]]
       tensor_elements = self.feature_map * self.feature_map_dim[0] * self.feature_map_dim[1]
       self.flatten_output = np.reshape(np.ravel(self.input_tensor), (self.input_tensor.shape[0], tensor_elements))
       return self.flatten_output

    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        reshaped_flatten = np.reshape(self.error_tensor, (self.batch_size, self.feature_map, self.feature_map_dim[0],self.feature_map_dim[1]))
        return reshaped_flatten




