from src_to_implement.Layers import Base
import numpy as np
import sys


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.batch_size = 0
        self.channel_size = 0
        self.image_H = 0
        self.image_W = 0
        self.mean = 0
        self.var = 0
        self.trainable = True
        # weights
        self.gamma = np.ones((1, channels))
        # bias
        self.beta = np.zeros((1, channels))
        # Moving average decay
        self.alpha = 0.8
        self.epsilon = 1e-10
        self.first_step = True

    def forward(self, input_tensor):
        if not self.testing_phase:
            # Convolutional layer
            if np.size(input_tensor.shape) > 2:
                self.batch_size, self.channel_size, self.image_H, self.image_W = input_tensor.shape
                reformat_image_tensor = self.reformat(input_tensor)
                mean = np.mean(reformat_image_tensor, axis=0)
                mean_vector = mean.reshape(1, reformat_image_tensor.shape[1])
                var = np.var(reformat_image_tensor, axis=0)
                var_vector = var.reshape(1, reformat_image_tensor.shape[1])
                # Moving average estimation
                if self.first_step:
                    self.first_step = False
                    self.mean = mean_vector
                    self.var = var_vector
                self.mean = (1 - self.alpha) * mean_vector + self.alpha * self.mean
                self.var = (1 - self.alpha) * var_vector + self.alpha * self.var
                # -----------------------------
                normed_input = (reformat_image_tensor - mean_vector)/np.sqrt(var_vector + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                reformat_vector_tensor = self.reformat(normed_input)
                return reformat_vector_tensor
            # fully connected layer
            else:
                mean = np.mean(input_tensor, axis=0)
                mean_vector = mean.reshape(1, input_tensor.shape[1])
                var = np.var(input_tensor, axis=0)
                var_vector = var.reshape(1, input_tensor.shape[1])
                # Moving average estimation
                if self.first_step:
                    self.first_step = False
                    self.mean = mean_vector
                    self.var = var_vector
                self.mean = (1 - self.alpha) * mean_vector + self.alpha * self.mean
                self.var = (1 - self.alpha) * var_vector + self.alpha * self.var
                # -----------------------------
                normed_input = (input_tensor - mean_vector)/np.sqrt(var_vector + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                return normed_input
        # Test Phase --------------------------------
        else:
            if np.size(input_tensor.shape) > 2:
                reformat_image_tensor = self.reformat(input_tensor)
                normed_input = (reformat_image_tensor - self.mean)/np.sqrt(self.var + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                reformat_vector_tensor = self.reformat(normed_input)
                return reformat_vector_tensor
            # fully connected layer
            else:
                normed_input = (input_tensor - self.mean)/np.sqrt(self.var + self.epsilon)
                normed_input = self.gamma * normed_input + self.beta
                return normed_input

    def reformat(self, image_tensor):
        if np.size(image_tensor.shape) > 2:
            reformat_image_tensor = image_tensor.reshape((image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]*image_tensor.shape[3]))
            reformat_image_tensor = reformat_image_tensor.transpose(1, 0, 2)
            reformat_image_tensor = reformat_image_tensor.reshape(reformat_image_tensor.shape[0], reformat_image_tensor.shape[1]* reformat_image_tensor.shape[2])
            reformat_image_tensor = reformat_image_tensor.transpose()
            return reformat_image_tensor
        else:
            reformat_vector_tensor = image_tensor.transpose()
            reformat_vector_tensor = reformat_vector_tensor.reshape(reformat_vector_tensor.shape[0], self.batch_size, self.image_H * self.image_W)
            reformat_vector_tensor = reformat_vector_tensor.transpose(1, 0, 2)
            reformat_vector_tensor = reformat_vector_tensor.reshape(self.batch_size, self.channel_size, self.image_H, self.image_W)
            return reformat_vector_tensor






