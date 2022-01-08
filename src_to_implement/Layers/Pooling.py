import numpy as np
from src_to_implement.Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_position = []
        self.input_tensor = 0
        self.error_tensor = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        pooling_out_H = int((input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0] + 1)
        pooling_out_W = int((input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1] + 1)
        pooling_out = np.zeros((input_tensor.shape[0], input_tensor.shape[1], pooling_out_H, pooling_out_W))
        for b in range(input_tensor.shape[0]):
            for k in range(input_tensor.shape[1]):
                for x in range(pooling_out_H):
                    for y in range(pooling_out_W):
                        pool = self.input_tensor[b, k, x * self.stride_shape[0]:x * self.stride_shape[0]
                                           + self.pooling_shape[0], y * self.stride_shape[1]:y * self.stride_shape[1]
                                           + self.pooling_shape[1]]
                        max_in_pool = np.max(pool)
                        pooling_out[b, k, x, y] = max_in_pool

        return pooling_out

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        bp_error_tensor = np.zeros_like(self.input_tensor)
        for b in range(error_tensor.shape[0]):
            for k in range(error_tensor.shape[1]):
                for x in range(error_tensor.shape[2]):
                    for y in range(error_tensor.shape[3]):
                        pool = self.input_tensor[b, k, x * self.stride_shape[0]:x * self.stride_shape[0]
                                           + self.pooling_shape[0], y * self.stride_shape[1]:y * self.stride_shape[1]
                                           + self.pooling_shape[1]]
                        max_location = (pool == np.max(pool)).astype(int)
                        bp_error_tensor[b, k, x * self.stride_shape[0]:x * self.stride_shape[0] + self.pooling_shape[0],
                                            y * self.stride_shape[1]:y * self.stride_shape[1] + self.pooling_shape[1]] \
                                            += max_location * self.error_tensor[b, k, x, y]
        return bp_error_tensor
