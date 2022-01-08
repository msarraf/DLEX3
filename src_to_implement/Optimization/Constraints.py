import numpy as np

class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha*np.sign(weights)

    def norm(self, weights):
        calculated_norm = np.sum(np.abs(weights))
        return calculated_norm * self.alpha


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        calculated_norm = np.linalg.norm(weights)
        return calculated_norm*calculated_norm*self.alpha


