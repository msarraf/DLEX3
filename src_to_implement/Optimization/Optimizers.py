import numpy as np
import sys


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.step_update = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        momentum_term = np.multiply(self.momentum_rate, self.step_update)
        gradient_term = np.multiply(self.learning_rate, gradient_tensor)
        self.step_update = np.subtract(momentum_term, gradient_term)
        new_weight = np.add(self.step_update, weight_tensor)
        return new_weight


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.k = self.k + 1
        # update v_k
        v_pass_term = np.multiply(self.mu, self.v_k)
        v_gradient_term = np.multiply((1-self.mu), gradient_tensor)
        self.v_k = np.add(v_pass_term, v_gradient_term)

        # update r_k
        r_pass_term = np.multiply(self.rho, self.r_k)
        r_gradient_term = np.multiply((1 - self.rho), gradient_tensor**2)
        self.r_k = np.add(r_pass_term, r_gradient_term)

        # bias correction
        updated_v_k = np.divide(self.v_k, (1-self.mu**self.k))
        updated_r_k = np.divide(self.r_k, (1-self.rho**self.k))
        epsilon = np.finfo(float).eps
        new_weight_update = self.learning_rate * (np.divide(updated_v_k, (np.add(np.sqrt(updated_r_k), epsilon))))
        new_weights = np.subtract(weight_tensor, new_weight_update)

        return new_weights


class Sgd:
    def __init__(self,learning_rate):
        # if not learning_rate.is_float():
        if not isinstance(learning_rate, float):
            if not isinstance(learning_rate, int):
                raise TypeError("learning rate must be set to an float or int")
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        updated_weights = np.subtract(weight_tensor, (np.multiply(self.learning_rate, gradient_tensor)))
        return updated_weights




