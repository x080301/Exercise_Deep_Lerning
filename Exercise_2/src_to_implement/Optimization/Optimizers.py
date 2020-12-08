import numpy as np

class Sgd:
    
    def __init__(self, learning_rate):

        self.learning_rate = learning_rate
    
    def calculate_update(self,weight_tensor, gradient_tensor):
        
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        
        return updated_weights

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0
    
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        
        updated_weights = weight_tensor + self.v
        
        return updated_weights
        
class Adam:

    def __init__(self, learning_rate, mu, rho):
                        
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 0    

    def calculate_update(self,weight_tensor, gradient_tensor):

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor

        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
        
        v_hat = np.true_divide(self.v, (1 - np.power(self.mu, self.k)))
        r_hat = np.true_divide(self.r, (1 - np.power(self.rho, self.k)))
        self.k = self.k + 1
        
        updated_weights = weight_tensor - self.learning_rate * np.true_divide(np.sqrt(v_hat, r_hat) + np.finfo(float).eps)
        
        return updated_weights

        