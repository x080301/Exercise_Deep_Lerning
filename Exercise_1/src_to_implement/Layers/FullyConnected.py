import numpy as np
#from .. import Optimization


class FullyConnected:

    optimizer = False
    biases = False

    def __init__(self,input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.rand(output_size,input_size)
        
        #self.run_optimizer = Optimization.Optimizers.Sgd(0.001)

    def setter_optimizer(self, set_optimizer):

        self.optimizer = set_optimizer

    def getter_optimizer(self):

        return self.optimizer 

    def set_biases(self, biases):
        
        self.biases = biases
        

    def forward(self,input_tensor_prime):

        self.input_tensor = input_tensor_prime
        output_tensor = np.matmul(self.input_tensor, self.weights)
                        
        return output_tensor

       

    def backward(self,error_tensor_1_prime):
        
        error_tensor_1 = error_tensor_1_prime
        
        weights_T = np.transpose(self.weights)

        error_tensor_0 = np.matmul(error_tensor_1, weights_T)
        
        #caculate gradient_weights
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor_1)
        
        #update weights
        assert (self.optimizer), "the optimizer is not set."
        
        #self.run_optimizer = self.optimizer
        
        #self.weights = self.run_optimizer.calculate_update(self.weights, self.gradient_weights)
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        if self.biases:
            self.weights = self.weights + self.biases
           


        return error_tensor_0
    

        
