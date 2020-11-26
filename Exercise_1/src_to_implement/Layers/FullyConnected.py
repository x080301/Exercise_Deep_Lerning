import numpy as np
#from .. import Optimization


class FullyConnected:

    optimizer = False
    biases = False

    def __init__(self,input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.rand(input_size+1,output_size)        
        
        #self.run_optimizer = Optimization.Optimizers.Sgd(0.001)

    def setter_optimizer(self, set_optimizer):

        self.optimizer = set_optimizer

    def getter_optimizer(self):

        return self.optimizer 

    def set_biases(self, biases):
        
        self.biases = biases
        

    def forward(self,input_tensor):

        self.input_tensor = input_tensor
        
        input_add_one_column = np.ones((self.input_tensor.shape[0],1))
        self.input_tensor = np.concatenate((self.input_tensor, input_add_one_column), axis=1)  
        
        #print(self.input_tensor.shape, "_______", self.weights.shape)
        #print(self.input_size,"_____________",self.output_size)
        #import  os
        #os.system('pause')
        


        output_tensor = np.matmul(self.input_tensor, self.weights)
        

        return output_tensor

       

    def backward(self,error_tensor_1_prime):


        weights_T = np.transpose(self.weights)

        error_tensor_0 = np.matmul(error_tensor_1_prime, weights_T)
        error_tensor_0 = np.delete(error_tensor_0, -1, axis=1)
       
        #caculate gradient_weights
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor_1_prime)
        
        #update weights
        if self.optimizer:        
            
            #self.run_optimizer = self.optimizer
            
            #self.weights = self.run_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)


           


        return error_tensor_0
    

        
