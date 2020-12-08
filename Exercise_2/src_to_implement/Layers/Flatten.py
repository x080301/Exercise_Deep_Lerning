import numpy as np

class Flatten:

    def __init__(self):

        self.tensor_shape = None
        

    def forward(self, input_tensor):
        
        self.tensor_shape = input_tensor.shape
        
        output_tensor = np.reshape(input_tensor, -1)

        return output_tensor       

    def backward(self, error_tensor):
        
        output_error_tensor = np.reshape(error_tensor, self.tensor_shape)
        
        return output_error_tensor

