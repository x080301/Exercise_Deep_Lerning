import numpy as np

class SoftMax:

    def __init__(self):
        pass

    def forward(self,input_tensor):
        
        #Numeric
        input_tensor = input_tensor - input_tensor.max()
        
        exp_input_tensor = np.exp(input_tensor)

        sum_exp_input_tensor = np.sum(exp_input_tensor, axis=1)        
        sum_exp_input_tensor = np.transpose(np.expand_dims(sum_exp_input_tensor, 0).repeat(input_tensor.shape[1], axis=0))

        self.estimated_class_probabilities = np.true_divide(exp_input_tensor, sum_exp_input_tensor)
        
        return self.estimated_class_probabilities


    def backward(self, error_tensor_1):
        
        Ey = np.multiply(error_tensor_1, self.estimated_class_probabilities)
        Ey = np.sum(Ey, axis=1)
        Ey = np.transpose(np.expand_dims(Ey, 0).repeat(error_tensor_1.shape[1], axis=0))

        error_tensor_0 = error_tensor_1 - Ey
        error_tensor_0 = np.multiply(error_tensor_0, self.estimated_class_probabilities)
        
        return error_tensor_0
        