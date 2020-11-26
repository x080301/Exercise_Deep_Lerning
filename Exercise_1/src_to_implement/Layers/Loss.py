import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        
        self.input_tensor = input_tensor

        CrossEntropy = np.multiply(np.log(input_tensor + np.finfo(float).eps), label_tensor)
               
        CrossEntropy = -np.sum(CrossEntropy)

        return CrossEntropy

    def backward(self, label_tensor):  #(self,error tensor)
        
        error_tensor_0 = np.true_divide(label_tensor, self.input_tensor) * (-1)
        
        
        return error_tensor_0
        