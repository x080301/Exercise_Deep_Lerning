import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        
        self.input_tensor = input_tensor

        CrossEntropy = -np.log(np.multiply(input_tensor, label_tensor)+np.finfo(float).eps)
        
        CrossEntropy = np.sum(Ey, axis=0)

        return CrossEntropy

    def backward(self, label_tensor):  #(self,error tensor)
        
        error_tensor_0 = -np.true_divide(label_tensor, self.input_tensor)
        
        return error_tensor_0
        