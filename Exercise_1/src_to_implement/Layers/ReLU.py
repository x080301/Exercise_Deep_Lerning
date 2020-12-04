class ReLU:

    

    def __init__(self):
        pass

    def forward(self,input_tensor):
        
        self.input_tensor = input_tensor
        
        input_tensor_next_layer = input_tensor * (input_tensor > 0)
        
        return input_tensor_next_layer
        
    def backward(self,error_tensor_1):
        
        error_tensor_0 = error_tensor_1 * (self.input_tensor > 0)

        return error_tensor_0
        
        