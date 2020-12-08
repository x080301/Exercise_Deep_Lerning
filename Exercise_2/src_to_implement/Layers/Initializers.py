import numpy as np

class Constant:

    def __init__(self,init_value=0.1):
        
        self.fan_in=None   
        self.fan_out = None
        self.init_value = init_value
        

    def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out

            weights = np.ones(shape) * self.init_value
            
            return weights

class UniformRandom:

    def __init__(self):
        
        self.fan_in=None   
        self.fan_out = None
    
    def initialize(self, shape, fan_in, fan_out):

        self.fan_in = fan_in
        self.fan_out = fan_out

        weights = np.random.rand(shape[0], shape[1])
        
        return weights  
        

class Xavier:

    def __init__(self):

        self.fan_in=None   
        self.fan_out = None

    def initialize(self, shape, fan_in, fan_out):

        self.fan_in = fan_in
        self.fan_out = fan_out

        sigma = np.sqrt(2 / (fan_out + fan_in))
        
        weights = np.random.randn(shape[0], shape[1]) * sigma
        
        return weights  


class He:

    def __init__(self,init_value):

        self.fan_in=None   
        self.fan_out = None

    def initialize(self, shape, fan_in, fan_out):

        self.fan_in = fan_in
        self.fan_out = fan_out

        sigma = np.sqrt(2 / fan_in)
                
        weights = np.random.randn(shape[0], shape[1]) * sigma
        
        return weights