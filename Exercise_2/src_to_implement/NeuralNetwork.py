import sys
import os

sys.path.append(os.path.split(os.path.realpath(__file__))[0])#添加路径，这个是临时的
import copy
import numpy as np
from Layers import *

class NeuralNetwork:
    
    
    def __init__(self, optimizer,weights_initializer,bias_initializer):

        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = False   
        self.loss_layer = False
        
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
                
    def forward(self):
        
        self.input_tensor,self.label_tensor = self.data_layer.next()

        loop_IO = self.input_tensor
        
        for i in range(len(self.layers)):
            
            loop_IO = self.layers[i].forward(loop_IO)

        output_tensor_Loss = self.loss_layer.forward(loop_IO, self.label_tensor)
        self.loss.append(output_tensor_Loss)
                
        return output_tensor_Loss
    
    def backward(self):
        
        loop_IO = self.loss_layer.backward(self.label_tensor)
        
        for i in range(len(self.layers) - 1, -1, -1):
            
            loop_IO = self.layers[i].backward(loop_IO)    

        return loop_IO
    
    def append_trainable_layer(self, layer):
                
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(copy.deepcopy(layer))



    def train(self, iterations):
        
        for i in range(iterations):

            loss = self.forward()
            self.loss.append(loss)

            self.backward()
    
    def test(self,input_test_data):

        loop_IO = input_test_data
        
        for i in range(len(self.layers)):
            
            loop_IO = self.layers[i].forward(loop_IO)

        return loop_IO
        

        
        
        

        