import sys
import os
import copy
sys.path.append(os.path.split(os.path.realpath(__file__))[0])#添加路径，这个是临时的

import numpy as np
from Layers import *

class NeuralNetwork:
    
    
    loss = list()
    layers = list()
    data_layer = False   

    loss_layer = False
    
    def __init__(self, optimizer):

        self.optimizer = optimizer
        
    def forward(self):
        
        self.input_tensor = self.data_layer.next()[0]
        self.label_tensor = self.data_layer.next()[1]

        input_size = self.input_tensor.shape[0]
        output_size = self.label_tensor.shape[0]

        self.layers.append(FullyConnected.FullyConnected(input_size, output_size))
        self.layers[0].optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(ReLU.ReLU())
        self.layers.append(SoftMax.SoftMax())
        self.layers.append(Loss.CrossEntropyLoss())

        output_tensor_FullyConnected = self.layers[0].forward(self.input_tensor)
        output_tensor_ReLU = self.layers[1].forward(output_tensor_FullyConnected)
        output_tensor_SoftMax = self.layers[2].forward(output_tensor_ReLU)
        output_tensor_Loss = self.layers[3].forward(output_tensor_SoftMax, self.label_tensor)

        return output_tensor_Loss
    
    def backward(self):

        error_tensor_0_Loss = self.layers[3].backward(self.label_tensor)
        error_tensor_0_SoftMax = self.layers[2].backward(error_tensor_0_Loss)
        error_tensor_0_ReLU = self.layers[1].backward(error_tensor_0_SoftMax)
        error_tensor_0_FullyConnected = self.layers[0].backward(error_tensor_0_ReLU)

        return error_tensor_0_FullyConnected
    
    def append_trainable_layer(self, layer):
                
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        
        for i in range(iterations):

            loss = self.forward()
            self.loss.append(loss)

            self.backward()
    
    def test(self,input_tensor):

        output_tensor_FullyConnected = self.layers[0].forward(input_tensor)
        output_tensor_ReLU = self.layers[1].forward(output_tensor_FullyConnected)
        output_tensor_SoftMax = self.layers[2].forward(output_tensor_ReLU)

        return output_tensor_SoftMax
        

        
        
        

        