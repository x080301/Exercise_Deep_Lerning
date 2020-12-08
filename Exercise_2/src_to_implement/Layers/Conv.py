import numpy as np
import math

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        # For 1D, it has the shape [c, m], 
        # whereas for 2D, it has the shape [c, m, n], 
        # where c represents the number of input channels, 
        # and m, n represent the spatial extent of the filter kernel.

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
        self.kernels = list()
        
        for i in range(num_kernels):

            if len(convolution_shape) == 2:  # 1D
                
                self.kernels.append(np.random.rand(convolution_shape[0], convolution_shape[1]))
            
            else:  # 2D
                
                self.kernels.append(np.random.rand(convolution_shape[0], convolution_shape[1], convolution_shape[2]))

    def forward(self, input_tensor):

        # The input layout for 1D is defined in b, c, y 
        # order, for 2D in b, c, y, x order. 
        # Here, b stands for the batch, c represents the channels and x, y represent the spatial dimensions.


        # caculates output_tensor shape and generates an empty output_tensor
        if len(input_tensor.shape) == 3:  # 1D
            
            output_tensor_y = math.ceil(input_tensor.shape[2] / self.stride_shape)

            output_tensor = np.empty((input_tensor.shape[0], self.num_kernels, output_tensor_y), dtype=np.float64)
        
        else:  # 2D
            
            output_tensor_y = math.ceil(input_tensor.shape[2] / self.stride_shape[0])
            output_tensor_x = math.ceil(input_tensor[3] / self.stride_shape[1])
                        
            output_tensor = np.empty((input_tensor.shape[0], self.num_kernels, output_tensor_y, output_tensor_x), dtype=np.float64)


        for b in range(input_tensor.shape[0]):

            single_data_tensor = input_tensor[b]


            # generate Tuple for padding
            padding_size = [((pad - 1) // 2, pad - 1 - (pad - 1) // 2) for pad in self.convolution_shape]                
            padding_size[0] = [(0, 0)]
            padding_size = tuple(padding_size)
        
            # padding
            single_data_tensor = np.lib.pad(single_data_tensor, padding_size, 'constant', constant_values=0)


            # cross correlation
            if len(input_tensor.shape) == 3: # 1D

                for index_kernels in range(self.num_kernels):

                    for y in range(0, input_tensor.shape[2], self.stride_shape):

                        y_start = y
                        y_end = y + self.convolution_shape[1]

                        output_tensor[b,index_kernels, y] = np.sum(
                            np.multiply(
                                single_data_tensor[: , y_start: y_end],                                
                                self.kernels[index_kernels]                                
                            )
                        )
                
            else: # 2D
                


                for index_kernels in range(self.num_kernels):

                    for y in range(0, input_tensor.shape[2], self.stride_shape[0]):
                        
                        for x in range(0, input_tensor.shape[3], self.stride_shape[1]):
                            
                            y_start = y
                            y_end = y + self.convolution_shape[1]
                            x_start = x
                            x_end = x + self.convolution_shape[2]
                            
                            output_tensor[b,index_kernels, y, x] = np.sum(
                                np.multiply(
                                    single_data_tensor[: , y_start: y_end, x_start: x_end],
                                    self.kernels[index_kernels]
                                )
                            )                      
                        
        return output_tensor             




    def backward(self, error_tensor):

        pass