import numpy as np

error_tensor_1 = np.random.rand(3, 2) - 0.5
estimated_class_probabilities = np.random.rand(3, 2) - 0.5

print(error_tensor_1)
Ey = np.multiply(error_tensor_1, estimated_class_probabilities)
Ey = np.sum(Ey, axis=1)
Ey = np.transpose(np.expand_dims(Ey, 0).repeat(error_tensor_1.shape[1], axis=0))

print(Ey)