import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(tensor, tensor.dtype)
print(array, tensor.dtype)

array = array + 1
print(array)

tensor = tensor + 1
print(tensor)

#Default numpy datatype is float64
#Default tensor datatype is float32
#When convert torch tensor to numpy array, it won't share the same memory location, and the data type will be the same as the tensor.