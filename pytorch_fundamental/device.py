import torch
import numpy as np
import cupy as cp
device = "cuda" if torch.cuda.is_available else "cpu"
print(device, torch.cuda.device_count())
num_gpus = cp.cuda.runtime.getDeviceCount()
print(num_gpus)

tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)

# tensor_numpy = tensor_on_gpu.numpy() #this will throw an error because numpy don't support GPU
tensor_numpy = tensor_on_gpu.cpu().numpy()
print(tensor_numpy, type(tensor_numpy))

tensor_cupy = cp.asarray(tensor_numpy)
print(tensor_cupy, type(tensor_cupy))