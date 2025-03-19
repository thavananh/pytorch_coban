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

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random_tensor_A = torch.rand(7, 7)
print(random_tensor_A)
torch.manual_seed(RANDOM_SEED)
random_tensor_B = torch.rand(1, 7)
print(random_tensor_B)
random_tensor_C = torch.matmul(random_tensor_A, random_tensor_B.T)
print(random_tensor_C)

torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
random_tensor_D = torch.rand(2,3).to(device)
random_tensor_E = torch.rand(2,3).to(device)
random_tensor_F = torch.matmul(random_tensor_D, random_tensor_E.T)
print(random_tensor_F)
print(random_tensor_F.min())
print(random_tensor_F.max())

torch.cuda.manual_seed(7)
torch.manual_seed(7)
random_tensor_H = torch.rand(1, 1, 1, 10).to(device)
random_tensor_G = torch.rand(10,).to(device)

print(random_tensor_H, random_tensor_H.shape)
print(random_tensor_G, random_tensor_G.shape)