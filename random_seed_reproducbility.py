import random
import torch

random_tensor_A = torch.rand(2, 3)
random_tensor_B = torch.rand(2, 3)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)


random_seed = 42
torch.manual_seed(random_seed)
random_tensor_C = torch.rand(2, 3)
print(random_tensor_C)

torch.manual_seed(random_seed)
random_tensor_D = torch.rand(2, 3)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)