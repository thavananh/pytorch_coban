import torch
from tqdm import tqdm

tensor = torch.tensor([1, 2, 3])
print(tensor)
print(tensor * tensor)
print(torch.matmul(tensor, tensor))

value = 0
for i in tqdm(range(len(tensor))):
    value += tensor[i] * tensor[i] #this method always slower than torch.matmul a lot
print(value)

print(torch.matmul(tensor, tensor))
print(torch.matmul(torch.rand(3, 4), torch.rand(4, 3))) # will give a tensor with 3x3 dimession
# print(torch.matmul(torch.rand(2, 3), torch.rand(2, 9))) #Will throw an error because inner dimession not match

tensor_A = torch.tensor([[1, 2],[3, 4],[4, 5]])
tensor_B = torch.tensor([[1, 2],[3, 4],[4, 5]])
print(tensor_A.shape)
print(tensor_B.shape)
print(tensor_B.T)
print(tensor_B.T.shape)
print(torch.matmul(tensor_A, tensor_B.T))