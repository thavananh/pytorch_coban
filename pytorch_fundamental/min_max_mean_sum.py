from numpy import argmin
import torch
x = torch.arange(0, 100, 10)
print(x)
print(x.min())
print(x.max())
print(x.type(torch.float32).mean())
print(x.sum())

#fiding positional min and max

shuffle_index = torch.randperm(x.size(0))
x = x[shuffle_index]
print(x)

print(torch.argmin(x))
print(torch.argmax(x))