import torch
x = torch.arange(10, 100, 10)
print(x, x.shape)

x_reshaped = x.reshape(1, 3, 3)
print(x_reshaped, x_reshaped.shape)
print(x_reshaped[0, 0, 1])
print(x_reshaped[0, 1])
print(x_reshaped[:,1])
print(x_reshaped[:, 0])
print(x_reshaped[:, 1:3, 1])
print(x_reshaped[0, 0, :])