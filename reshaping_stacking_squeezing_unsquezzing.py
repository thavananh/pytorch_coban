from numpy import arange, int16
import torch


# Reshaping - Reshape a input tensor into a defined tensor
# View - Return a view of an input tensor of certain shape but the same memory as original tensor
# Stacking - comebine multiple tensor on top of each other (vstack) or side by side (hstack)
# Squeeze - Removes all `1` dimensions from a tensor
# Unsqueeze - Add a `1` dimensions to a target tensor
# Permute - Return a view of the input with dimensions permuted in a certain way

x = torch.arange(1., 10.)
print(x, x.shape)

x_reshaped = x.reshape(1, 9)
print(x, x_reshaped.shape)
print(x.reshape(3, 3), x.reshape(3, 3).shape)

x_1 = torch.arange(1.0, 13.0)
x_1_reshaped_1 = x_1.reshape(1, 12)
x_1_reshaped_2 = x_1.reshape(12)
x_1_reshaped_3 = x_1.reshape(3, 4)
x_1_reshaped_4 = x_1.reshape(2, 6)

print(x_1, x_1.shape)
print(x_1_reshaped_1, x_1_reshaped_1.shape)
print(x_1_reshaped_2, x_1_reshaped_2.shape)
print(x_1_reshaped_3, x_1_reshaped_3.shape)
print(x_1_reshaped_4, x_1_reshaped_4.shape)

x_2 = torch.arange(1.0, 13)
z = x_2.view(3, 4)
print(x_2)
print(z)

z[:, 2] = 5
print(x_2)
print(z)

print('\n')
print(x, x.shape)
print('\n')
# stack tensor on top each other
x_stacked_vertical = torch.vstack([x, x, x]) #vertical stack
print(x_stacked_vertical, x_stacked_vertical.shape)
x_stack_horizontal = torch.hstack([x, x, x]) #horizontal stack
print('\n')
print(x_stack_horizontal, x_stack_horizontal.shape)

print('\n')
x_3 = torch.randint(size=(2, 3, 4), low=0, high=100, dtype=torch.int32)
print(x_3)
print(x_3.reshape(2, 12))
print(x_3.reshape(6, 2, 2))
print(x_3.min())
print(x_3.max())
print(x_3.type(torch.float32).mean())
x_3_stacked = torch.stack([x_3, x_3], dim=1)

print('\n\n\n\n-----------------------------------------')
print(x_3)
print("x3_original_shape: ", x_3.shape)
print(x_3_stacked, x_3_stacked.shape)