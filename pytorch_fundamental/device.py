import torch
import numpy as np
CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError as e:
    CUPY_AVAILABLE = False
    print("Failed to import cupy:", e)
    
device = "cuda" if torch.cuda.is_available else "cpu"
print(device)
if device == "cuda" and CUPY_AVAILABLE:
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(num_gpus)

tensor = torch.tensor([1, 2, 3])
# tensor = tensor.to(device)
print(tensor, tensor.device)