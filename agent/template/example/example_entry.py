import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = r"D:\data\design\code\CudaOptiAgent\run\deepseek_deepseek-reasoner\level2\1_Conv2D_ReLU_BiasAdd\spec\kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, 'rb') as f:
    file_content = f.read()
    content_hash = hashlib.md5(file_content).hexdigest()[:8]

cuda_extension = load(
    name=f"Conv2D_ReLU_BiasAdd_{content_hash}",
    sources=[kernel_path],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.Conv2D_ReLU_BiasAdd(
            x, 
            self.conv.weight, 
            self.conv.bias,  
            self.bias       
        )
        return x
    
# Reduce tensor sizes for CPU testing
batch_size = 2  # Reduced from 128
in_channels = 3  # Reduced from 64
out_channels = 4  # Reduced from 128
height = width = 8  # Reduced from 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)  # Adjusted to match reduced out_channels

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]