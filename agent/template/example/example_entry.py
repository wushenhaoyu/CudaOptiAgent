import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = r"D:\data\design\code\CudaOptiAgent-clear\run\qwen_qwen3.5-plus-2026-02-15\level3\1_MLP\spec\kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, 'rb') as f:
    file_content = f.read()
    content_hash = hashlib.md5(file_content).hexdigest()[:8]

cuda_extension = load(
    name=f"MLP_{content_hash}",
    sources=[kernel_path],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):

        params = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                params.append(module.weight)
                params.append(module.bias)
        return cuda_extension.MLP(x, *params)

batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]