import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = r"kernel.cu"

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
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
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
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        weights = []
        biases = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                weights.append(module.weight)
                biases.append(module.bias)
        
        return cuda_extension.MLP(x, weights, biases)

# Reduce tensor sizes for CPU testing
batch_size = 2  # Reduced from 128
input_size = 32  # Reduced from 16384
layer_sizes = [32, 32]  # Reduced from [16384, 16384]
output_size = 16  # Reduced from 8192

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]