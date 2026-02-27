
import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = "kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, "rb") as f:
    file_bytes = f.read()
content_hash = hashlib.md5(file_bytes).hexdigest()[:8]

cuda_extension = load(
    name=f"ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU_{content_hash}",
    sources=[kernel_path],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        
        # Create linear layers for parameter storage
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.linear_layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Ensure all weights and biases are on the same device as input
        device = x.device
        weights = []
        biases = []
        
        for layer in self.linear_layers:
            weights.append(layer.weight.to(device))
            biases.append(layer.bias.to(device))
        
        # Call the CUDA kernel
        return cuda_extension.MLP(x, weights, biases)

# Test code
batch_size = 4
input_size = 32
layer_sizes = [32, 32]
output_size = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]