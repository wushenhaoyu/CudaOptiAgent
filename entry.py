import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = "kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, 'rb') as f:
    file_content = f.read()
    content_hash = hashlib.md5(file_content).hexdigest()[:8]

cuda_extension = load(
    name=f"DeepNarrowMLP_{content_hash}",
    sources=[kernel_path],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        params = []
        # Collect weights and biases from all Linear layers in order
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Ensure weights and biases are contiguous and on the same device as x
                weight = module.weight.detach()
                bias = module.bias.detach()
                params.append(weight)
                params.append(bias)
        # The CUDA kernel expects: input + 16 weights + 16 biases = 33 tensors
        # So params length must be 32 (16 weights + 16 biases)
        # Pass input tensor first, then all params unpacked
        return cuda_extension.DeepNarrowMLP(x, *params)

# Test code
batch_size = 1024
input_size = 8192
hidden_layer_sizes = [1024] * 16  # deep network with wider layers
output_size = 8192

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]