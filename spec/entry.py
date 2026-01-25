import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

# Compute hash of kernel.cu file
kernel_path = r"kernel.cu"
with open(kernel_path, 'rb') as f:
    file_hash = hashlib.md5(f.read()).hexdigest()

# Load CUDA extension with hash-based name
extension_name = f"square_matrix_multiplication_{file_hash}"
cuda_extension = load(
    name=extension_name,
    sources=[kernel_path],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    using custom CUDA extension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using CUDA extension.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return cuda_extension.Square_matrix_multiplication_(A, B)

N = 16

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed