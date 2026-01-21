import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_ext

# Load the CUDA extension
cuda_extension = cpp_ext.load(
    name='Square_matrix_multiplication_',
    sources=[r'D:\data\design\code\CudaOptiAgent\run\deepseek_deepseek-reasoner\level1\1_Square_matrix_multiplication_\spec\kernel.cu'],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    using custom CUDA extension.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return cuda_extension.Square_matrix_multiplication_(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

if __name__ == '__main__':
    model = ModelNew()
    inputs = get_inputs()
    init_inputs = get_init_inputs()
    model(*init_inputs)
    model(*inputs)
    print(model(*inputs))