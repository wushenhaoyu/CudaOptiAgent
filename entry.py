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
    name=f"ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU_{content_hash}",
    sources=[kernel_path],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Extract parameters from conv_transpose
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        # sum_weight is a parameter tensor
        sum_weight = self.sum_weight
        # LayerNorm parameters
        norm_weight = self.norm.weight
        norm_bias = self.norm.bias
        # Pool kernel size
        pool_kernel_size = self.avg_pool.kernel_size

        # Call the CUDA extension function
        # The CUDA function signature is assumed to accept:
        # input x,
        # conv weight, conv bias,
        # sum_weight,
        # layernorm weight and bias,
        # avgpool kernel size,
        # and possibly stride, padding, output_padding for conv transpose
        # Since the original forward does conv_transpose, then sum, then norm, then avgpool, then gelu,
        # we pass all needed params to CUDA function to replicate this.

        # We pass stride, padding, output_padding as tuples
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        output_padding = self.conv_transpose.output_padding

        return cuda_extension.ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU(
            x,
            weight,
            bias,
            sum_weight,
            norm_weight,
            norm_bias,
            pool_kernel_size,
            stride,
            padding,
            output_padding
        )

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]