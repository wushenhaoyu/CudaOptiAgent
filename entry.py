
import torch
import torch.nn as nn
import hashlib
import os
from torch.utils.cpp_extension import load

# Load CUDA extension with a name derived from the content hash to force recompilation on changes
kernel_path = "kernel.cu"
if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, "rb") as f:
    file_bytes = f.read()
content_hash = hashlib.md5(file_bytes).hexdigest()[:8]

cuda_extension = load(
    name=f"ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU_{content_hash}",
    sources=[kernel_path],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization,
    average pooling, and GELU activation via a fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Keep modules/parameters for state, shape, and training compatibility; fused CUDA op consumes them.
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        
        return cuda_extension.ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.kernel_size,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
            self.conv_transpose.groups,
            self.sum_weight,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
            self.avg_pool.kernel_size,
            self.avg_pool.stride
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