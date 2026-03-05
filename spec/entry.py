import os
import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

kernel_path = r"spec/kernel/kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, "rb") as f:
    content_hash = hashlib.md5(f.read()).hexdigest()[:8]

cuda_extension = load(
    name=f"convtranspose3d_sum_layernorm_avgpool_gelu_{content_hash}",
    sources=[kernel_path],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = cuda_extension.conv_transpose3d_add(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.sum_weight,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
            self.conv_transpose.groups,
            self.conv_transpose.dilation,
        )
        x = cuda_extension.layernorm_channel_reduction(
            x,
            self.norm.weight,
            self.norm.bias,
            self.norm.normalized_shape,
            self.norm.eps,
        )
        x = cuda_extension.avgpool3d_gelu(
            x,
            self.avg_pool.kernel_size,
            self.avg_pool.stride,
            self.avg_pool.padding,
            self.avg_pool.ceil_mode,
            self.avg_pool.count_include_pad,
            self.avg_pool.divisor_override,
            self.gelu.approximate,
        )
        return x

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


if __name__ == "__main__":
    device = torch.device('cuda:2')
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    inputs = [x.to(device) for x in inputs]
    modelnew = ModelNew(*init_inputs).to(device).eval()
    with torch.inference_mode():
        out_entry = modelnew(*inputs)
        print("out_entry\n")