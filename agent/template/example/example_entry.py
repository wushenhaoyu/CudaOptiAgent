import os
import hashlib
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# -----------------------------
# Load CUDA extension
# -----------------------------
kernel_path = r"D:\data\design\code\CudaOptiAgent-clear\run\qwen_qwen3.5-plus-2026-02-15\level3\1_MLP\spec\kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, "rb") as f:
    content_hash = hashlib.md5(f.read()).hexdigest()[:8]

cuda_extension = load(
    name=f"MLP_{content_hash}",
    sources=[kernel_path],
    verbose=False,
)

# -----------------------------
# Model Definition
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.layer_sizes = layer_sizes

        layers = []
        current_input = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input, layer_size))
            layers.append(nn.ReLU())
            current_input = layer_size
        layers.append(nn.Linear(current_input, output_size))
        self.network = nn.ModuleList(layers)

    def forward(self, x):
        out = x

        # -------- Direct kernel calls --------
        # Linear0 + ReLU fused kernel
        out = cuda_extension.Linear_GEMM_with_ReLU_epilogue(
            out,
            self.network[0].weight,
            self.network[0].bias
        )

        # Linear1 + ReLU fused kernel
        out = cuda_extension.Linear_GEMM_with_ReLU_epilogue(
            out,
            self.network[2].weight,
            self.network[2].bias
        )

        # Linear2 standalone kernel
        out = cuda_extension.Linear_GEMM_standalone(
            out,
            self.network[4].weight,
            self.network[4].bias
        )

        return out

# -----------------------------
# Test / Inputs
# -----------------------------
batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]