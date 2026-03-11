
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os, hashlib

kernel_path = "kernel/kernel.cu"
if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel not found: {kernel_path}")

with open(kernel_path, "rb") as f:
    content_hash = hashlib.md5(f.read()).hexdigest()[:8]

cuda_extension = load(
    name=f"debug_{content_hash}",
    sources=[kernel_path],
    verbose=False,
    extra_cuda_cflags=["-O3","-lineinfo","-G"]
)


batch_size = 32 
num_classes = 20

def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32)]

class ModelDebug(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(ModelDebug, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        self._cache = {}
        self.report = []

    def _compare_once(self, kernel_name, cuda_out, torch_out):
        if kernel_name in self._cache:
            return

        diff = (cuda_out.detach() - torch_out.detach()).abs().max().item()
        status = "ok" if diff < 5e-3 else "mismatch"

        self.report.append({
            "kernel": kernel_name,
            "status": status,
            "max_diff": diff
        })

        self._cache[kernel_name] = True

        if status == "mismatch":
            raise RuntimeError(f"FIRST_KERNEL_MISMATCH: {kernel_name}")

    def forward(self, x):
        torch_out = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        cuda_out = cuda_extension.conv_relu_maxpool(x, self.conv1.weight, self.conv1.bias, 2, 2)
        self._compare_once("conv_relu_maxpool", cuda_out, torch_out)
        x = torch_out

        torch_out = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        cuda_out = cuda_extension.conv_relu_maxpool(x, self.conv2.weight, self.conv2.bias, 2, 2)
        self._compare_once("conv_relu_maxpool", cuda_out, torch_out)
        x = torch_out

        x = x.view(-1, 16*5*5)

        torch_out = F.relu(self.fc1(x))
        cuda_out = cuda_extension.linear_gemm_relu(x, self.fc1.weight, self.fc1.bias)
        self._compare_once("linear_gemm_relu", cuda_out, torch_out)
        x = torch_out

        torch_out = F.relu(self.fc2(x))
        cuda_out = cuda_extension.linear_gemm_relu(x, self.fc2.weight, self.fc2.bias)
        self._compare_once("linear_gemm_relu", cuda_out, torch_out)
        x = torch_out

        torch_out = self.fc3(x)
        cuda_out = cuda_extension.linear_gemm(x, self.fc3.weight, self.fc3.bias)
        self._compare_once("linear_gemm", cuda_out, torch_out)
        x = torch_out

        return x
