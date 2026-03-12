import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os, hashlib
import math

kernel_path = "/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level3/50_ReLUSelfAttention/spec/kernel/kernel.cu"
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


batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]

class ModelDebug(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super(ModelDebug, self).__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
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
        B, T, C = x.size()

        torch_qkv = self.c_attn(x)
        cuda_qkv = cuda_extension.c_attn_linear_gemm(x, self.c_attn.weight, self.c_attn.bias)
        self._compare_once("c_attn_linear_gemm", cuda_qkv, torch_qkv)

        torch_q, torch_k, torch_v = torch_qkv.split(self.n_embd, dim=2)
        torch_k = torch_k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        torch_q = torch_q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        torch_v = torch_v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        cuda_q, cuda_k, cuda_v = cuda_extension.slice_reshape_transpose_template(cuda_qkv, self.n_embd, self.n_head, B, T, C)
        cuda_cat = torch.cat([cuda_q, cuda_k, cuda_v], dim=0)
        torch_cat = torch.cat([torch_q, torch_k, torch_v], dim=0)
        self._compare_once("slice_reshape_transpose_template", cuda_cat, torch_cat)

        torch_att = (torch_q @ torch_k.transpose(-2, -1)) * (1.0 / math.sqrt(torch_k.size(-1)))
        torch_att = torch_att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        torch_att = F.relu(torch_att)

        cuda_att = cuda_extension.attn_gemm_with_mask_relu_epilogue(cuda_q, cuda_k, self.bias[:, :, :T, :T])
        self._compare_once("attn_gemm_with_mask_relu_epilogue", cuda_att, torch_att)

        torch_y = torch_att @ torch_v
        torch_y = torch_y.transpose(1, 2).contiguous().view(B, T, C)

        cuda_y = cuda_extension.att_mul_v_and_relayout(cuda_att, cuda_v, B, T, C)
        self._compare_once("att_mul_v_and_relayout", cuda_y, torch_y)

        return cuda_y