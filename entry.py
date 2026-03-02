import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hashlib
import os
from torch.utils.cpp_extension import load

kernel_path = r"kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, 'rb') as f:
    file_content = f.read()
    content_hash = hashlib.md5(file_content).hexdigest()[:8]

cuda_extension = load(
    name=f"MiniGPTBlock_{content_hash}",
    sources=[kernel_path],
    verbose=False
)

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class ModelNew(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        # Collect all necessary parameters for the custom CUDA kernel
        # LayerNorm 1 parameters
        ln1_weight = self.ln_1.weight
        ln1_bias = self.ln_1.bias
        
        # CausalSelfAttention parameters
        c_attn_weight = self.attn.c_attn.weight
        c_attn_bias = self.attn.c_attn.bias
        c_proj_weight = self.attn.c_proj.weight
        c_proj_bias = self.attn.c_proj.bias
        attn_bias = self.attn.bias # This is a buffer, not a parameter, but needed for the kernel
        
        # LayerNorm 2 parameters
        ln2_weight = self.ln_2.weight
        ln2_bias = self.ln_2.bias
        
        # MLP parameters
        mlp_c_fc_weight = self.mlp.c_fc.weight
        mlp_c_fc_bias = self.mlp.c_fc.bias
        mlp_c_proj_weight = self.mlp.c_proj.weight
        mlp_c_proj_bias = self.mlp.c_proj.bias

        # Other necessary attributes
        n_embd = self.attn.n_embd
        n_head = self.attn.n_head
        
        # Call the custom CUDA kernel
        # The kernel is expected to perform the entire forward pass of the Transformer block
        # and return the output.
        return cuda_extension.MiniGPTBlock(
            x,
            ln1_weight, ln1_bias,
            c_attn_weight, c_attn_bias,
            c_proj_weight, c_proj_bias,
            attn_bias,
            ln2_weight, ln2_bias,
            mlp_c_fc_weight, mlp_c_fc_bias,
            mlp_c_proj_weight, mlp_c_proj_bias,
            n_embd, n_head
        )
    


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]