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




# =========================
# ✅ ModelNew（纯 CUDA）
# =========================
class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self.heads = heads
        self.dim = dim
        self.depth = depth

    def forward(self, img):
        p = self.patch_size

        # 1. Patch Embedding
        x = cuda_extension.PatchEmbedding_GEMM(
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            p
        )

        # 2. CLS + Pos
        x = cuda_extension.CLS_PosAdd_Elementwise(
            x,
            self.cls_token,
            self.pos_embedding
        )
        
        # 3. Transformer（逐 kernel，完全照抄 debug）
        for i in range(self.depth):
            layer = self.transformer.layers[i]
            residual_attn = x

            # QKV
            qkv = cuda_extension.SelfAttn_QKV_Proj(
                x,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias
            )
            

            # Scores
            attn_scores = cuda_extension.SelfAttn_QK_Scores(qkv, self.heads)

            # Softmax
            attn_probs = cuda_extension.Softmax_Reduction(attn_scores)

            # AttnV
            attn_ctx = cuda_extension.SelfAttn_AttnV_MatMul(
                attn_probs,
                qkv,
                self.heads
            )

            # OutProj + Residual
            x = cuda_extension.OutProj_GEMM_plus_Residual(
                attn_ctx,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                residual_attn
            )

            # LN1
            x = cuda_extension.LayerNorm_Reduction(
                x,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps
            )

            # MLP
            residual_mlp = x

            x = cuda_extension.MLP_FC1_GEMM_plus_GELU(
                x,
                layer.linear1.weight,
                layer.linear1.bias
            )

            x = cuda_extension.MLP_FC2_GEMM_plus_Residual(
                x,
                layer.linear2.weight,
                layer.linear2.bias,
                residual_mlp
            )

            x = cuda_extension.LayerNorm_Reduction(
                x,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps
            )

        # 4. Head
        x = cuda_extension.ToCLS_Slice(x)
        return x
        x = cuda_extension.Head_FC1_GEMM_plus_GELU(
            x,
            self.mlp_head[0].weight,
            self.mlp_head[0].bias
        )

        x = cuda_extension.Head_FC2_GEMM(
            x,
            self.mlp_head[3].weight,
            self.mlp_head[3].bias
        )

        return x
    
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.rand(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]