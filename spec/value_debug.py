import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os, hashlib

kernel_path = "/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level3/28_VisionTransformer/spec/kernel/kernel.cu"
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

class ModelDebug(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelDebug, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
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

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self._cache = {}
        self.report = []
        self.heads = heads
        self.dim = dim
        self.depth = depth

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

    def forward(self, img):
        p = self.patch_size

        # 1. Patch Embedding
        x_patches = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        torch_out = self.patch_to_embedding(x_patches)
        cuda_out = cuda_extension.PatchEmbedding_GEMM(img, self.patch_to_embedding.weight, self.patch_to_embedding.bias, p)
        self._compare_once("PatchEmbedding_GEMM", cuda_out, torch_out)
        x = torch_out # 强制对齐

        # 2. CLS Token & Pos Add
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        torch_out = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding
        cuda_out = cuda_extension.CLS_PosAdd_Elementwise(x, self.cls_token, self.pos_embedding)
        self._compare_once("CLS_PosAdd_Elementwise", cuda_out, torch_out)
        x = torch_out # 强制对齐

        # 3. Transformer Layers
        for i in range(self.depth):
            layer = self.transformer.layers[i]

            # --- Part A: Attention Block ---
            # 源码逻辑: x = self.norm1(x + self._sa_block(x))
            residual_attn = x 
            
            # (1) QKV Proj
            torch_qkv = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            cuda_qkv = cuda_extension.SelfAttn_QKV_Proj(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            self._compare_once(f"L{i}_SelfAttn_QKV_Proj", cuda_qkv, torch_qkv)
            
            # (2) Scores
            batch, seq, _ = torch_qkv.shape
            head_dim = self.dim // self.heads
            qkv_reshaped = torch_qkv.view(batch, seq, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
            torch_attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            cuda_attn_scores = cuda_extension.SelfAttn_QK_Scores(torch_qkv, self.heads)
            self._compare_once(f"L{i}_SelfAttn_QK_Scores", cuda_attn_scores, torch_attn_scores)

            # (3) Softmax
            torch_attn_probs = F.softmax(torch_attn_scores, dim=-1)
            cuda_attn_probs = cuda_extension.Softmax_Reduction(torch_attn_scores)
            self._compare_once(f"L{i}_Softmax_Reduction", cuda_attn_probs, torch_attn_probs)

            # (4) AttnV MatMul
            torch_attn_ctx = torch.matmul(torch_attn_probs, v).permute(0, 2, 1, 3).reshape(batch, seq, self.dim)
            cuda_attn_ctx = cuda_extension.SelfAttn_AttnV_MatMul(cuda_attn_probs, torch_qkv, self.heads)
            self._compare_once(f"L{i}_SelfAttn_AttnV_MatMul", cuda_attn_ctx, torch_attn_ctx)

            # (5) OutProj + Residual (关键：这是 norm1 的输入)
            torch_pre_norm1 = F.linear(torch_attn_ctx, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias) + residual_attn
            cuda_pre_norm1 = cuda_extension.OutProj_GEMM_plus_Residual(cuda_attn_ctx, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias, residual_attn)
            self._compare_once(f"L{i}_OutProj_plus_Residual", cuda_pre_norm1, torch_pre_norm1)

            # (6) LayerNorm 1
            torch_out_ln1 = F.layer_norm(torch_pre_norm1, (self.dim,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
            cuda_out_ln1 = cuda_extension.LayerNorm_Reduction(cuda_pre_norm1, layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
            self._compare_once(f"L{i}_LayerNorm_1", cuda_out_ln1, torch_out_ln1)
            x = torch_out_ln1 # 更新 x 进入 MLP 部分

            # --- Part B: MLP Block ---
            # 源码逻辑: x = self.norm2(x + self._ff_block(x))
            residual_mlp = x 
            
            # (1) FC1 + GELU
            torch_fc1 = F.gelu(F.linear(x, layer.linear1.weight, layer.linear1.bias))
            cuda_fc1 = cuda_extension.MLP_FC1_GEMM_plus_GELU(x, layer.linear1.weight, layer.linear1.bias)
            self._compare_once(f"L{i}_MLP_FC1_GELU", cuda_fc1, torch_fc1)
            
            # (2) FC2 + Residual (关键：这是 norm2 的输入)
            torch_pre_norm2 = F.linear(torch_fc1, layer.linear2.weight, layer.linear2.bias) + residual_mlp
            cuda_pre_norm2 = cuda_extension.MLP_FC2_GEMM_plus_Residual(cuda_fc1, layer.linear2.weight, layer.linear2.bias, residual_mlp)
            self._compare_once(f"L{i}_MLP_FC2_plus_Residual", cuda_pre_norm2, torch_pre_norm2)

            # (3) LayerNorm 2
            torch_out_ln2 = F.layer_norm(torch_pre_norm2, (self.dim,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
            cuda_out_ln2 = cuda_extension.LayerNorm_Reduction(cuda_pre_norm2, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
            self._compare_once(f"L{i}_LayerNorm_2", cuda_out_ln2, torch_out_ln2)
            x = torch_out_ln2 # 更新 x 进入下一层

        # 4. Final Head
        # (1) Slice
        torch_sliced = x[:, 0]
        cuda_sliced = cuda_extension.ToCLS_Slice(x)
        self._compare_once("ToCLS_Slice", cuda_sliced, torch_sliced)
        x = torch_sliced

        # (2) Head FC1 + GELU
        torch_head_fc1 = F.gelu(F.linear(x, self.mlp_head[0].weight, self.mlp_head[0].bias))
        cuda_head_fc1 = cuda_extension.Head_FC1_GEMM_plus_GELU(x, self.mlp_head[0].weight, self.mlp_head[0].bias)
        self._compare_once("Head_FC1_GELU", cuda_head_fc1, torch_head_fc1)
        x = torch_head_fc1

        # (3) Head FC2 (Final)
        torch_final = F.linear(x, self.mlp_head[3].weight, self.mlp_head[3].bias)
        cuda_final = cuda_extension.Head_FC2_GEMM(x, self.mlp_head[3].weight, self.mlp_head[3].bias)
        self._compare_once("Final_Output", cuda_final, torch_final)
        
        return torch_final


        