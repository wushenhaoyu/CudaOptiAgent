import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# ✅ Model（纯 Torch）
# =========================
class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(Model, self).__init__()
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
        x_patches = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        x = self.patch_to_embedding(x_patches)

        # 2. CLS + Pos
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding
        
        # 3. Transformer（原生）
        x = self.transformer(x)
        
        # 4. Head
        x = x[:, 0]
        return x
        x = F.gelu(F.linear(x, self.mlp_head[0].weight, self.mlp_head[0].bias))
        x = F.linear(x, self.mlp_head[3].weight, self.mlp_head[3].bias)

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