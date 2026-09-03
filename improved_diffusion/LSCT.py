# LSCT -> LowRank Sparse Compress Transformer

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output

class FeedForwardLSC(nn.Module):
    def __init__(self, dim, loRaRatio=8, mlpRatio=4, dropout=0.):
        super().__init__()
        assert dim % loRaRatio == 0
        assert dim % mlpRatio == 0
        self.paramS = torch.zeros((1, 1, dim), requires_grad=False)
        self.mlp4L = nn.Sequential(
            nn.Linear(dim, dim // loRaRatio),
            nn.SiLU(),
            nn.Linear(dim // loRaRatio, dim),
        )
        self.mlp4S = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim // mlpRatio),
            nn.SiLU(),
            nn.Linear(2 * dim // mlpRatio, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        paramS = self.paramS.detach().to(x.device)
        paramZ = x
        paramL = self.mlp4L(x - paramS)
        assert paramZ.shape == paramL.shape
        paramS = self.mlp4S(torch.cat((paramZ, paramL), dim=2))
        self.paramS = paramS
        assert paramS.shape == paramL.shape
        return self.act(paramS + paramL)
    
# class FeedForwardLSC(nn.Module):
#     def __init__(self, dim, loRaRatio=8, mlpRatio=4, dropout=0.):
#         super().__init__()
#         assert dim % loRaRatio == 0
#         assert dim % mlpRatio == 0
#         self.paramS = torch.zeros((1, 1, dim), requires_grad=False)
#         # self.paramS = nn.Parameter(torch.zeros(1, 256, dim))
#         self.mlp4L = nn.Sequential(
#             nn.Linear(dim, dim // loRaRatio),
#             nn.SiLU(),
#             nn.Linear(dim // loRaRatio, dim),
#         )
#         self.mlp4S = nn.Sequential(
#             nn.Linear(2 * dim, 2 * dim // mlpRatio),
#             nn.SiLU(),
#             nn.Linear(2 * dim // mlpRatio, dim),
#         )
#         self.mlp4Z = nn.Sequential(
#             nn.Linear(2 * dim, 2 * dim // mlpRatio),
#             nn.SiLU(),
#             nn.Linear(2 * dim // mlpRatio, dim),
#         )
#         self.Linear4D = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.act = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         paramS = self.paramS.detach().to(x.device)
#         paramZ = x
#         paramL = self.mlp4L(x - paramS)
#         assert paramZ.shape == paramL.shape
#         paramS = self.mlp4S(torch.cat((paramZ, paramL), dim=2))
#         self.paramS = paramS
#         assert paramS.shape == paramL.shape
#         return self.act(self.mlp4Z(torch.cat((paramS + paramL, self.Linear4D(paramZ)), dim=2)))
        

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LSCTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForwardLSC(dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            grad_x = attn(x) + x

            x = ff(grad_x)

        return x
    
class LSCClassifier(nn.Module):
    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth,
            heads, pool='mean', channels=1, dim_heads=64, dropout=0., emb_dropout=0.,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0
        assert image_width % patch_width == 0

        num_patches = int((image_height / patch_height) * (image_width / patch_width))
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = LSCTransformer(dim, depth, heads, dim_heads, dropout, ista=5)
        self.pool = pool
        
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, num_classes)
        )
        

    def forward(self, x):
        x = self.to_patch_embedding(x)
        # print(x.shape)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.shape)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        
        feature = self.mlp_norm(x)
        return self.mlp_head(feature), feature

class LSCClassifier_Visulation(LSCClassifier):
    def forward(self, x):
        x = self.to_patch_embedding(x)
        
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # print(f"before Transformer {x.shape}")

        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        print(f"after mean pooling {x.shape}")

        # with torch.no_grad():
        feature = self.mlp_norm(x)
        print(f"before mlp {feature.shape}")
        out = self.mlp_head(feature)
        print(f"after mlp {out.shape})")
        return out
        
def LSCClassifier_Small(num_classes=5):
    return LSCClassifier(
        image_size=256,
        patch_size=16,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_heads=576 // 12
    )

def LSCClassifier_Base(num_classes=5):
    return LSCClassifier(
        image_size=256,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_heads=768 // 12,
    )

if __name__ == "__main__":
    model = LSCClassifier_Base()
    x = torch.randn((2, 1, 256, 256))
    c, f = model(x)
    pass