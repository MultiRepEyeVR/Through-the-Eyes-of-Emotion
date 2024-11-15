import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class UnifiedCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out
    

class AttentionFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class LearnableWeightedSum(nn.Module):
    def __init__(self, num_modalities):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities))

    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        return (x * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)