import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.init as init
from einops import rearrange


class PatchEmbeddingLayer(nn.Module):
    def __init__(self, patch_height, patch_width, frame_patch_size, patch_dim, dim):
        super(PatchEmbeddingLayer, self).__init__()
        self.rearrange = Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)',
                                   p1=patch_height, p2=patch_width, pf=frame_patch_size)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.norm1(x)
        x = self.linear(x)
        x = self.norm2(x)
        return x
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, length, dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.length = length
        self.positional_embedding = nn.Embedding(self.length, dim)
        init.normal_(self.positional_embedding.weight, std=0.02)  # same

    def forward(self, x):
        positions = torch.arange(self.length, dtype=torch.long, device=x.device)
        all_positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        pos_embeddings = self.positional_embedding(all_positions)
        pos_embeddings = pos_embeddings.view(x.shape)
        x = x + pos_embeddings
        return x
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
