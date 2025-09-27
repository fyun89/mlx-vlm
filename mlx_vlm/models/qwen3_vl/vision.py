# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Any, Dict, List
import mlx.core as mx
import mlx.nn as nn

# Names mirror HF: visual.patch_embed.proj.* / visual.blocks.N.* / visual.deepstack_* / visual.merger.* / visual.pos_embed.weight

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
    def __call__(self, x: mx.array) -> mx.array:  # x: [B,C,H,W]
        x = self.proj(x)              # [B,D,H',W']
        b, d, h, w = x.shape
        x = x.reshape(b, d, h*w).transpose(0, 2, 1)  # [B,T,D]
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        h = int(dim * mlp_ratio)
        self.linear_fc1 = nn.Linear(dim, h, bias=True)
        self.linear_fc2 = nn.Linear(h, dim, bias=True)
        self.act = nn.GELU()
    def __call__(self, x): return self.linear_fc2(self.act(self.linear_fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)  # attn.qkv.*
        self.proj = nn.Linear(dim, dim, bias=True)       # attn.proj.*
    def __call__(self, x):
        B,T,D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d)
        q,k,v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]       # [B,T,h,d]
        attn = (q * self.scale) @ k.transpose(0,1,3,2)         # [B,T,h,T]
        attn = nn.softmax(attn, axis=-1)
        out = attn @ v                                         # [B,T,h,d]
        out = out.reshape(B, T, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)     # norm1.*
        self.attn  = Attention(dim, heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)     # norm2.*
        self.mlp   = MLP(dim, mlp_ratio)   # mlp.linear_fc1/2.*
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DeepStackMerger(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)                 # deepstack_merger_list.i.norm.*
        self.linear_fc1 = nn.Linear(dim, dim, True)   # ...linear_fc1.*
        self.linear_fc2 = nn.Linear(dim, dim, True)   # ...linear_fc2.*
        self.act = nn.GELU()
    def __call__(self, x): return self.linear_fc2(self.act(self.linear_fc1(self.norm(x))))

class FinalMerger(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)                 # visual.merger.norm.*
        self.linear_fc1 = nn.Linear(dim, dim, True)   # visual.merger.linear_fc1.*
        self.linear_fc2 = nn.Linear(dim, dim, True)   # visual.merger.linear_fc2.*
        self.act = nn.GELU()
    def __call__(self, x): return self.linear_fc2(self.act(self.linear_fc1(self.norm(x))))

class Qwen3Vision(nn.Module):
    def __init__(self, vcfg: Dict[str, Any]):
        super().__init__()
        ps = int(vcfg.get("patch_size", 16))
        in_ch = int(vcfg.get("in_channels", 3))
        dim = int(vcfg.get("hidden_size", 1024))
        nblks = int(vcfg.get("num_hidden_layers", 24))
        heads = int(vcfg.get("num_attention_heads", 16))
        mlp_ratio = float(vcfg.get("mlp_ratio", 4.0))
        qkv_bias = bool(vcfg.get("qkv_bias", True))

        # keep for dynamic growth
        self._dim = dim
        self._heads = heads
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias

        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=ps, stride=ps, bias=True)
        self.blocks: list[Block] = [Block(dim, heads, mlp_ratio, qkv_bias) for _ in range(nblks)]
        self.norm = nn.LayerNorm(dim)

        # simple buffer for pos embed (if present in weights, it’ll get overwritten)
        self.pos_embed = mx.zeros((1, 1, dim))

        ds_count = int(vcfg.get("deepstack_blocks", 0))
        self.deepstack_merger_list: list[DeepStackMerger] = [DeepStackMerger(dim) for _ in range(ds_count)]
        self.merger = FinalMerger(dim)

    # --- dynamic growth helpers ---
    def ensure_block_count(self, n: int):
        cur = len(self.blocks)
        for _ in range(cur, n):
            self.blocks.append(Block(self._dim, self._heads, self._mlp_ratio, self._qkv_bias))

    def ensure_deepstack_count(self, n: int):
        cur = len(self.deepstack_merger_list)
        for _ in range(cur, n):
            self.deepstack_merger_list.append(DeepStackMerger(self._dim))

    def __call__(self, images: List[mx.array]) -> mx.array:
        x = mx.concatenate(images, axis=0) if isinstance(images, (list, tuple)) else images  # [B,C,H,W]
        x = self.patch_embed(x)           # [B,T,D]
        if self.pos_embed is not None and self.pos_embed.shape[-1] == x.shape[-1]:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        for m in self.deepstack_merger_list:
            x = m(x)
        x = self.merger(x)
        return x                          # [B,T_img,D]