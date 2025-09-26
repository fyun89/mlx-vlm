from __future__ import annotations
from typing import Any, Dict, List
import mlx.core as mx
import mlx.nn as nn

# ---- Vision building blocks (ViT + DeepStack mergers) ----

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] -> [B, HW/ps^2, D]
        x = self.proj(x)            # [B, D, H', W']
        b, d, h, w = x.shape
        x = x.reshape(b, d, h * w).transpose(0, 2, 1)
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, bias: bool = True):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.linear_fc1 = nn.Linear(dim, hidden, bias=bias)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=bias)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act(self.linear_fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # names: attn.qkv.{weight,bias}
        self.proj = nn.Linear(dim, dim, bias=True)          # names: attn.proj.{weight,bias}

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B,T,h,d]
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)   # [B,T,h,h?] actually [B,h,T,T] if we permute—use implicit
        attn = nn.softmax(attn, axis=-1)
        out = attn @ v                                     # [B,T,h,d]
        out = out.reshape(B, T, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)                     # norm1.{weight,bias}
        self.attn = Attention(dim, num_heads, qkv_bias)    # attn.*
        self.norm2 = nn.LayerNorm(dim)                     # norm2.{weight,bias}
        self.mlp = MLP(dim, mlp_ratio)                     # mlp.linear_fc1/2.*

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DeepStackMerger(nn.Module):
    """One ‘merger’ block that fuses multi-level features."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)                      # deepstack_merger_list.N.norm.*
        self.linear_fc1 = nn.Linear(dim, dim, bias=True)   # ...linear_fc1.*
        self.linear_fc2 = nn.Linear(dim, dim, bias=True)   # ...linear_fc2.*
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act(self.linear_fc1(self.norm(x))))

class FinalMerger(nn.Module):
    """Final aggregator after deepstack list."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)                      # visual.merger.norm.*
        self.linear_fc1 = nn.Linear(dim, dim, bias=True)   # visual.merger.linear_fc1.*
        self.linear_fc2 = nn.Linear(dim, dim, bias=True)   # visual.merger.linear_fc2.*
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act(self.linear_fc1(self.norm(x))))

class Qwen3Vision(nn.Module):
    """
    Vision tower that mirrors Qwen3-VL parameter names:
      - vision.patch_embed.proj.*
      - vision.blocks.{i}.(norm1|attn|norm2|mlp.*).*
      - vision.deepstack_merger_list.{j}.(norm|linear_fc1|linear_fc2).*
      - vision.merger.(norm|linear_fc1|linear_fc2).*
      - vision.pos_embed.weight
    """
    def __init__(self, vcfg: Dict[str, Any]):
        super().__init__()
        ps = int(vcfg.get("patch_size", 16))
        in_ch = int(vcfg.get("in_channels", 3))
        dim = int(vcfg.get("hidden_size", 1024))
        nblks = int(vcfg.get("num_hidden_layers", 24))
        heads = int(vcfg.get("num_attention_heads", 16))
        mlp_ratio = float(vcfg.get("mlp_ratio", 4.0))
        qkv_bias = bool(vcfg.get("qkv_bias", True))
        self.patch_embed = PatchEmbed(in_ch, dim, ps)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, qkv_bias) for _ in range(nblks)])
        self.norm = nn.LayerNorm(dim)                      # visual.norm.* (top-level)
        # optional positional embedding (if present in weights)
        self.pos_embed = nn.Parameter(mx.zeros((1, 1, dim)))  # visual.pos_embed.weight
        # DeepStack
        ds_count = int(vcfg.get("deepstack_blocks", 3))    # default; overwritten by weights presence
        self.deepstack_merger_list = nn.ModuleList([DeepStackMerger(dim) for _ in range(ds_count)])
        self.merger = FinalMerger(dim)

    def __call__(self, images: List[mx.array]) -> mx.array:
        # Assume images already preprocessed to [B, C, H, W] tensors
        x = mx.concatenate(images, axis=0) if isinstance(images, (list, tuple)) else images
        x = self.patch_embed(x)
        # add pos_embed if has the correct length
        if self.pos_embed.value is not None and self.pos_embed.shape[-1] == x.shape[-1]:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # DeepStack path (simple serial merges)
        for m in self.deepstack_merger_list:
            x = m(x)
        x = self.merger(x)
        return x  # [B, T_img, dim]