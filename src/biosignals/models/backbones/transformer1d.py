# src/biosignals/models/backbones/transformer1d.py
from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_pos_emb(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if seq_len <= 0:
        return torch.zeros((1, 0, dim), device=device, dtype=dtype)
    half = dim // 2
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
        * torch.arange(half, device=device, dtype=dtype)
        / max(half, 1)
    )
    angles = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((seq_len, 1), device=device, dtype=dtype)], dim=-1)
    return emb.unsqueeze(0)  # (1,N,D)


def padded_len_for_conv1d(t: int, kernel: int, stride: int) -> int:
    T = max(int(t), int(kernel))
    rem = (T - kernel) % stride
    if rem != 0:
        T += (stride - rem)
    return T


def pad_time(x: torch.Tensor, mask_t: torch.Tensor, T_pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = int(x.shape[-1])
    if T_pad == T:
        return x, mask_t
    if T_pad < T:
        return x[..., :T_pad], mask_t[..., :T_pad]
    pad = T_pad - T
    x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)
    m_pad = F.pad(mask_t, (0, pad), mode="constant", value=False)
    return x_pad, m_pad


class PatchEmbed1D(nn.Module):
    """
    (B,C,T) -> (B,N,D) via Conv1d(kernel=patch_size, stride=patch_stride)
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, patch_stride: Optional[int] = None) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_size)
        if self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be > 0")
        self.proj = nn.Conv1d(int(in_channels), int(embed_dim), kernel_size=self.patch_size, stride=self.patch_stride, bias=False)
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)      # (B,D,N)
        z = z.transpose(1, 2) # (B,N,D)
        return self.norm(z)


def downsample_mask_to_tokens(mask_t: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    m = mask_t.unsqueeze(1).float()  # (B,1,T)
    pooled = F.max_pool1d(m, kernel_size=int(patch_size), stride=int(patch_stride))
    return pooled.squeeze(1) > 0.0   # (B,N)


class Transformer1DEncoder(nn.Module):
    """
    Encoder only:
      (B,C,T) + mask_t(B,T) -> tokens(B,N,D) + tok_mask(B,N)
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 50,
        patch_stride: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_size)

        self.patch_embed = PatchEmbed1D(in_channels, self.embed_dim, self.patch_size, self.patch_stride)
        self.in_drop = nn.Dropout(float(dropout))

        ff = int(self.embed_dim * float(mlp_ratio))
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, mask_t: Optional[torch.Tensor] = None):
        B, C, T_in = x.shape
        device = x.device

        if mask_t is None:
            mask_t = torch.ones((B, T_in), device=device, dtype=torch.bool)
        else:
            mask_t = mask_t.to(device=device).bool()

        T_pad = padded_len_for_conv1d(T_in, kernel=self.patch_size, stride=self.patch_stride)
        x, mask_t = pad_time(x, mask_t, T_pad)

        tokens = self.patch_embed(x)  # (B,N,D)
        N = int(tokens.shape[1])

        tok_mask = downsample_mask_to_tokens(mask_t, self.patch_size, self.patch_stride)  # (B,N)
        none_valid = ~tok_mask.any(dim=1)
        if none_valid.any():
            tok_mask[none_valid, 0] = True

        pos = sinusoidal_pos_emb(N, self.embed_dim, device=device, dtype=tokens.dtype)
        tokens = self.in_drop(tokens + pos)

        key_padding = ~tok_mask  # True=ignore
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding)
        tokens = self.out_norm(tokens)
        return tokens, tok_mask
