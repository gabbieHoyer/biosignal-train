# src/biosignals/models/fusion/perceiver_fusion.py
from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_pos_emb(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    if seq_len <= 0:
        return torch.zeros((1, 0, dim), device=device)
    half = dim // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device).float() / max(half, 1)
    )
    angles = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((seq_len, 1), device=device)], dim=-1)
    return emb.unsqueeze(0)


def pad_to_multiple_1d(x: torch.Tensor, mask_t: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if multiple <= 0:
        return x, mask_t
    b, c, t = x.shape
    rem = t % multiple
    if rem == 0:
        return x, mask_t
    pad = multiple - rem
    x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)
    m_pad = F.pad(mask_t, (0, pad), mode="constant", value=False)
    return x_pad, m_pad


def downsample_mask_to_tokens(mask_t: torch.Tensor, patch_size: int, n_tokens: int) -> torch.Tensor:
    b, t = mask_t.shape
    need = n_tokens * patch_size
    if t < need:
        pad = need - t
        mask_t = torch.cat([mask_t, torch.zeros((b, pad), device=mask_t.device, dtype=torch.bool)], dim=1)
    else:
        mask_t = mask_t[:, :need]
    return mask_t.view(b, n_tokens, patch_size).any(dim=-1)


class PatchEmbed1D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)      # (B, D, N)
        z = z.transpose(1, 2) # (B, N, D)
        return self.norm(z)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.cross_q = nn.LayerNorm(dim)
        self.cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.lat_ln1 = nn.LayerNorm(dim)
        self.lat_ln2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        hidden = int(dim * mlp_ratio)
        self.mlp1 = MLP(dim, hidden, dropout)
        self.mlp2 = MLP(dim, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.cross_q(latents)
        kv = self.cross_kv(inputs)
        x, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + self.drop(x)
        latents = latents + self.mlp1(self.lat_ln1(latents))

        q2 = self.lat_ln2(latents)
        y, _ = self.self_attn(q2, q2, q2, need_weights=False)
        latents = latents + self.drop(y)
        latents = latents + self.mlp2(latents)
        return latents


class PerceiverFusionEncoder(nn.Module):
    """
    Multimodal fusion encoder.

    Returns:
      pooled embedding (B, D)
    """
    def __init__(
        self,
        modalities: List[str],
        in_channels: Mapping[str, int],
        patch_size: Mapping[str, int],
        embed_dim: int = 256,
        num_latents: int = 64,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(modalities) == 0:
            raise ValueError("modalities must be non-empty")

        self.modalities = list(modalities)
        self.embed_dim = int(embed_dim)

        self.patch_embeds = nn.ModuleDict()
        self.patch_size = {}
        self.in_channels = {}
        for m in self.modalities:
            self.in_channels[m] = int(in_channels[m])
            self.patch_size[m] = int(patch_size[m])
            self.patch_embeds[m] = PatchEmbed1D(self.in_channels[m], self.embed_dim, self.patch_size[m])

        self.mod_emb = nn.Embedding(len(self.modalities), self.embed_dim)

        self.latents = nn.Parameter(torch.randn(int(num_latents), self.embed_dim) * 0.02)
        self.null_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([PerceiverBlock(self.embed_dim, int(num_heads), float(mlp_ratio), float(dropout)) for _ in range(int(depth))])
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        any_tensor = next(iter(signals.values()))
        b = int(any_tensor.shape[0])
        device = any_tensor.device
        dtype = any_tensor.dtype

        modality_mask = None
        if meta is not None and meta.get("modality_mask", None) is not None:
            modality_mask = meta["modality_mask"].to(device=device).bool()  # (B,M)

        mask_by_modality = None
        if meta is not None and isinstance(meta.get("mask_by_modality", None), dict):
            mask_by_modality = meta["mask_by_modality"]

        tokens_all: List[torch.Tensor] = []
        tokenmask_all: List[torch.Tensor] = []

        for mi, m in enumerate(self.modalities):
            present = (m in signals)
            if present:
                x = signals[m]
            else:
                t = int(any_tensor.shape[-1])
                x = torch.zeros((b, self.in_channels[m], t), device=device, dtype=dtype)

            if mask_by_modality is not None and m in mask_by_modality:
                mask_t = mask_by_modality[m].to(device=device).bool()
            elif meta is not None and meta.get("mask", None) is not None:
                mask_t = meta["mask"].to(device=device).bool()
            else:
                mask_t = torch.ones((b, x.shape[-1]), device=device, dtype=torch.bool)

            if modality_mask is not None:
                mask_t = mask_t & modality_mask[:, mi].unsqueeze(1)

            if not present:
                mask_t = torch.zeros_like(mask_t)

            x, mask_t = pad_to_multiple_1d(x, mask_t, multiple=self.patch_size[m])

            tok = self.patch_embeds[m](x)  # (B,N,D)
            n = int(tok.shape[1])

            tok = tok + sinusoidal_pos_emb(n, self.embed_dim, device=device)
            tok = tok + self.mod_emb.weight[mi].view(1, 1, -1)

            tok_mask = downsample_mask_to_tokens(mask_t, patch_size=self.patch_size[m], n_tokens=n)  # (B,N)

            tokens_all.append(tok)
            tokenmask_all.append(tok_mask)

        inputs = torch.cat(tokens_all, dim=1)      # (B,S,D)
        tok_mask = torch.cat(tokenmask_all, dim=1) # (B,S)

        null_tok = self.null_token.expand(b, 1, -1).to(device=device, dtype=dtype)
        inputs = torch.cat([null_tok, inputs], dim=1)
        tok_mask = torch.cat([torch.ones((b, 1), device=device, dtype=torch.bool), tok_mask], dim=1)

        key_padding_mask = ~tok_mask

        latents = self.latents.unsqueeze(0).expand(b, -1, -1).to(device=device, dtype=dtype)
        for blk in self.blocks:
            latents = blk(latents, inputs, key_padding_mask=key_padding_mask)

        pooled = self.out_norm(latents).mean(dim=1)  # (B,D)
        return pooled
