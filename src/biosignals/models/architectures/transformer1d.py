# src/biosignals/models/architectures/transformer1d.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from biosignals.models.backbones.transformer1d import Transformer1DEncoder
from biosignals.models.heads.classification import ClassificationHead


class Transformer1DClassifier(nn.Module):
    """
    End-to-end classifier: patchify + Transformer encoder + masked pooling head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 50,
        patch_stride: int = 50,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        primary_modality: str = "main",
    ) -> None:
        super().__init__()
        self.primary_modality = str(primary_modality)
        self.encoder = Transformer1DEncoder(
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            patch_size=int(patch_size),
            patch_stride=int(patch_stride),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        # Pool over token axis using same head module (treat tokens as "time")
        self.head = ClassificationHead(
            in_dim=int(embed_dim), num_classes=int(num_classes), dropout=float(dropout)
        )

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        x = signals[self.primary_modality]  # (B,C,T)
        mask_t = meta.get("mask") if meta is not None else None
        tokens, tok_mask = self.encoder(x, mask_t=mask_t)  # (B,N,D), (B,N)
        feats = tokens.transpose(1, 2)  # (B,D,N)
        return self.head.pool(feats, mask_t=tok_mask)  # (B,D)

    def forward(
        self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None
    ) -> torch.Tensor:
        x = signals[self.primary_modality]
        mask_t = meta.get("mask") if meta is not None else None
        tokens, tok_mask = self.encoder(x, mask_t=mask_t)
        feats = tokens.transpose(1, 2)  # (B,D,N)
        return self.head(feats, mask_t=tok_mask)  # (B,K)
