# src/biosignals/models/architectures/encoder_classifier.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from biosignals.models.backbones.conv1d import ConvEncoder1D
from biosignals.models.heads.classification import ClassificationHead


class EncoderClassifier(nn.Module):
    """
    End-to-end classifier: ConvEncoder1D + masked pooling + linear head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        emb_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
        primary_modality: str = "main",
    ) -> None:
        super().__init__()
        self.primary_modality = str(primary_modality)
        self.encoder = ConvEncoder1D(in_channels=in_channels, emb_dim=emb_dim, depth=depth)
        self.head = ClassificationHead(
            in_dim=int(emb_dim), num_classes=int(num_classes), dropout=float(dropout)
        )

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        x = signals[self.primary_modality]
        z = self.encoder(x)  # (B,D,T)
        mask_t = meta.get("mask") if meta is not None else None
        return self.head.pool(z, mask_t=mask_t)  # (B,D)

    def forward(
        self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None
    ) -> torch.Tensor:
        x = signals[self.primary_modality]
        z = self.encoder(x)  # (B,D,T)
        mask_t = meta.get("mask") if meta is not None else None
        return self.head(z, mask_t=mask_t)  # (B,K)
