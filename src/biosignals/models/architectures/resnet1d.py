# src/biosignals/models/architectures/resnet1d.py
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from biosignals.models.backbones.resnet1d import ResNet1DEncoder
from biosignals.models.heads.classification import ClassificationHead


def _resample_mask_to_len(mask_t: torch.Tensor, T_out: int) -> torch.Tensor:
    """
    mask_t: (B,T) bool -> (B,T_out) bool via nearest interpolation.
    Works well for padded sequences where valid region is a prefix.
    """
    m = mask_t.float().unsqueeze(1)  # (B,1,T)
    m2 = F.interpolate(m, size=int(T_out), mode="nearest")
    return m2.squeeze(1) > 0.5


class ResNet1DClassifier(nn.Module):
    """
    End-to-end classifier: ResNet1DEncoder + masked pooling head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        primary_modality: str = "main",
    ) -> None:
        super().__init__()
        self.primary_modality = str(primary_modality)
        self.encoder = ResNet1DEncoder(in_channels=int(in_channels))
        self.head = ClassificationHead(
            in_dim=int(self.encoder.out_channels),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        x = signals[self.primary_modality]  # (B,C,T)
        z = self.encoder(x)  # (B,D,T')
        mask_t = meta.get("mask") if meta is not None else None
        if mask_t is not None:
            mask_z = _resample_mask_to_len(mask_t.to(device=z.device).bool(), z.shape[-1])
        else:
            mask_z = None
        return self.head.pool(z, mask_t=mask_z)  # (B,D)

    def forward(
        self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None
    ) -> torch.Tensor:
        x = signals[self.primary_modality]
        z = self.encoder(x)
        mask_t = meta.get("mask") if meta is not None else None
        if mask_t is not None:
            mask_z = _resample_mask_to_len(mask_t.to(device=z.device).bool(), z.shape[-1])
        else:
            mask_z = None
        return self.head(z, mask_t=mask_z)  # (B,K)
