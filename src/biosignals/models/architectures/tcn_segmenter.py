# src/biosignals/models/architectures/tcn_segmenter.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from biosignals.models.backbones.tcn_segmenter import TCNEncoder1D
from biosignals.models.heads.segmentation import SegmentationHead


class TCNSegmenter(nn.Module):
    """
    End-to-end segmenter: TCNEncoder1D + 1x1 conv segmentation head.
    Returns (B,K,T).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 128,
        depth: int = 4,
        primary_modality: str = "main",
    ) -> None:
        super().__init__()
        self.primary_modality = str(primary_modality)
        self.encoder = TCNEncoder1D(
            in_channels=int(in_channels), width=int(width), depth=int(depth)
        )
        self.head = SegmentationHead(
            in_dim=int(self.encoder.out_channels), num_classes=int(num_classes)
        )

    def forward(
        self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None
    ) -> torch.Tensor:
        x = signals[self.primary_modality]
        z = self.encoder(x)
        return self.head(z)  # (B,K,T)
