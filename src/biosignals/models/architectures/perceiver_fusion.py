# src/biosignals/models/architectures/perceiver_fusion.py
from __future__ import annotations
from typing import Dict, List, Mapping, Optional

import torch
from torch import nn

from biosignals.models.fusion.perceiver_fusion import PerceiverFusionEncoder


class PerceiverFusionClassifier(nn.Module):
    """
    End-to-end classifier: PerceiverFusionEncoder -> Linear head
    """
    def __init__(
        self,
        modalities: List[str],
        in_channels: Mapping[str, int],
        patch_size: Mapping[str, int],
        num_classes: int,
        embed_dim: int = 256,
        num_latents: int = 64,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = PerceiverFusionEncoder(
            modalities=modalities,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=int(embed_dim),
            num_latents=int(num_latents),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        return self.encoder(signals, meta)

    def forward(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        z = self.encode(signals, meta)
        return self.head(z)
