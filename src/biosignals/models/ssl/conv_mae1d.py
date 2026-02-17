# src/biosignals/models/ssl/conv_mae1d.py
from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import nn

from biosignals.models.backbones.conv1d import ConvEncoder1D

"""
SSL model: masked reconstruction (denoising) on time axis.

Forward returns x_hat: (B,C,T) to match MaskedReconstructionTask.
"""

class ConvMAE1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_dim: int = 256,
        depth: int = 4,
        primary_modality: str = "main",
    ) -> None:
        super().__init__()
        self.primary_modality = str(primary_modality)
        self.encoder = ConvEncoder1D(in_channels=in_channels, emb_dim=emb_dim, depth=depth)
        self.decoder = nn.Conv1d(int(emb_dim), int(in_channels), kernel_size=1)

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        x = signals[self.primary_modality]   # (B,C,T)
        return self.encoder(x)               # (B,D,T)

    def forward(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        z = self.encode(signals, meta)
        return self.decoder(z)               # (B,C,T)
