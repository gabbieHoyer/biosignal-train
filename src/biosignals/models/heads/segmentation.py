# src/biosignals/models/heads/segmentation.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SegmentationHead(nn.Module):
    in_dim: int
    num_classes: int

    def __post_init__(self):
        super().__init__()
        self.proj = nn.Conv1d(int(self.in_dim), int(self.num_classes), kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B,D,T) -> logits: (B,K,T)
        return self.proj(feats)
