# src/biosignals/models/base.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

"""
Lightweight conventions for end-to-end models.

All trainable models should accept:
  forward(signals: Dict[str, Tensor], meta: Optional[dict]) -> Tensor or Dict[str, Tensor]

Optional convention:
  encode(...) -> Tensor   (useful for embedding extraction / SSL -> finetune reuse)
"""


class BiosignalModel(nn.Module):
    primary_modality: str = "main"

    def forward(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None):
        raise NotImplementedError

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        raise NotImplementedError("encode() is optional; implement when you want embeddings.")
