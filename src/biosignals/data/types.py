# src/biosignals/data/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


@dataclass(frozen=True)
class Sample:
    """
    Canonical unit emitted by any Dataset.

    signals: dict of modality -> np.ndarray float32, shape (C, T)
    targets: task-dependent labels (scalar, vector, sequence, events...)
    meta: subject/session ids, fs, timestamps, etc.
    """

    signals: Dict[str, Array]
    targets: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class Batch:
    """
    Canonical batch produced by Task.collate_fn().
    """

    signals: Dict[str, Tensor]  # e.g. (B, C, T) padded
    targets: Dict[str, Any]  # tensors or ragged lists
    meta: Dict[str, Any]  # lengths, ids, fs, masks...
