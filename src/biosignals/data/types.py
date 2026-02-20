# src/biosignals/data/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]
Tensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class Sample:
    """
    Canonical unit emitted by any Dataset.

    signals: modality -> numpy array, typically float32, shape (C, T)
    targets: task-dependent labels (scalar, vector, sequence, events...)
    meta: subject/session ids, fs, timestamps, etc.
    """

    signals: dict[str, Array]
    targets: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class Batch:
    """
    Canonical batch produced by Task.collate_fn().
    """

    signals: dict[str, Tensor]  # e.g. (B, C, T) padded
    targets: dict[str, Any]  # tensors or ragged lists
    meta: dict[str, Any]  # lengths, ids, fs, masks...
