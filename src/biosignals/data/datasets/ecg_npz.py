# src/biosignals/data/datasets/ecg_npz.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from biosignals.data.datasets.base import BiosignalDataset
from biosignals.data.types import Sample

"""
Example dataset implementations:

Example 1: single-modality ECG in NPZ

"""


class EcgNpZDataset(BiosignalDataset):
    """
    Expected structure:
      root/
        train/
          000000.npz  (contains x: (C,T), y: (K,))
          ...
        val/
          ...
    """

    def __init__(self, root: str, split: str) -> None:
        super().__init__(split=split)
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, idx: int) -> Sample:
        fp = self.files[idx]
        data = np.load(fp, allow_pickle=True)
        x = data["x"].astype(np.float32)  # (C,T)
        y = data["y"]  # (K,) or scalar
        meta = {"id": fp.stem, "fs": float(data.get("fs", 500.0))}
        return Sample(signals={"main": x}, targets={"y": y}, meta=meta)
