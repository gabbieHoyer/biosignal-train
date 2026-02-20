# src/biosignals/data/datasets/wrappers.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from biosignals.data.types import Sample

Transform = Callable[[Sample], Sample]


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Optional[Transform] = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Sample:
        sample = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


class CacheDataset(Dataset):
    def __init__(self, dataset: Dataset, cache_dir: str, *, prefix: str) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = str(prefix)

    def __len__(self) -> int:
        return len(self.dataset)

    def _path(self, idx: int) -> Path:
        return self.cache_dir / f"{self.prefix}_{idx:07d}.pt"

    def count_cached(self) -> int:
        """
        Count cached *.pt files for this prefix.
        Robust even with num_workers>0 because it scans the filesystem.
        """
        pat = f"{self.prefix}_*.pt"
        return sum(1 for _ in self.cache_dir.glob(pat))

    def __getitem__(self, idx: int) -> Sample:
        path = self._path(idx)
        if path.exists():
            return torch.load(path)

        sample = self.dataset[idx]

        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(sample, tmp)
        tmp.replace(path)

        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


# ---------------------------------------------------------------
