# src/biosignals/data/datasets/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from biosignals.data.types import Sample


class BiosignalDataset(Dataset, ABC):
    """
    Base class for *raw* biosignal datasets.

    Responsibilities (Option B):
      - Load raw samples only (signals/targets/meta).
      - No transforms.
      - No disk caching.
      - No Hydra instantiation logic.

    Transforms/caching are applied via wrappers in biosignals.data.datamodule.
    """

    def __init__(self, split: str) -> None:
        self.split = str(split)

    def __getitem__(self, idx: int) -> Sample:
        return self._load_raw(idx)

    @abstractmethod
    def _load_raw(self, idx: int) -> Sample:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


# -------------------------------------------------
