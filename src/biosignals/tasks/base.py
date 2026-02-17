# src/biosignals/tasks/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import torch
from biosignals.data.types import Batch, Sample

class Task(ABC):
    name: str

    @abstractmethod
    def collate_fn(self) -> Callable[[List[Sample]], Batch]:
        ...

    @abstractmethod
    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Returns a dict containing at least: {"loss": loss_tensor}
        Additional scalars are logged.
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        ...


# tasks own collation: cleanly supports segmentation/detection with potentially different batching needs