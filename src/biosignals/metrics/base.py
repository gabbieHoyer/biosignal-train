# src/biosignals/metrics/base.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray, float, int]


def to_tensor(
    x: ArrayLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def _extract_from_mapping(
    obj: Any,
    key: Optional[str],
    fallbacks: Sequence[str],
) -> Any:
    """
    If obj is a Mapping, return obj[key] if key exists, else try fallbacks.
    If obj is not a Mapping, return obj unchanged.
    """
    if not isinstance(obj, Mapping):
        return obj

    if key is not None and key in obj:
        return obj[key]

    for k in fallbacks:
        if k in obj:
            return obj[k]

    return obj


class Metric:
    """
    Minimal metric interface.

    Important for this codebase:
      - compute() returns a torch scalar tensor (0-dim), so trainer can do v.item().
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__.lower()

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, preds: Any, target: Any = None, **kwargs: Any) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device) -> Metric:
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self


class MeanMetric(Metric):
    """
    Running mean of a scalar.
    Typical use: MeanMetric(value_key="loss") and update(out_dict).
    """

    def __init__(self, name: str = "mean", value_key: Optional[str] = "loss") -> None:
        super().__init__(name=name)
        self.value_key = value_key
        self.total = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def reset(self) -> None:
        self.total = torch.tensor(0.0, device=self.total.device)
        self.count = torch.tensor(0.0, device=self.count.device)

    def update(
        self, value: Any, target: Any = None, n: Optional[int] = None, **kwargs: Any
    ) -> None:
        v = _extract_from_mapping(value, self.value_key, fallbacks=("loss", "value", self.name))
        if isinstance(v, Mapping):
            raise TypeError(
                f"MeanMetric could not extract scalar from mapping. Available keys: {list(v.keys())}."
            )

        t = to_tensor(v, device=self.total.device, dtype=torch.float32).detach()

        if t.ndim > 0:
            self.total += t.sum()
            self.count += torch.tensor(float(t.numel()), device=self.count.device)
        else:
            self.total += t
            self.count += torch.tensor(float(n or 1), device=self.count.device)

    def compute(self) -> torch.Tensor:
        if float(self.count.item()) == 0.0:
            return torch.tensor(float("nan"), device=self.total.device)
        return self.total / self.count


class Accuracy(Metric):
    """
    Top-k accuracy (binary or multiclass).

    Locked defaults to your codebase conventions:
      - preds: model output tensor OR dict containing "logits"
      - target: label tensor OR dict containing "y"
    """

    def __init__(
        self,
        name: str = "acc",
        topk: int = 1,
        threshold: float = 0.5,
        pred_key: Optional[str] = "logits",
        target_key: Optional[str] = "y",
    ) -> None:
        super().__init__(name=name)
        if topk < 1:
            raise ValueError("topk must be >= 1")
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.pred_key = pred_key
        self.target_key = target_key

        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

    def reset(self) -> None:
        self.correct = torch.tensor(0.0, device=self.correct.device)
        self.total = torch.tensor(0.0, device=self.total.device)

    def update(self, preds: Any, target: Any = None, **kwargs: Any) -> None:
        p = _extract_from_mapping(preds, self.pred_key, fallbacks=("logits", "preds", "y_pred"))
        y = _extract_from_mapping(
            target, self.target_key, fallbacks=("y", "label", "labels", "target")
        )

        if isinstance(p, Mapping):
            raise TypeError(
                f"Accuracy could not extract tensor from preds mapping. Keys: {list(p.keys())}. "
                f"Set pred_key=... if needed."
            )
        if y is None or isinstance(y, Mapping):
            raise TypeError(
                "Accuracy requires targets as a tensor or a dict containing 'y'. "
                "Set target_key=... if needed."
            )

        p = to_tensor(p, device=self.correct.device).detach()
        y = to_tensor(y, device=self.correct.device).long().view(-1)

        # Binary: (N,) or (N,1)
        if p.ndim == 1 or (p.ndim == 2 and p.shape[-1] == 1):
            p1 = p.view(-1)
            # If probs in [0,1] threshold=threshold, else treat as logits with threshold=0
            pmin = float(p1.min().item())
            pmax = float(p1.max().item())
            is_prob = (pmin >= 0.0) and (pmax <= 1.0)
            thr = self.threshold if is_prob else 0.0
            yhat = (p1 > thr).long()

            self.correct += (yhat == y).float().sum()
            self.total += torch.tensor(float(y.numel()), device=self.total.device)
            return

        # Multiclass: (N, C)
        if self.topk == 1:
            yhat = p.argmax(dim=-1).view(-1)
            self.correct += (yhat == y).float().sum()
            self.total += torch.tensor(float(y.numel()), device=self.total.device)
            return

        k = min(self.topk, p.shape[-1])
        topk_idx = p.topk(k, dim=-1).indices  # (N, k)
        y_exp = y.view(-1, 1).expand_as(topk_idx)
        hits = (topk_idx == y_exp).any(dim=-1).float()

        self.correct += hits.sum()
        self.total += torch.tensor(float(y.numel()), device=self.total.device)

    def compute(self) -> torch.Tensor:
        if float(self.total.item()) == 0.0:
            return torch.tensor(float("nan"), device=self.correct.device)
        return self.correct / self.total


class MetricCollection:
    """Small container to manage multiple metrics together."""

    def __init__(self, metrics: Mapping[str, Metric]) -> None:
        self.metrics: Dict[str, Metric] = dict(metrics)

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def to(self, device: torch.device) -> MetricCollection:
        for m in self.metrics.values():
            m.to(device)
        return self

    def update(self, preds: Any, target: Any = None, **kwargs: Any) -> None:
        for m in self.metrics.values():
            m.update(preds, target, **kwargs)

    def compute(self) -> Dict[str, torch.Tensor]:
        return {name: m.compute() for name, m in self.metrics.items()}


# ------------
# task can do:

# out = model(batch.signals)                  # dict with "logits"
# y = batch.targets["y"]
# acc_metric.update(out, batch.targets)       # uses logits + y by default
# acc = acc_metric.compute()                  # torch scalar
# return {"loss": loss, "acc": acc}

# and trainer will print acc.item().
