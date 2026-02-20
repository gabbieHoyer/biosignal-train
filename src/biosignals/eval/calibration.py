# src/biosignals/eval/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=1, keepdims=True)).astype(np.float32)


def nll_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    lt = torch.from_numpy(logits).float()
    yt = torch.from_numpy(y_true).long()
    return float(F.cross_entropy(lt, yt, reduction="mean").item())


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Multi-class ECE based on confidence of predicted class.
    probs: (N,K), y_true: (N,)
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        w = float(np.sum(m)) / n
        ece += w * abs(float(np.mean(acc[m])) - float(np.mean(conf[m])))

    return float(ece)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if t <= 0:
        t = 1.0
    return (logits / t).astype(np.float32)


@dataclass(frozen=True)
class TemperatureResult:
    temperature: float
    nll_before: float
    nll_after: float


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, max_iter: int = 50
) -> TemperatureResult:
    """
    Fit a single temperature scalar on validation logits to minimize NLL.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lt = torch.from_numpy(logits).float().to(device)
    yt = torch.from_numpy(y_true).long().to(device)

    t = torch.ones((), device=device, requires_grad=True)

    def _loss():
        return F.cross_entropy(lt / t.clamp(min=1e-6), yt, reduction="mean")

    nll_before = float(F.cross_entropy(lt, yt, reduction="mean").item())

    optim = torch.optim.LBFGS([t], lr=0.5, max_iter=int(max_iter))

    def closure():
        optim.zero_grad()
        loss = _loss()
        loss.backward()
        return loss

    optim.step(closure)

    temp = float(t.detach().clamp(min=1e-6).item())
    nll_after = float(F.cross_entropy(lt / temp, yt, reduction="mean").item())

    return TemperatureResult(temperature=temp, nll_before=nll_before, nll_after=nll_after)


def calibration_summary(
    logits: np.ndarray, y_true: np.ndarray, temperature: float, n_bins: int = 15
) -> Dict[str, float]:
    probs = softmax_np(apply_temperature(logits, temperature))
    return {
        "nll": nll_from_logits(apply_temperature(logits, temperature), y_true),
        "ece": expected_calibration_error(probs, y_true, n_bins=n_bins),
        "avg_conf": float(np.mean(probs.max(axis=1))),
        "acc": float(np.mean((probs.argmax(axis=1) == y_true).astype(np.float32))),
    }
