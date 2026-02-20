# src/biosignals/metrics/detection.py
from __future__ import annotations

from typing import List, Tuple

import torch

Event = Tuple[int, int]  # (start_idx, end_idx) inclusive start, exclusive end


def extract_events_from_probs(
    probs_1d: torch.Tensor,  # (T,) float in [0,1]
    valid_mask: torch.Tensor,  # (T,) bool
    threshold: float = 0.5,
    min_len: int = 1,
) -> List[Event]:
    # only consider valid region
    x = probs_1d.clone()
    x[~valid_mask] = 0.0

    active = x >= threshold
    events: List[Event] = []
    in_evt = False
    start = 0

    for i in range(active.numel()):
        if active[i] and not in_evt:
            in_evt = True
            start = i
        elif (not active[i]) and in_evt:
            end = i
            if end - start >= min_len:
                events.append((start, end))
            in_evt = False

    if in_evt:
        end = int(active.numel())
        if end - start >= min_len:
            events.append((start, end))

    return events


def match_events_with_tolerance(
    pred: List[Event],
    true: List[Event],
    tol: int,  # tolerance in samples
) -> Tuple[int, int, int]:
    """
    Greedy matching:
      A predicted event matches a true event if the start times are within tol
      OR the intervals overlap (either criterion works well in practice; adjust to taste).
    """
    used = [False] * len(true)
    tp = 0

    for ps, pe in pred:
        best_j = -1
        best_dist = 10**18

        for j, (ts, te) in enumerate(true):
            if used[j]:
                continue

            overlap = not (pe <= ts or te <= ps)
            dist = abs(ps - ts)

            if overlap or dist <= tol:
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

        if best_j >= 0:
            used[best_j] = True
            tp += 1

    fp = len(pred) - tp
    fn = len(true) - tp
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


@torch.no_grad()
def f1_from_counts_tensor(
    tp: int, fp: int, fn: int, device: torch.device | None = None
) -> torch.Tensor:
    denom = 2 * tp + fp + fn
    val = (2 * tp / denom) if denom > 0 else 0.0
    return torch.tensor(val, device=device if device is not None else "cpu", dtype=torch.float32)


@torch.no_grad()
def event_f1_1d(
    probs_1d: torch.Tensor,  # (T,) in [0,1]
    true_events,  # List[Event]
    valid_mask: torch.Tensor,  # (T,) bool
    *,
    threshold: float = 0.5,
    min_len: int = 1,
    tol: int = 0,
) -> torch.Tensor:
    pred_events = extract_events_from_probs(
        probs_1d, valid_mask, threshold=threshold, min_len=min_len
    )
    tp, fp, fn = match_events_with_tolerance(pred_events, true_events, tol=tol)
    return f1_from_counts_tensor(tp, fp, fn, device=probs_1d.device)
