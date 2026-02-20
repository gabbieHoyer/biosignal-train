# src/biosignals/engine/loops.py
from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast

from biosignals.utils.distributed import is_distributed, is_main_process

SUM_KEYS_DEFAULT: Set[str] = {"tp", "fp", "fn"}


def move_to_device(obj: Any, device: str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def batch_to_device(batch: Any, device: str) -> Any:
    batch.signals = move_to_device(batch.signals, device)
    batch.targets = move_to_device(batch.targets, device)
    batch.meta = move_to_device(batch.meta, device)
    return batch


def _batch_size(batch: Any) -> int:
    if hasattr(batch, "signals") and isinstance(batch.signals, dict) and len(batch.signals) > 0:
        x0 = next(iter(batch.signals.values()))
        if torch.is_tensor(x0) and x0.ndim >= 1:
            return int(x0.shape[0])
    return 1


def _accumulate(
    sums: Dict[str, torch.Tensor],
    out: Dict[str, Any],
    weight: int,
    device: str,
    sum_keys: Set[str],
) -> None:
    w = torch.tensor(float(weight), device=device)
    for k, v in out.items():
        if v is None:
            continue

        if torch.is_tensor(v):
            if v.ndim != 0:
                continue
            t = v.detach()
        else:
            t = torch.tensor(float(v), device=device)

        if k in sum_keys:
            sums[k] = sums.get(k, torch.tensor(0.0, device=device)) + t
        else:
            sums[k] = sums.get(k, torch.tensor(0.0, device=device)) + t * w


def _ddp_all_reduce_epoch_sums(
    sums: Dict[str, torch.Tensor],
    total_weight: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not is_distributed():
        return sums, total_weight

    import torch.distributed as dist

    keys = sorted(sums.keys())
    if len(keys) == 0:
        dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
        return sums, total_weight

    stacked = torch.stack([sums[k] for k in keys] + [total_weight])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)

    reduced = {k: stacked[i] for i, k in enumerate(keys)}
    total = stacked[-1]
    return reduced, total


def _finalize(
    sums: Dict[str, torch.Tensor],
    total_weight: torch.Tensor,
    sum_keys: Set[str],
) -> Dict[str, float]:
    denom = float(total_weight.item()) if float(total_weight.item()) > 0 else 1.0
    out: Dict[str, float] = {}

    for k, v in sums.items():
        if k in sum_keys:
            out[k] = float(v.item())
        else:
            out[k] = float((v / denom).item())

    # Detection: compute global F1 from aggregated counts
    if all(k in out for k in ("tp", "fp", "fn")):
        tp, fp, fn = out["tp"], out["fp"], out["fn"]
        denom2 = 2 * tp + fp + fn
        out["event_f1"] = (2 * tp / denom2) if denom2 > 0 else 0.0

    return out


def train_one_epoch(
    model: torch.nn.Module,
    task: Any,
    loader: Any,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    *,
    epoch: int,
    amp: bool = True,
    grad_clip: float = 1.0,
    log_every: int = 50,
    device: str = "cuda",
    sum_keys: Set[str] = SUM_KEYS_DEFAULT,
) -> Dict[str, float]:
    model.train()

    sums: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=device)

    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        bs = _batch_size(batch)

        optim.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            out = task.training_step(model, batch)
            if "loss" not in out:
                raise KeyError("task.training_step must return a dict containing key 'loss'")
            loss = out["loss"]

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optim)
        scaler.update()

        _accumulate(sums, out, bs, device=device, sum_keys=sum_keys)
        total += torch.tensor(float(bs), device=device)

        if is_main_process() and (step % log_every == 0):
            parts = []
            for k, v in out.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    parts.append(f"{k}={v.item():.4f}")
            print(f"[epoch {epoch} step {step}] " + " ".join(parts))

    sums, total = _ddp_all_reduce_epoch_sums(sums, total)
    return _finalize(sums, total, sum_keys=sum_keys)


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    task: Any,
    loader: Any,
    *,
    device: str = "cuda",
    sum_keys: Set[str] = SUM_KEYS_DEFAULT,
) -> Dict[str, float]:
    model.eval()

    sums: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=device)

    for batch in loader:
        batch = batch_to_device(batch, device)
        bs = _batch_size(batch)

        out = task.validation_step(model, batch)
        _accumulate(sums, out, bs, device=device, sum_keys=sum_keys)
        total += torch.tensor(float(bs), device=device)

    sums, total = _ddp_all_reduce_epoch_sums(sums, total)
    return _finalize(sums, total, sum_keys=sum_keys)


# -----------------------------------------------
