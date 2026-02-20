# src/biosignals/utils/checkpointing.py
# src/biosignals/utils/checkpointing.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

"""
Partial weight loading utility (SSL encoder → finetune encoder)
"""


def load_partial_state_dict(
    model: torch.nn.Module,
    ckpt_path: str,
    src_prefix: str,
    dst_prefix: str,
    strict_shapes: bool = True,
) -> Dict[str, int]:
    """
    Loads parameters from checkpoint whose keys start with src_prefix
    into model keys with dst_prefix.

    Returns counts: {"loaded": n, "skipped": m}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # allow raw state_dict or {"model": ...}

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    new_state = {}

    for k, v in state.items():
        if not k.startswith(src_prefix):
            continue
        new_k = dst_prefix + k[len(src_prefix) :]
        if new_k not in model_state:
            skipped += 1
            continue
        if strict_shapes and tuple(model_state[new_k].shape) != tuple(v.shape):
            skipped += 1
            continue
        new_state[new_k] = v
        loaded += 1

    model.load_state_dict({**model_state, **new_state})
    return {"loaded": loaded, "skipped": skipped}


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves a training checkpoint. Uses atomic-ish replace to reduce partial files.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": _unwrap_model(model).state_dict(),
    }
    if optim is not None:
        payload["optim"] = optim.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = dict(extra)

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
