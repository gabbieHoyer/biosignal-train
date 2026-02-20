# src/biosignals/utils/reproducibility.py
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:
    torch = None


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """
    Seed python, numpy, and torch (if installed).

    If deterministic=True, also configures torch/cudnn determinism.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: helps determinism for some CUDA GEMMs when deterministic algos enabled.
    # Only meaningful if you care about strict reproducibility.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    Seed numpy/random inside each DataLoader worker from torch's per-worker seed.

    Use by passing: DataLoader(..., worker_init_fn=worker_init_fn).
    """
    if torch is not None:
        # DataLoader sets a distinct initial_seed per worker for you.
        seed = int(torch.initial_seed() % 2**32)
    else:
        # Fallback: derive from numpy state
        base_seed = int(np.random.get_state()[1][0])
        seed = int((base_seed + worker_id) % 2**32)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)


def set_float32_matmul_precision(precision: str = "high") -> None:
    """
    Torch 2.x helper for matmul precision:
      - "highest" | "high" | "medium"
    Safe no-op if unavailable.
    """
    if torch is None:
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception:
            pass


def get_rng_state() -> Dict[str, Any]:
    """Capture RNG state snapshots (python/numpy/torch) for checkpointing."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state snapshots."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
