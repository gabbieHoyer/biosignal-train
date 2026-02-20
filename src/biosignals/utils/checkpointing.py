# src/biosignals/utils/checkpointing.py
# src/biosignals/utils/checkpointing.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

# from __future__ import annotations
# from typing import Dict, Optional

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Optional, Union

# from biosignals.utils.distributed import is_main_process

# PathLike = Union[str, Path]

# import torch

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


# ------------------------------------------------------------------------


# def _unwrap_model(model: Any) -> Any:
#     return model.module if hasattr(model, "module") else model


# def save_checkpoint(
#     path: PathLike,
#     *,
#     model: Any,
#     optim: Optional[Any] = None,
#     scaler: Optional[Any] = None,
#     epoch: int = 0,
#     step: int = 0,
#     metrics: Optional[Dict[str, float]] = None,
# ) -> None:
#     if not is_main_process():
#         return

#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)

#     state: Dict[str, Any] = {
#         "model": _unwrap_model(model).state_dict(),
#         "epoch": int(epoch),
#         "step": int(step),
#         "metrics": dict(metrics or {}),
#     }
#     if optim is not None:
#         state["optim"] = optim.state_dict()
#     if scaler is not None:
#         state["scaler"] = scaler.state_dict()

#     tmp = path.with_suffix(path.suffix + ".tmp")
#     torch.save(state, tmp)
#     tmp.replace(path)


# @dataclass
# class CheckpointManager:
#     dir: Path
#     monitor: str = "val/loss"
#     mode: str = "min"  # "min" or "max"
#     save_best: bool = True
#     save_last: bool = True
#     save_every_epochs: int = 1

#     def __post_init__(self) -> None:
#         self.dir = Path(self.dir)
#         self.dir.mkdir(parents=True, exist_ok=True)
#         self.best: Optional[float] = None

#     def _is_better(self, x: float) -> bool:
#         if self.best is None:
#             return True
#         return (x < self.best) if self.mode == "min" else (x > self.best)

#     def maybe_save(
#         self,
#         *,
#         model: Any,
#         optim: Any,
#         scaler: Any,
#         epoch: int,
#         step: int,
#         metrics: Dict[str, float],
#     ) -> Dict[str, Optional[str]]:
#         saved: Dict[str, Optional[str]] = {"last": None, "best": None, "epoch": None}

#         if self.save_last:
#             p = self.dir / "last.pt"
#             save_checkpoint(p, model=model, optim=optim, scaler=scaler, epoch=epoch, step=step, metrics=metrics)
#             saved["last"] = str(p)

#         if self.save_every_epochs and (epoch % int(self.save_every_epochs) == 0):
#             p = self.dir / f"epoch_{epoch:03d}.pt"
#             save_checkpoint(p, model=model, optim=optim, scaler=scaler, epoch=epoch, step=step, metrics=metrics)
#             saved["epoch"] = str(p)

#         if self.save_best and (self.monitor in metrics):
#             val = float(metrics[self.monitor])
#             if self._is_better(val):
#                 self.best = val
#                 p = self.dir / "best.pt"
#                 save_checkpoint(p, model=model, optim=optim, scaler=scaler, epoch=epoch, step=step, metrics=metrics)
#                 saved["best"] = str(p)

#         return saved
