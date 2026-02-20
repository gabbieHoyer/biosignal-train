# src/biosignals/utils/distributed.py
from __future__ import annotations

import os

import torch
import torch.distributed as dist

"""
DDP utilities (single GPU + torchrun compatible).
"""


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    # torchrun sets these env vars
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return

    # LOCAL_RANK should exist under torchrun, but guard anyway
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
