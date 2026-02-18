# src/biosignals/data/datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from biosignals.utils.distributed import is_distributed, get_rank
from biosignals.utils.reproducibility import worker_init_fn

from biosignals.data.transforms.factory import build_transform
from biosignals.data.datasets.wrappers import TransformDataset, CacheDataset


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2


def make_loader(dataset, task, cfg: DataConfig, shuffle: bool) -> DataLoader:
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=cfg.drop_last)
        shuffle = False

    kwargs = {}
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
        kwargs["persistent_workers"] = True
    else:
        kwargs["persistent_workers"] = False

    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=bool(cfg.drop_last),
        collate_fn=task.collate_fn(),
        worker_init_fn=worker_init_fn,
        **kwargs,
    )


def _maybe_cache_dir(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    return s


def _get_cache_dir(dataset_cfg: DictConfig, split: str) -> Optional[str]:
    """
    Reads cache config from dataset_cfg.cache.{train|val|test} if present.
    """
    if "cache" not in dataset_cfg or dataset_cfg.cache is None:
        return None
    cache_cfg = dataset_cfg.cache
    if split not in cache_cfg:
        return None
    return _maybe_cache_dir(cache_cfg.get(split))


def build_dataset(
    *,
    split_cfg: DictConfig,
    transform_cfg: Any,
    cache_dir: Optional[str],
    cache_prefix: str,
):
    """
    Option B pipeline:
      1) instantiate RAW dataset from split_cfg
      2) build callable transform from transform_cfg
      3) wrap raw dataset with TransformDataset
      4) optionally wrap with CacheDataset (post-transform caching)
    """
    raw_ds = instantiate(split_cfg)  # IMPORTANT: no callables passed through Hydra
    tf = build_transform(transform_cfg)

    ds = TransformDataset(raw_ds, transform=tf)

    if cache_dir is not None:
        ds = CacheDataset(ds, cache_dir=cache_dir, prefix=cache_prefix)

    return ds


def make_train_val_loaders(
    dataset_cfg: DictConfig,
    transforms_cfg: DictConfig,
    task,
    data_cfg: DataConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    rank = get_rank() if is_distributed() else 0

    train_cache_dir = _get_cache_dir(dataset_cfg, "train")
    val_cache_dir = _get_cache_dir(dataset_cfg, "val")

    train_ds = build_dataset(
        split_cfg=dataset_cfg.train,
        transform_cfg=transforms_cfg.train,
        cache_dir=train_cache_dir,
        cache_prefix=f"train_r{rank}",
    )

    train_loader = make_loader(train_ds, task, data_cfg, shuffle=True)

    val_loader: Optional[DataLoader] = None
    if "val" in dataset_cfg and dataset_cfg.val is not None:
        val_ds = build_dataset(
            split_cfg=dataset_cfg.val,
            transform_cfg=transforms_cfg.val,
            cache_dir=val_cache_dir,
            cache_prefix=f"val_r{rank}",
        )
        val_loader = make_loader(val_ds, task, data_cfg, shuffle=False)

    return train_loader, val_loader



# -------------------------------------------------------
