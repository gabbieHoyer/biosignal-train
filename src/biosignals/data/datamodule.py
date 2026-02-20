# src/biosignals/data/datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from biosignals.data.datasets.wrappers import CacheDataset, TransformDataset
from biosignals.data.transforms.factory import build_transform
from biosignals.utils.distributed import get_rank, is_distributed
from biosignals.utils.reproducibility import worker_init_fn


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


def _get_transform_cfg(transforms_cfg: DictConfig, split: str):
    """
    Transforms convention:
      - train uses transforms.train
      - val uses transforms.val
      - test uses transforms.test if present else transforms.val
    """
    if split in transforms_cfg and transforms_cfg.get(split) is not None:
        return transforms_cfg.get(split)
    if split != "train" and "val" in transforms_cfg:
        return transforms_cfg.val
    if "train" in transforms_cfg:
        return transforms_cfg.train
    return transforms_cfg


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


def make_split_loader(
    *,
    dataset_cfg: DictConfig,
    transforms_cfg: DictConfig,
    task,
    data_cfg: DataConfig,
    split: str,
) -> DataLoader:
    """
    Build one loader for a given split: train | val | test
    """
    if split not in dataset_cfg or dataset_cfg.get(split) is None:
        raise KeyError(
            f"dataset config has no split '{split}'. Available: {list(dataset_cfg.keys())}"
        )

    rank = get_rank() if is_distributed() else 0
    cache_dir = _get_cache_dir(dataset_cfg, split)
    tf_cfg = _get_transform_cfg(transforms_cfg, split)

    ds = build_dataset(
        split_cfg=dataset_cfg.get(split),
        transform_cfg=tf_cfg,
        cache_dir=cache_dir,
        cache_prefix=f"{split}_r{rank}",
    )
    return make_loader(ds, task, data_cfg, shuffle=(split == "train"))


def make_train_val_loaders(
    dataset_cfg: DictConfig,
    transforms_cfg: DictConfig,
    task,
    data_cfg: DataConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = make_split_loader(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        task=task,
        data_cfg=data_cfg,
        split="train",
    )

    val_loader: Optional[DataLoader] = None
    if "val" in dataset_cfg and dataset_cfg.val is not None:
        val_loader = make_split_loader(
            dataset_cfg=dataset_cfg,
            transforms_cfg=transforms_cfg,
            task=task,
            data_cfg=data_cfg,
            split="val",
        )

    return train_loader, val_loader


def make_train_val_test_loaders(
    dataset_cfg: DictConfig,
    transforms_cfg: DictConfig,
    task,
    data_cfg: DataConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_loader, val_loader = make_train_val_loaders(dataset_cfg, transforms_cfg, task, data_cfg)

    test_loader: Optional[DataLoader] = None
    if "test" in dataset_cfg and dataset_cfg.test is not None:
        test_loader = make_split_loader(
            dataset_cfg=dataset_cfg,
            transforms_cfg=transforms_cfg,
            task=task,
            data_cfg=data_cfg,
            split="test",
        )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
