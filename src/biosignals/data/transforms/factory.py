# src/biosignals/data/transforms/factory.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from biosignals.data.transforms.base import Identity
from biosignals.data.transforms.compose import Compose
from biosignals.data.types import Sample

Transform = Callable[[Sample], Sample]


def build_transform(spec: Any) -> Transform:
    """
    Convert a transform specification into a callable Transform.

    Supported:
      - None -> Identity()
      - callable -> returned as-is
      - list/tuple of transform configs/objects -> Compose(...)
      - OmegaConf ListConfig -> Compose(instantiated items)
      - OmegaConf DictConfig:
          * {_target_: ...} -> instantiate + validate
          * {transforms: [...]} -> Compose(instantiated items)
    """
    if spec is None:
        return Identity()

    if callable(spec):
        return spec  # type: ignore[return-value]

    # Plain python containers
    if isinstance(spec, (list, tuple)):
        items = [
            _coerce_item(x, context=f"python {type(spec).__name__}[{i}]")
            for i, x in enumerate(spec)
        ]
        _assert_all_callable(items, context=f"python {type(spec).__name__}")
        return Compose(items)

    # OmegaConf containers
    if OmegaConf.is_config(spec):
        # Common mistake: passing cfg.transforms (group) instead of cfg.transforms.train/val
        if (
            isinstance(spec, DictConfig)
            and ("train" in spec and "val" in spec)
            and ("_target_" not in spec)
            and ("transforms" not in spec)
        ):
            raise TypeError(
                "You passed the *transforms group* (has keys train/val) into build_transform.\n"
                "Use cfg.transforms.train or cfg.transforms.val (or set train_tf/val_tf separately)."
            )

        if isinstance(spec, ListConfig):
            items = [_coerce_item(x, context=f"ListConfig[{i}]") for i, x in enumerate(list(spec))]
            _assert_all_callable(items, context="ListConfig")
            return Compose(items)

        if isinstance(spec, DictConfig):
            # Shorthand style: {transforms: [...]}
            if "transforms" in spec and "_target_" not in spec:
                items_cfg = spec["transforms"]
                if not OmegaConf.is_config(items_cfg) and not isinstance(items_cfg, (list, tuple)):
                    raise TypeError(f"DictConfig.transforms must be a list; got {type(items_cfg)}")
                items = [
                    _coerce_item(x, context=f"DictConfig.transforms[{i}]")
                    for i, x in enumerate(list(items_cfg))
                ]
                _assert_all_callable(items, context="DictConfig.transforms")
                return Compose(items)

            # Explicit _target_ style (possibly Compose, possibly something else)
            if "_target_" in spec:
                target = str(spec.get("_target_"))
                # If user explicitly points at Compose, do NOT trust Hydra recursion;
                # we build Compose ourselves to guarantee callables.
                if target.endswith(".Compose") or target.endswith("Compose"):
                    items_cfg = spec.get("transforms", [])
                    items = [
                        _coerce_item(x, context=f"Compose.transforms[{i}]")
                        for i, x in enumerate(list(items_cfg))
                    ]
                    _assert_all_callable(items, context="Compose.transforms")
                    return Compose(items)

                inst = instantiate(spec, _recursive_=True)
                return _ensure_transform_callable(inst, context=f"DictConfig _target_={target}")

            keys = list(spec.keys())
            raise TypeError(
                "Unsupported DictConfig transform spec.\n"
                f"Keys={keys}\n"
                "Expected either:\n"
                "  - a node with '_target_'\n"
                "  - or a shorthand node with 'transforms: [...]'\n"
            )

    raise TypeError(f"Unsupported transform spec type: {type(spec)}")


def _coerce_item(x: Any, *, context: str) -> Transform:
    # Already callable
    if callable(x):
        return x  # type: ignore[return-value]

    # OmegaConf node -> instantiate single item
    if OmegaConf.is_config(x):
        if isinstance(x, DictConfig) and "_target_" not in x:
            raise TypeError(
                f"{context}: transform item is a DictConfig without _target_. " f"Value={x}"
            )
        inst = instantiate(x, _recursive_=True)
        if OmegaConf.is_config(inst):
            raise TypeError(
                f"{context}: instantiate() returned an OmegaConf node instead of an object. "
                f"type={type(inst)} value={inst}"
            )
        if not callable(inst):
            raise TypeError(
                f"{context}: instantiated item is not callable. type={type(inst)} value={inst}"
            )
        return inst  # type: ignore[return-value]

    # Anything else is invalid
    raise TypeError(
        f"{context}: transform item is neither callable nor OmegaConf config. type={type(x)} value={x}"
    )


def _ensure_transform_callable(inst: Any, *, context: str) -> Transform:
    if inst is None:
        return Identity()
    if callable(inst):
        return inst  # type: ignore[return-value]
    if isinstance(inst, (list, tuple, ListConfig)):
        items = list(inst)
        _assert_all_callable(items, context=f"{context} (list returned)")
        return Compose(items)  # type: ignore[arg-type]
    raise TypeError(
        f"{context}: instantiated object is not callable. type={type(inst)} value={inst}"
    )


def _assert_all_callable(items: Sequence[Any], *, context: str) -> None:
    for i, t in enumerate(items):
        if not callable(t):
            raise TypeError(f"{context}: item[{i}] is not callable. type={type(t)} value={t}")


# --------------------------------------------------------
