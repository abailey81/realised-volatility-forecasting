"""
Configuration loading, path resolution, and logging utilities.

Every script in this repository starts by calling :func:`load_config`, which
reads ``config/config.yaml`` and returns a nested ``Config`` object that
supports both attribute-style and dictionary-style access.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the project root directory (the parent of ``src``)."""
    return Path(__file__).resolve().parent.parent


def resolve(path_like: str | Path) -> Path:
    """Resolve a path string relative to the project root.

    Absolute paths are returned unchanged.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (project_root() / p).resolve()


# ---------------------------------------------------------------------------
# Config container
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    """A dict that also exposes its keys as attributes.

    Nested dicts inside a YAML file become nested ``_AttrDict`` instances so
    accessing ``config.data.split.train_frac`` works.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        for k, v in data.items():
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return _AttrDict(value)
        if isinstance(value, list):
            return [_AttrDict._wrap(x) for x in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def load_config(path: str | Path | None = None) -> _AttrDict:
    """Load the YAML configuration into an attribute-accessible dict.

    Parameters
    ----------
    path
        Path to the YAML file. Defaults to ``config/config.yaml``.
    """
    if path is None:
        path = project_root() / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return _AttrDict(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed Python, numpy and (if installed) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured stream logger.

    Repeated calls return the same logger; idempotent handler setup.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass(frozen=True)
class RunContext:
    """Bundle of common run-time objects passed around the pipeline."""
    config: _AttrDict
    seed: int
    logger: logging.Logger
