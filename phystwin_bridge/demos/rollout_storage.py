#!/usr/bin/env python3
"""On-disk rollout storage for long bridge demo histories."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RolloutStorage:
    """Storage backend for large rollout arrays.

    Args:
        out_dir: Output directory for any on-disk arrays.
        prefix: Prefix used when naming storage files.
        mode: Storage mode. ``"memory"`` uses in-memory ndarrays.
            ``"memmap"`` uses ``.npy`` memory-mapped arrays on disk.
    """

    out_dir: Path
    prefix: str
    mode: str = "memmap"
    files: dict[str, str] = field(default_factory=dict)
    """Mapping from logical array name to on-disk file path."""

    def allocate(self, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        """Allocate an array either in RAM or as an on-disk memmap."""
        if self.mode == "memory":
            return np.empty(shape, dtype=dtype)
        if self.mode != "memmap":
            raise ValueError(f"Unsupported rollout storage mode: {self.mode!r}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.prefix}_{name}.npy"
        self.files[name] = str(path)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    def summary_dict(self) -> dict[str, Any]:
        """Return serializable metadata for summaries."""
        return {
            "history_storage_mode": self.mode,
            "history_storage_files": dict(self.files),
        }
