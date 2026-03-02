#!/usr/bin/env python3
"""Shared path/device defaults for the PhysTwin <-> Newton bridge tools.

These scripts are frequently executed directly by file path (not as a package),
so we keep a tiny set of helpers here to:
- locate the Newton checkout root robustly (without fragile `parents[k]` indexing)
- pick a default Python interpreter (prefer Newton's `.venv`)
- choose a default Warp device string
- resolve the overlay image base directory for visualization tools
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the Newton folder that contains `newton/` and `phystwin_bridge/`.

    Note: this file may live under `phystwin_bridge/tools/core/`, so avoid
    hard-coded `parents[k]` indexing.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "newton").is_dir() and (parent / "phystwin_bridge").is_dir():
            return parent
    raise RuntimeError(
        "Failed to locate Newton repo root (expected a parent directory containing "
        "`newton/` and `phystwin_bridge/`). Starting from:\n"
        f"  {here}"
    )


def bridge_root() -> Path:
    return repo_root() / "phystwin_bridge"


def default_python() -> str:
    candidate = repo_root() / "newton/.venv/bin/python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def default_device() -> str:
    return os.environ.get("NEWTON_DEVICE", "cuda:0")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(Path(normalized))
    return deduped


def overlay_base_candidates() -> list[Path]:
    root = repo_root()
    env_path = os.environ.get("PHYSTWIN_OVERLAY_BASE")

    candidates: list[Path] = []
    if env_path:
        # Explicit override (used by pipeline scripts).
        candidates.append(Path(env_path))

    candidates.extend(
        [
            # In-repo location (if overlays were copied into Newton checkout).
            root / "phystwin_bridge" / "inputs" / "overlays",
            # Common local PhysTwin data layouts.
            root.parent / "PhysTwin" / "data" / "different_types",
            root.parent / "NewPhysTwin" / "data" / "different_types",
        ]
    )
    return _dedupe_paths(candidates)


def resolve_overlay_base(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--overlay-base does not exist: {resolved}")
        return resolved

    for candidate in overlay_base_candidates():
        if candidate.exists():
            return candidate

    checked = "\n  ".join(str(p) for p in overlay_base_candidates())
    raise FileNotFoundError(
        "Failed to auto-resolve overlay base directory.\n"
        "Provide --overlay-base explicitly or set PHYSTWIN_OVERLAY_BASE.\n"
        f"Checked candidates:\n  {checked}"
    )
