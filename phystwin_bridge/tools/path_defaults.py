#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        candidates.append(Path(env_path))

    candidates.extend(
        [
            root / "phystwin_bridge" / "inputs" / "overlays",
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

