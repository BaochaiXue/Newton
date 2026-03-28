#!/usr/bin/env python3
"""Shared runtime bootstrap for PhysTwin bridge demos.

This module centralizes the demo-side environment setup that used to be copied
across many scripts:

- resolve bridge/core/newton paths relative to the repository layout
- add the required roots to :mod:`sys.path`
- load bridge core helpers from ``tools/core``
- expose stable demo-side handles for ``newton``, ``newton_import_ir``, and
  ``path_defaults``

The intent is to keep path bootstrapping and dynamic module loading in one
place instead of repeating slightly different variants in every demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

from bridge_shared import BRIDGE_ROOT, CORE_DIR, load_core_module

WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
THIS_DIR = Path(__file__).resolve().parent
NEWTON_PY_ROOT = WORKSPACE_ROOT / "Newton" / "newton"


def ensure_bridge_runtime_paths() -> None:
    """Ensure the bridge core path and Newton Python root are importable."""
    for path in (CORE_DIR, NEWTON_PY_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_bridge_runtime_paths()

path_defaults = load_core_module(
    "phystwin_bridge_path_defaults",
    CORE_DIR / "path_defaults.py",
)
newton_import_ir = load_core_module(
    "phystwin_bridge_newton_import_ir",
    CORE_DIR / "newton_import_ir.py",
)

import newton  # noqa: E402

__all__ = [
    "BRIDGE_ROOT",
    "CORE_DIR",
    "NEWTON_PY_ROOT",
    "THIS_DIR",
    "WORKSPACE_ROOT",
    "ensure_bridge_runtime_paths",
    "newton",
    "newton_import_ir",
    "path_defaults",
]
