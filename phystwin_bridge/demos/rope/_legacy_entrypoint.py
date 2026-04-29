#!/usr/bin/env python3
"""Utilities for transitional rope-family entrypoint shims."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_legacy_demo(module_name: str) -> ModuleType:
    """Import a legacy top-level demo module from the demos directory."""

    demos_dir = Path(__file__).resolve().parents[1]
    demos_dir_text = str(demos_dir)
    if demos_dir_text not in sys.path:
        sys.path.insert(0, demos_dir_text)
    return importlib.import_module(module_name)


__all__ = ["load_legacy_demo"]
