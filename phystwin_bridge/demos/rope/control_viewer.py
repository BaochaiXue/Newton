#!/usr/bin/env python3
"""Canonical rope-family path for the controller realtime viewer.

Implementation still lives in the legacy top-level module during this cleanup
pass. This shim establishes the package path without changing behavior.
"""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(demos_dir))
    from rope._legacy_entrypoint import load_legacy_demo
else:
    from ._legacy_entrypoint import load_legacy_demo

_legacy = load_legacy_demo("demo_rope_control_realtime_viewer")

create_parser = _legacy.create_parser
NewtonRopeControlViewer = _legacy.NewtonRopeControlViewer
main = _legacy.main

__all__ = ["NewtonRopeControlViewer", "create_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
