#!/usr/bin/env python3
"""Canonical rope-family path for the rope-vs-bunny offline demo."""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(demos_dir))
    from rope._legacy_entrypoint import load_legacy_demo
else:
    from ._legacy_entrypoint import load_legacy_demo

_legacy = load_legacy_demo("demo_rope_bunny_drop")

parse_args = _legacy.parse_args
build_model = _legacy.build_model
simulate = _legacy.simulate
render_video = _legacy.render_video
build_summary = _legacy.build_summary
main = _legacy.main

__all__ = [
    "build_model",
    "build_summary",
    "main",
    "parse_args",
    "render_video",
    "simulate",
]


if __name__ == "__main__":
    raise SystemExit(main())
