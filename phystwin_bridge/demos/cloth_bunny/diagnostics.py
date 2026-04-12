#!/usr/bin/env python3
"""Public diagnostic surface for cloth+bunny force-analysis workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_force_dump_dir(args) -> Path:
    from .offline import _resolve_force_dump_dir

    return _resolve_force_dump_dir(args)


def render_force_artifacts_subprocess(**kwargs):
    from .offline import _render_force_artifacts_subprocess

    return _render_force_artifacts_subprocess(**kwargs)


def write_force_render_bundle(**kwargs) -> Path:
    from .offline import _write_force_render_bundle

    return _write_force_render_bundle(**kwargs)


def capture_force_snapshot_from_explicit_state(**kwargs) -> dict[str, Any]:
    from .offline import _capture_force_snapshot_from_explicit_state

    return _capture_force_snapshot_from_explicit_state(**kwargs)


def make_explicit_force_snapshot_context(**kwargs) -> dict[str, Any]:
    from .offline import _make_explicit_force_snapshot_context

    return _make_explicit_force_snapshot_context(**kwargs)


def build_full_process_force_sequence_from_rollout(**kwargs):
    from .offline import _build_full_process_force_sequence_from_rollout

    return _build_full_process_force_sequence_from_rollout(**kwargs)


def finalize_force_diagnostic_artifacts(*args, **kwargs) -> None:
    from .offline import _finalize_force_diagnostic_artifacts

    _finalize_force_diagnostic_artifacts(*args, **kwargs)


__all__ = [
    "build_full_process_force_sequence_from_rollout",
    "capture_force_snapshot_from_explicit_state",
    "finalize_force_diagnostic_artifacts",
    "make_explicit_force_snapshot_context",
    "render_force_artifacts_subprocess",
    "resolve_force_dump_dir",
    "write_force_render_bundle",
]
