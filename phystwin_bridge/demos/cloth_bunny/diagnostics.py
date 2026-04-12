#!/usr/bin/env python3
"""Public diagnostic surface for cloth+bunny force-analysis workflows."""
from __future__ import annotations

import pickle
from pathlib import Path
import subprocess
import sys
from typing import Any


def resolve_force_dump_dir(args) -> Path:
    """Resolve the canonical force-diagnostic artifact directory."""

    if args.force_dump_dir is not None:
        return args.force_dump_dir.expanduser().resolve()
    return (args.out_dir / "force_diagnostic").resolve()


def render_force_artifacts_subprocess(**kwargs):
    """Serialize a force-artifact bundle and render it in a subprocess."""

    args = kwargs["args"]
    force_dir = resolve_force_dump_dir(args)
    force_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = force_dir / "force_render_bundle.pkl"
    bundle = {
        "args": args,
        "device": kwargs["device"],
        "ir_obj": kwargs["ir_obj"],
        "diag_snapshot": kwargs["diag_snapshot"],
        "sequence_snapshots": kwargs["sequence_snapshots"],
        "render_sim_data": kwargs.get("render_sim_data"),
        "video_mp4_path": "" if kwargs.get("video_mp4_path") is None else str(kwargs["video_mp4_path"]),
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    helper = Path(__file__).resolve().parents[4] / "scripts" / "render_bunny_force_artifacts.py"
    cmd = [sys.executable, str(helper), "--bundle", str(bundle_path)]
    subprocess.run(cmd, check=True)

    snapshot_png_path = force_dir / "force_diag_trigger_snapshot.png"
    snapshot_mp4_path = force_dir / "force_diag_trigger_snapshot.mp4"
    sequence_mp4_path = force_dir / "force_diag_trigger_window.mp4"
    return (
        snapshot_png_path,
        snapshot_mp4_path if snapshot_mp4_path.exists() else None,
        sequence_mp4_path if sequence_mp4_path.exists() else None,
    )


def write_force_render_bundle(**kwargs) -> Path:
    """Write a serialized force-render bundle without launching rendering."""

    args = kwargs["args"]
    force_dir = resolve_force_dump_dir(args)
    force_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = force_dir / "force_render_bundle.pkl"
    bundle = {
        "args": args,
        "device": kwargs["device"],
        "ir_obj": kwargs["ir_obj"],
        "diag_snapshot": kwargs["diag_snapshot"],
        "sequence_snapshots": [],
        "render_sim_data": kwargs["render_sim_data"],
        "trigger_substep_global": int(kwargs["trigger_substep_global"]),
        "summary_json_path": str(kwargs["summary_json_path"]),
        "video_mp4_path": "" if kwargs.get("video_mp4_path") is None else str(kwargs["video_mp4_path"]),
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return bundle_path


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
