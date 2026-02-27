#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from path_defaults import default_python, resolve_overlay_base


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping at {path}")
    return loaded


def _select_best_report(case_out_dir: Path) -> Path:
    candidates = sorted(case_out_dir.glob("*_rollout_report.json"))
    if not candidates:
        candidates = sorted(case_out_dir.glob("*_parity_report.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_rollout_report.json or *_parity_report.json found under {case_out_dir}"
        )

    best_path = None
    best_score = None
    for path in candidates:
        try:
            report = _load_json(path)
        except Exception:
            continue
        passed = bool(report.get("passed", False))
        frames_run = 0
        rollout_summary = report.get("rollout_summary")
        if isinstance(rollout_summary, dict) and "frames_run" in rollout_summary:
            try:
                frames_run = int(rollout_summary["frames_run"])
            except Exception:
                frames_run = 0
        mtime = os.path.getmtime(path)
        score = (1 if passed else 0, frames_run, mtime)
        if best_score is None or score > best_score:
            best_score = score
            best_path = path

    if best_path is None:
        raise RuntimeError(f"Failed to parse any rollout report under {case_out_dir}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-render PhysTwin-style MP4 videos for Newton rollouts by selecting the "
            "best/latest rollout report per case."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("phystwin_bridge/outputs"),
        help="Directory containing per-case output subfolders.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path("phystwin_bridge/inputs/cases"),
        help="Directory containing per-case input subfolders (metadata/calibration/final_data).",
    )
    parser.add_argument(
        "--overlay-base",
        type=Path,
        default=None,
        help=(
            "Base directory containing <case_name>/color/<cam>/<frame>.png overlay frames. "
            "If unset, auto-resolve via PHYSTWIN_OVERLAY_BASE and common local paths."
        ),
    )
    parser.add_argument("--camera-idx", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    outputs_root = (
        (repo_root / args.outputs_root).resolve()
        if not args.outputs_root.is_absolute()
        else args.outputs_root.resolve()
    )
    cases_root = (
        (repo_root / args.cases_root).resolve()
        if not args.cases_root.is_absolute()
        else args.cases_root.resolve()
    )
    overlay_base = resolve_overlay_base(args.overlay_base)
    camera_idx = int(args.camera_idx)

    visualizer = Path(__file__).resolve().with_name("visualize_rollout_mp4.py")
    python = default_python()

    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs-root does not exist: {outputs_root}")

    case_dirs = sorted([path for path in outputs_root.iterdir() if path.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"No case output directories found under {outputs_root}")

    rendered = []
    for case_out_dir in case_dirs:
        case_name = case_out_dir.name
        case_dir = cases_root / case_name
        if not case_dir.exists():
            raise FileNotFoundError(f"Missing case directory for {case_name}: {case_dir}")

        report_path = _select_best_report(case_out_dir)
        report = _load_json(report_path)
        rollout_npz = report.get("rollout_npz")
        if not isinstance(rollout_npz, str) or not rollout_npz:
            raise KeyError(f"Parity report missing rollout_npz: {report_path}")

        cmd = [
            python,
            str(visualizer),
            "--case-dir",
            str(case_dir),
            "--rollout-npz",
            rollout_npz,
            "--overlay-base",
            str(overlay_base),
            "--camera-idx",
            str(camera_idx),
        ]
        subprocess.run(cmd, check=True)
        rendered.append(
            {
                "case": case_name,
                "report": str(report_path),
                "rollout_npz": rollout_npz,
            }
        )

    print(json.dumps({"rendered": rendered}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
