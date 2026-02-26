#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np

from path_defaults import default_python, resolve_overlay_base


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping at {path}")
    return loaded


def _default_visualizer() -> Path:
    return Path(__file__).resolve().with_name("visualize_rollout_mp4.py")


def _parse_camera_indices(value: str) -> list[int]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if len(items) != 3:
        raise ValueError(f"--camera-indices expects exactly 3 indices, got {value!r}")
    indices = [int(item) for item in items]
    if len(set(indices)) != 3:
        raise ValueError(f"--camera-indices must be distinct, got {indices}")
    return indices


def _load_rollout_frames(npz_path: Path) -> tuple[np.ndarray, int]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing rollout npz: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "particle_q_object" in data:
            positions = data["particle_q_object"].astype(np.float32)
        elif "particle_q_all" in data:
            positions = data["particle_q_all"].astype(np.float32)
        else:
            raise KeyError(f"{npz_path} missing particle positions")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Unexpected rollout shape in {npz_path}: {positions.shape}")
    return positions, int(positions.shape[0])


def _load_inference(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing inference.pkl: {path}")
    with path.open("rb") as handle:
        inference = pickle.load(handle)
    arr = np.asarray(inference, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected inference shape in {path}: {arr.shape}")
    return arr


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _render_panel_mp4(
    python: str,
    visualizer: Path,
    case_dir: Path,
    rollout_npz: Path,
    overlay_base: Path,
    camera_idx: int,
    max_frames: int,
    out_mp4: Path,
) -> None:
    cmd = [
        python,
        str(visualizer),
        "--case-dir",
        str(case_dir),
        "--rollout-npz",
        str(rollout_npz),
        "--overlay-base",
        str(overlay_base),
        "--camera-idx",
        str(camera_idx),
        "--max-frames",
        str(max_frames),
        "--out-mp4",
        str(out_mp4),
    ]
    _run(cmd)


def _compose_2x3_with_labels(
    newton_panels: list[Path],
    phystwin_panels: list[Path],
    camera_indices: list[int],
    newton_label: str,
    phystwin_label: str,
    out_mp4: Path,
) -> None:
    if len(newton_panels) != 3 or len(phystwin_panels) != 3:
        raise ValueError("Expected exactly 3 Newton panels and 3 PhysTwin panels.")

    inputs = [*newton_panels, *phystwin_panels]
    labels = [
        f"{newton_label} | View {camera_indices[0]}",
        f"{newton_label} | View {camera_indices[1]}",
        f"{newton_label} | View {camera_indices[2]}",
        f"{phystwin_label} | View {camera_indices[0]}",
        f"{phystwin_label} | View {camera_indices[1]}",
        f"{phystwin_label} | View {camera_indices[2]}",
    ]

    filters: list[str] = []
    for idx, text in enumerate(labels):
        safe_text = text.replace("'", "\\'")
        filters.append(
            f"[{idx}:v]drawtext=text='{safe_text}':"
            "x=20:y=20:fontsize=32:fontcolor=white:"
            "box=1:boxcolor=black@0.6:boxborderw=8[v"
            f"{idx}]"
        )
    filters.append(
        "[v0][v1][v2][v3][v4][v5]xstack=inputs=6:"
        "layout=0_0|w0_0|w0+w1_0|0_h0|w3_h0|w3+w4_h0[vout]"
    )
    filter_complex = ";".join(filters)

    cmd = ["ffmpeg", "-y"]
    for panel in inputs:
        cmd.extend(["-i", str(panel)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_mp4),
        ]
    )
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a labeled 2x3 comparison video per case: "
            "top row Newton rollout (3 camera views), bottom row PhysTwin inference rollout (3 views)."
        )
    )
    parser.add_argument("--report", type=Path, required=True, help="Parity report JSON path.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Optional case dir override.")
    parser.add_argument(
        "--inference-pkl",
        type=Path,
        default=None,
        help="Optional inference.pkl override. Defaults to value in report or case dir.",
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
    parser.add_argument("--camera-indices", default="0,1,2")
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--visualizer", type=Path, default=_default_visualizer())
    parser.add_argument("--newton-label", default="Newton Rollout")
    parser.add_argument("--phystwin-label", default="PhysTwin Rollout")
    parser.add_argument(
        "--out-mp4",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to <rollout_stem>_cmp_2x3_labeled.mp4",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report = _load_json(report_path)

    rollout_npz = Path(str(report["rollout_npz"])).resolve()
    case_name = str(report["case_name"])

    repo_root = Path(__file__).resolve().parents[2]
    case_dir = (
        args.case_dir.resolve()
        if args.case_dir is not None
        else (repo_root / "phystwin_bridge" / "inputs" / "cases" / case_name).resolve()
    )
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case-dir: {case_dir}")

    if args.inference_pkl is not None:
        inference_pkl = args.inference_pkl.resolve()
    elif isinstance(report.get("inference_path"), str) and report["inference_path"]:
        inference_pkl = Path(str(report["inference_path"])).resolve()
    else:
        inference_pkl = (case_dir / "inference.pkl").resolve()

    overlay_base = resolve_overlay_base(args.overlay_base)
    camera_indices = _parse_camera_indices(args.camera_indices)
    visualizer = args.visualizer.resolve()
    python = str(args.python)

    rollout_positions, rollout_frames = _load_rollout_frames(rollout_npz)
    inference_positions = _load_inference(inference_pkl)
    frames_to_render = min(int(rollout_frames), int(inference_positions.shape[0]))
    if frames_to_render <= 0:
        raise ValueError(
            f"No frames to render: rollout_frames={rollout_frames}, inference_frames={inference_positions.shape[0]}"
        )

    out_mp4 = (
        args.out_mp4.resolve()
        if args.out_mp4 is not None
        else rollout_npz.with_name(f"{rollout_npz.stem}_cmp_2x3_labeled.mp4")
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_mp4.parent / f"._tmp_cmp2x3_{rollout_npz.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    inference_npz = tmp_dir / f"{case_name}_inference_rollout.npz"
    np.savez_compressed(
        inference_npz,
        particle_q_object=inference_positions[:frames_to_render].astype(np.float32),
    )

    newton_panels: list[Path] = []
    phystwin_panels: list[Path] = []
    for cam in camera_indices:
        newton_mp4 = tmp_dir / f"newton_view{cam}.mp4"
        phystwin_mp4 = tmp_dir / f"phystwin_view{cam}.mp4"

        _render_panel_mp4(
            python=python,
            visualizer=visualizer,
            case_dir=case_dir,
            rollout_npz=rollout_npz,
            overlay_base=overlay_base,
            camera_idx=cam,
            max_frames=frames_to_render,
            out_mp4=newton_mp4,
        )
        _render_panel_mp4(
            python=python,
            visualizer=visualizer,
            case_dir=case_dir,
            rollout_npz=inference_npz,
            overlay_base=overlay_base,
            camera_idx=cam,
            max_frames=frames_to_render,
            out_mp4=phystwin_mp4,
        )
        newton_panels.append(newton_mp4)
        phystwin_panels.append(phystwin_mp4)

    _compose_2x3_with_labels(
        newton_panels=newton_panels,
        phystwin_panels=phystwin_panels,
        camera_indices=camera_indices,
        newton_label=str(args.newton_label),
        phystwin_label=str(args.phystwin_label),
        out_mp4=out_mp4,
    )

    if not args.keep_temp:
        shutil.rmtree(tmp_dir)

    print(
        json.dumps(
            {
                "case": case_name,
                "report": str(report_path),
                "rollout_npz": str(rollout_npz),
                "inference_pkl": str(inference_pkl),
                "frames_to_render": int(frames_to_render),
                "camera_indices": camera_indices,
                "out_mp4": str(out_mp4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
