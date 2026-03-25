#!/usr/bin/env python3
"""Compare native Newton rollouts before/after scaling total object mass.

This demo replays the same PhysTwin controller trajectory twice for each case:

1. baseline: load the original PhysTwin IR into Newton with no mass scaling
2. scaled: uniformly scale object-particle mass so total deformable mass equals
   ``--target-total-mass``

By default, the same scale factor is also applied to ``spring_ke`` and
``spring_kd`` through the importer CLI, matching the unified "weight_scale"
semantics already used in the bridge demos.

Outputs per case:
- baseline rollout `.npz` + `.json`
- scaled rollout `.npz` + `.json`
- a labeled 1x3 overlay MP4
- an optional GIF for slides
- a compact comparison summary JSON
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
OTHER_DIR = BRIDGE_ROOT / "tools" / "other"
IMPORTER_PATH = CORE_DIR / "newton_import_ir.py"
RENDER_OVERLAY_PATH = OTHER_DIR / "render_overlay_1x3_diff_mp4.py"
PATH_DEFAULTS_PATH = CORE_DIR / "path_defaults.py"

path_defaults = _load_module("compare_weight_path_defaults", PATH_DEFAULTS_PATH)
render_overlay = _load_module("compare_weight_render_overlay", RENDER_OVERLAY_PATH)


@dataclass(frozen=True)
class CaseSpec:
    alias: str
    case_name: str
    ir_path: Path
    case_dir: Path


def _default_case_specs() -> dict[str, CaseSpec]:
    root = BRIDGE_ROOT
    return {
        "rope": CaseSpec(
            alias="rope",
            case_name="rope_double_hand",
            ir_path=root / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz",
            case_dir=root / "inputs" / "cases" / "rope_double_hand",
        ),
        "sloth": CaseSpec(
            alias="sloth",
            case_name="double_lift_sloth_normal",
            ir_path=root / "ir" / "double_lift_sloth_normal" / "phystwin_ir_v2_bf_strict.npz",
            case_dir=root / "inputs" / "cases" / "double_lift_sloth_normal",
        ),
        "zebra": CaseSpec(
            alias="zebra",
            case_name="double_lift_zebra_normal",
            ir_path=root / "ir" / "double_lift_zebra_normal" / "phystwin_ir_v2_bf_strict.npz",
            case_dir=root / "inputs" / "cases" / "double_lift_zebra_normal",
        ),
        "cloth": CaseSpec(
            alias="cloth",
            case_name="blue_cloth_double_lift_around",
            ir_path=root / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz",
            case_dir=root / "inputs" / "cases" / "blue_cloth_double_lift_around",
        ),
    }


def parse_args() -> argparse.Namespace:
    specs = _default_case_specs()
    p = argparse.ArgumentParser(
        description=(
            "Compare Newton loading the same PhysTwin trajectory with no scale vs "
            "total object mass scaled to a target value."
        )
    )
    p.add_argument(
        "--cases",
        default="all",
        help="Comma-separated aliases from {rope,sloth,zebra,cloth}, or 'all'.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--target-total-mass",
        type=float,
        default=1.0,
        help="Target deformable total object mass in kg for the scaled run.",
    )
    p.add_argument(
        "--mass-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale object mass only. By default the same factor also scales spring_ke "
            "and spring_kd to match existing unified weight_scale demos."
        ),
    )
    p.add_argument("--python", default=path_defaults.default_python())
    p.add_argument("--device", default=path_defaults.default_device())
    p.add_argument("--camera-indices", default="0,1,2")
    p.add_argument("--point-radius", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--substeps-per-frame", type=int, default=None)
    p.add_argument(
        "--ground-restitution-mode",
        choices=["strict-native", "approximate-native"],
        default="approximate-native",
    )
    p.add_argument("--baseline-color", default="0,255,255")
    p.add_argument("--scaled-color", default="255,180,0")
    p.add_argument(
        "--make-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also convert the final comparison MP4 into a slide-friendly GIF.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing case outputs.",
    )
    args = p.parse_args()

    if args.target_total_mass <= 0.0:
        raise ValueError(f"--target-total-mass must be > 0, got {args.target_total_mass}")
    requested = [item.strip().lower() for item in str(args.cases).split(",") if item.strip()]
    if not requested:
        raise ValueError("--cases resolved to an empty list.")
    if requested == ["all"]:
        args.case_aliases = list(specs.keys())
    else:
        unknown = [item for item in requested if item not in specs]
        if unknown:
            raise ValueError(f"Unknown case alias(es): {unknown}. Known cases: {sorted(specs)}")
        args.case_aliases = requested
    return args


def _append_bool_flag(cmd: list[str], flag_name: str, enabled: bool | None) -> None:
    if enabled is None:
        return
    cmd.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def _mass_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _load_inference_length(case_dir: Path) -> int:
    inference_path = case_dir / "inference.pkl"
    with inference_path.open("rb") as handle:
        arr = np.asarray(pickle.load(handle), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected inference shape in {inference_path}: {arr.shape}")
    return int(arr.shape[0])


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing IR npz: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _copy_ir_dict(ir: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in ir.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        else:
            copied[key] = value
    return copied


def _build_scaled_ir(raw_ir: dict[str, Any], *, target_total_mass: float) -> tuple[dict[str, Any], float, float]:
    ir_scaled = _copy_ir_dict(raw_ir)
    n_obj = int(np.asarray(ir_scaled["num_object_points"]).ravel()[0])
    mass = np.asarray(ir_scaled["mass"], dtype=np.float32).copy()
    if mass.shape[0] < n_obj:
        raise ValueError(f"IR mass length {mass.shape[0]} < num_object_points {n_obj}")
    original_total_mass = float(mass[:n_obj].sum())
    if original_total_mass <= 0.0:
        raise ValueError(f"Original total object mass must be > 0, got {original_total_mass}")
    scale = float(target_total_mass) / original_total_mass
    mass[:n_obj] *= np.float32(scale)
    ir_scaled["mass"] = mass
    return ir_scaled, scale, original_total_mass


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _verify_same_initial_particle_state(
    raw_ir: dict[str, Any], scaled_ir: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    x0_raw = np.asarray(raw_ir["x0"], dtype=np.float32)
    x0_scaled = np.asarray(scaled_ir["x0"], dtype=np.float32)
    checks["x0_shape"] = list(x0_raw.shape)
    checks["x0_identical"] = bool(np.array_equal(x0_raw, x0_scaled))
    checks["x0_max_abs_diff"] = _max_abs_diff(x0_raw, x0_scaled)
    if not checks["x0_identical"]:
        raise ValueError(
            "Scaled IR changed `x0`. All particles, including controllers, must keep the same initial positions."
        )

    v0_raw = np.asarray(raw_ir["v0"], dtype=np.float32)
    v0_scaled = np.asarray(scaled_ir["v0"], dtype=np.float32)
    checks["v0_shape"] = list(v0_raw.shape)
    checks["v0_identical"] = bool(np.array_equal(v0_raw, v0_scaled))
    checks["v0_max_abs_diff"] = _max_abs_diff(v0_raw, v0_scaled)
    if not checks["v0_identical"]:
        raise ValueError("Scaled IR changed `v0`. This compare requires the same initial particle velocities.")

    ctrl_idx_raw = np.asarray(raw_ir.get("controller_idx", np.zeros((0,), dtype=np.int32)))
    ctrl_idx_scaled = np.asarray(scaled_ir.get("controller_idx", np.zeros((0,), dtype=np.int32)))
    checks["controller_idx_identical"] = bool(np.array_equal(ctrl_idx_raw, ctrl_idx_scaled))
    if not checks["controller_idx_identical"]:
        raise ValueError("Scaled IR changed `controller_idx`. Control-point identity must stay fixed.")

    ctrl_traj_raw = np.asarray(raw_ir.get("controller_traj", np.zeros((0, 0, 3), dtype=np.float32)))
    ctrl_traj_scaled = np.asarray(
        scaled_ir.get("controller_traj", np.zeros((0, 0, 3), dtype=np.float32))
    )
    checks["controller_traj_shape"] = list(ctrl_traj_raw.shape)
    checks["controller_traj_identical"] = bool(np.array_equal(ctrl_traj_raw, ctrl_traj_scaled))
    checks["controller_traj_max_abs_diff"] = _max_abs_diff(ctrl_traj_raw, ctrl_traj_scaled)
    if not checks["controller_traj_identical"]:
        raise ValueError(
            "Scaled IR changed `controller_traj`. This compare requires the exact same control trajectory."
        )

    n_obj = int(np.asarray(raw_ir["num_object_points"]).ravel()[0])
    checks["num_object_points"] = n_obj
    checks["num_total_particles"] = int(x0_raw.shape[0])
    checks["num_controller_particles"] = int(x0_raw.shape[0] - n_obj)
    return checks


def _save_ir(path: Path, ir: dict[str, Any]) -> None:
    serializable: dict[str, Any] = {}
    for key, value in ir.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value
        else:
            serializable[key] = np.asarray(value)
    np.savez_compressed(path, **serializable)


def _run_importer(
    *,
    python_exec: str,
    ir_path: Path,
    out_dir: Path,
    prefix: str,
    spring_scale: float,
    num_frames: int,
    device: str,
    sim_dt: float | None,
    substeps_per_frame: int | None,
    ground_restitution_mode: str,
) -> tuple[Path, Path]:
    cmd = [
        python_exec,
        str(IMPORTER_PATH),
        "--ir",
        str(ir_path),
        "--out-dir",
        str(out_dir),
        "--output-prefix",
        prefix,
        "--spring-ke-scale",
        str(float(spring_scale)),
        "--spring-kd-scale",
        str(float(spring_scale)),
        "--angular-damping",
        "0.05",
        "--friction-smoothing",
        "1.0",
        "--num-frames",
        str(int(num_frames)),
        "--device",
        device,
        "--up-axis",
        "Z",
        "--particle-contact-radius",
        "1e-5",
        "--particle-contact-kf-scale",
        "1.0",
        "--gravity-mag",
        "9.8",
        "--ground-mu-scale",
        "1.0",
        "--ground-restitution-scale",
        "1.0",
        "--ground-restitution-mode",
        str(ground_restitution_mode),
        "--drag-damping-scale",
        "1.0",
        "--shape-contact-damping-multiplier",
        "1.0",
    ]
    if sim_dt is not None:
        cmd.extend(["--sim-dt", str(float(sim_dt))])
    if substeps_per_frame is not None:
        cmd.extend(["--substeps-per-frame", str(int(substeps_per_frame))])
    _append_bool_flag(cmd, "enable-tri-contact", True)
    _append_bool_flag(cmd, "disable-particle-contact-kernel", True)
    _append_bool_flag(cmd, "gravity-from-reverse-z", True)
    _append_bool_flag(cmd, "shape-contacts", False)
    _append_bool_flag(cmd, "add-ground-plane", True)
    _append_bool_flag(cmd, "particle-contacts", False)
    _append_bool_flag(cmd, "interpolate-controls", True)
    _append_bool_flag(cmd, "strict-physics-checks", True)
    _append_bool_flag(cmd, "apply-drag", True)
    subprocess.run(cmd, check=True)
    return out_dir / f"{prefix}.npz", out_dir / f"{prefix}.json"


def _curve_stats(curve: np.ndarray) -> dict[str, float | int | None]:
    if curve.size == 0:
        return {
            "compared_frames": 0,
            "mean": None,
            "max": None,
            "first30": None,
            "last30": None,
        }
    return {
        "compared_frames": int(curve.shape[0]),
        "mean": float(np.mean(curve)),
        "max": float(np.max(curve)),
        "first30": float(np.mean(curve[: min(30, curve.shape[0])])),
        "last30": float(np.mean(curve[max(0, curve.shape[0] - 30) :])),
    }


def _baseline_vs_scaled_rmse(
    baseline_npz: Path, scaled_npz: Path
) -> dict[str, float | int | None]:
    baseline = render_overlay._load_rollout_positions(baseline_npz)
    scaled = render_overlay._load_rollout_positions(scaled_npz)
    compared_frames = min(baseline.shape[0], scaled.shape[0])
    compared_points = min(baseline.shape[1], scaled.shape[1])
    if compared_frames <= 0 or compared_points <= 0:
        return _curve_stats(np.zeros(0, dtype=np.float32))
    diff = baseline[:compared_frames, :compared_points] - scaled[:compared_frames, :compared_points]
    curve = np.sqrt(np.mean(diff**2, axis=(1, 2))).astype(np.float32)
    stats = _curve_stats(curve)
    stats["compared_points"] = int(compared_points)
    return stats


def _encode_mp4_quiet(frames_dir: Path, fps: float, out_mp4: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{float(fps)}",
            "-start_number",
            "0",
            "-i",
            str(frames_dir / "%d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_mp4),
        ],
        check=True,
    )


def _compose_1x3_with_labels_quiet(
    *,
    panel_mp4s: list[Path],
    camera_indices: list[int],
    label_text: str,
    out_mp4: Path,
) -> None:
    labels = [
        f"{label_text} | View {camera_indices[0]}",
        f"{label_text} | View {camera_indices[1]}",
        f"{label_text} | View {camera_indices[2]}",
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
    filters.append("[v0][v1][v2]xstack=inputs=3:layout=0_0|w0_0|w0+w1_0[vout]")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(panel_mp4s[0]),
            "-i",
            str(panel_mp4s[1]),
            "-i",
            str(panel_mp4s[2]),
            "-filter_complex",
            ";".join(filters),
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
        ],
        check=True,
    )


def _render_overlay_compare_mp4(
    *,
    case_dir: Path,
    baseline_npz: Path,
    scaled_npz: Path,
    out_mp4: Path,
    camera_indices: list[int],
    baseline_color: np.ndarray,
    scaled_color: np.ndarray,
    baseline_label: str,
    scaled_label: str,
    point_radius: int,
    max_frames: int | None,
) -> dict[str, Any]:
    baseline_rollout = render_overlay._load_rollout_positions(baseline_npz)
    scaled_rollout = render_overlay._load_rollout_positions(scaled_npz)
    frames_to_render = min(baseline_rollout.shape[0], scaled_rollout.shape[0])
    if max_frames is not None:
        frames_to_render = min(frames_to_render, int(max_frames))
    points_to_render = min(baseline_rollout.shape[1], scaled_rollout.shape[1])
    if frames_to_render <= 0 or points_to_render <= 0:
        raise ValueError("No shared frames/points available for overlay render.")
    baseline_rollout = baseline_rollout[:frames_to_render, :points_to_render].astype(np.float32)
    scaled_rollout = scaled_rollout[:frames_to_render, :points_to_render].astype(np.float32)

    metadata = render_overlay._load_metadata(case_dir)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    wh = metadata["WH"]
    width = int(wh[0])
    height = int(wh[1])
    fps = float(metadata.get("fps", 30.0))
    w2cs = render_overlay._load_w2c(case_dir)

    tmp_dir = out_mp4.parent / f"._tmp_compare_overlay_{out_mp4.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    baseline_color_name = render_overlay._rgb_to_color_name(baseline_color)
    scaled_color_name = render_overlay._rgb_to_color_name(scaled_color)
    label_text = f"{baseline_color_name}={baseline_label}, {scaled_color_name}={scaled_label}"
    panel_mp4s: list[Path] = []

    try:
        for cam in camera_indices:
            K = intrinsics[cam]
            w2c = w2cs[cam]
            sample_points = np.concatenate([baseline_rollout[0], scaled_rollout[0]], axis=0)
            y_sign, z_sign = render_overlay._choose_projection_convention(
                sample_points_world=sample_points,
                w2c=w2c,
                K=K,
                width=width,
                height=height,
            )

            panel_frames = tmp_dir / f"frames_cam{cam}"
            panel_frames.mkdir(parents=True, exist_ok=False)
            panel_mp4 = tmp_dir / f"overlay_cam{cam}.mp4"
            panel_mp4s.append(panel_mp4)

            for frame_idx in range(frames_to_render):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                u_b, v_b, z_b = render_overlay._project_points(
                    points_world=baseline_rollout[frame_idx],
                    w2c=w2c,
                    K=K,
                    y_sign=y_sign,
                    z_sign=z_sign,
                    z_eps=1.0e-6,
                )
                u_s, v_s, z_s = render_overlay._project_points(
                    points_world=scaled_rollout[frame_idx],
                    w2c=w2c,
                    K=K,
                    y_sign=y_sign,
                    z_sign=z_sign,
                    z_eps=1.0e-6,
                )
                if u_b.size > 0 and u_s.size > 0:
                    u = np.concatenate([u_b, u_s], axis=0)
                    v = np.concatenate([v_b, v_s], axis=0)
                    z = np.concatenate([z_b, z_s], axis=0)
                    colors_u8 = np.concatenate(
                        [
                            np.tile(baseline_color[None, :], (u_b.shape[0], 1)),
                            np.tile(scaled_color[None, :], (u_s.shape[0], 1)),
                        ],
                        axis=0,
                    )
                elif u_b.size > 0:
                    u, v, z = u_b, v_b, z_b
                    colors_u8 = np.tile(baseline_color[None, :], (u_b.shape[0], 1))
                elif u_s.size > 0:
                    u, v, z = u_s, v_s, z_s
                    colors_u8 = np.tile(scaled_color[None, :], (u_s.shape[0], 1))
                else:
                    u = np.zeros((0,), dtype=np.float64)
                    v = np.zeros((0,), dtype=np.float64)
                    z = np.zeros((0,), dtype=np.float64)
                    colors_u8 = np.zeros((0, 3), dtype=np.uint8)

                render_overlay._draw_points(
                    image=image,
                    u=u,
                    v=v,
                    z=z,
                    colors_u8=colors_u8.astype(np.uint8, copy=False),
                    radius=int(point_radius),
                )
                Image.fromarray(image).save(panel_frames / f"{frame_idx}.png")

            _encode_mp4_quiet(frames_dir=panel_frames, fps=fps, out_mp4=panel_mp4)

        _compose_1x3_with_labels_quiet(
            panel_mp4s=panel_mp4s,
            camera_indices=camera_indices,
            label_text=label_text,
            out_mp4=out_mp4,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "frames_to_render": int(frames_to_render),
        "points_to_render": int(points_to_render),
        "camera_indices": [int(v) for v in camera_indices],
        "fps": float(fps),
        "label_text": label_text,
    }


def _mp4_to_gif(mp4_path: Path, gif_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    for fps, width, colors in ((10, 960, 192), (8, 800, 128), (6, 720, 96)):
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
            f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4_path),
                "-vf",
                vf,
                "-loop",
                "0",
                str(gif_path),
            ],
            check=True,
        )
        if gif_path.exists() and gif_path.stat().st_size <= 45 * 1024 * 1024:
            return
    raise RuntimeError(f"Failed to compress {mp4_path} into a GIF under 45 MB")


def _case_outputs_exist(case_out_dir: Path, case_alias: str, target_total_mass: float) -> bool:
    mass_tag = _mass_tag(target_total_mass)
    required = [
        case_out_dir / f"{case_alias}_baseline.json",
        case_out_dir / f"{case_alias}_scaled_total{mass_tag}kg.json",
        case_out_dir / f"{case_alias}_baseline_vs_total{mass_tag}kg_overlay_1x3.mp4",
        case_out_dir / f"{case_alias}_comparison_summary.json",
    ]
    return all(path.exists() for path in required)


def _pick_case_specs(case_aliases: list[str]) -> list[CaseSpec]:
    specs = _default_case_specs()
    picked = [specs[alias] for alias in case_aliases]
    for spec in picked:
        if not spec.ir_path.exists():
            raise FileNotFoundError(f"Missing IR for case '{spec.alias}': {spec.ir_path}")
        if not spec.case_dir.exists():
            raise FileNotFoundError(f"Missing case dir for case '{spec.alias}': {spec.case_dir}")
    return picked


def main() -> int:
    args = parse_args()
    specs = _pick_case_specs(args.case_aliases)
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    camera_indices = render_overlay._parse_camera_indices(args.camera_indices)
    baseline_color = render_overlay._parse_rgb(args.baseline_color)
    scaled_color = render_overlay._parse_rgb(args.scaled_color)

    overall_summary: dict[str, Any] = {
        "target_total_mass_kg": float(args.target_total_mass),
        "mass_only": bool(args.mass_only),
        "cases": [],
    }

    for spec in specs:
        case_out_dir = out_root / spec.alias
        case_out_dir.mkdir(parents=True, exist_ok=True)
        mass_tag = _mass_tag(args.target_total_mass)

        if _case_outputs_exist(case_out_dir, spec.alias, args.target_total_mass) and not args.overwrite:
            print(f"[skip] {spec.alias}: outputs already exist at {case_out_dir}", flush=True)
            summary_path = case_out_dir / f"{spec.alias}_comparison_summary.json"
            overall_summary["cases"].append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        print(f"[case] {spec.alias}", flush=True)
        raw_ir = _load_ir(spec.ir_path.resolve())
        scaled_ir, scale, original_total_mass = _build_scaled_ir(
            raw_ir, target_total_mass=float(args.target_total_mass)
        )
        same_state_checks = _verify_same_initial_particle_state(raw_ir, scaled_ir)
        controller_frames = int(np.asarray(scaled_ir["controller_traj"]).shape[0])
        inference_frames = _load_inference_length(spec.case_dir)
        num_frames = min(controller_frames, inference_frames)
        if args.max_frames is not None:
            num_frames = min(num_frames, int(args.max_frames))
        if num_frames <= 0:
            raise ValueError(f"No frames available for case '{spec.alias}'")

        baseline_prefix = f"{spec.alias}_baseline"
        scaled_prefix = f"{spec.alias}_scaled_total{mass_tag}kg"
        scaled_ir_path = case_out_dir / f"{spec.alias}_scaled_total{mass_tag}kg_ir.npz"
        _save_ir(scaled_ir_path, scaled_ir)

        spring_scale = 1.0 if args.mass_only else scale

        baseline_npz, baseline_json = _run_importer(
            python_exec=str(args.python),
            ir_path=spec.ir_path,
            out_dir=case_out_dir,
            prefix=baseline_prefix,
            spring_scale=1.0,
            num_frames=num_frames,
            device=str(args.device),
            sim_dt=args.sim_dt,
            substeps_per_frame=args.substeps_per_frame,
            ground_restitution_mode=str(args.ground_restitution_mode),
        )
        scaled_npz, scaled_json = _run_importer(
            python_exec=str(args.python),
            ir_path=scaled_ir_path,
            out_dir=case_out_dir,
            prefix=scaled_prefix,
            spring_scale=spring_scale,
            num_frames=num_frames,
            device=str(args.device),
            sim_dt=args.sim_dt,
            substeps_per_frame=args.substeps_per_frame,
            ground_restitution_mode=str(args.ground_restitution_mode),
        )

        overlay_mp4 = case_out_dir / f"{spec.alias}_baseline_vs_total{mass_tag}kg_overlay_1x3.mp4"
        overlay_meta = _render_overlay_compare_mp4(
            case_dir=spec.case_dir,
            baseline_npz=baseline_npz,
            scaled_npz=scaled_npz,
            out_mp4=overlay_mp4,
            camera_indices=camera_indices,
            baseline_color=baseline_color,
            scaled_color=scaled_color,
            baseline_label="NoScale",
            scaled_label=f"Total{float(args.target_total_mass):g}kg",
            point_radius=int(args.point_radius),
            max_frames=args.max_frames,
        )

        overlay_gif = None
        if args.make_gif:
            overlay_gif = case_out_dir / f"{spec.alias}_baseline_vs_total{mass_tag}kg_overlay_1x3.gif"
            _mp4_to_gif(overlay_mp4, overlay_gif)

        baseline_summary = json.loads(baseline_json.read_text(encoding="utf-8"))
        scaled_summary = json.loads(scaled_json.read_text(encoding="utf-8"))
        comparison_summary = {
            "case_alias": spec.alias,
            "case_name": spec.case_name,
            "ir_path": str(spec.ir_path),
            "scaled_ir_path": str(scaled_ir_path),
            "case_dir": str(spec.case_dir),
            "original_total_object_mass_kg": float(original_total_mass),
            "target_total_object_mass_kg": float(args.target_total_mass),
            "mass_spring_scale": float(scale),
            "spring_scale_applied": float(spring_scale),
            "mass_only": bool(args.mass_only),
            "num_frames_run": int(num_frames),
            "baseline_rollout_npz": str(baseline_npz),
            "scaled_rollout_npz": str(scaled_npz),
            "overlay_mp4": str(overlay_mp4),
            "overlay_gif": str(overlay_gif) if overlay_gif is not None else None,
            "overlay_render": overlay_meta,
            "same_initial_particle_state": same_state_checks,
            "baseline_vs_scaled_rmse": _baseline_vs_scaled_rmse(baseline_npz, scaled_npz),
            "baseline_vs_phystwin": baseline_summary.get("baseline", {}),
            "scaled_vs_phystwin": scaled_summary.get("baseline", {}),
        }
        summary_path = case_out_dir / f"{spec.alias}_comparison_summary.json"
        summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
        overall_summary["cases"].append(comparison_summary)
        print(f"  overlay mp4: {overlay_mp4}", flush=True)
        if overlay_gif is not None:
            print(f"  overlay gif: {overlay_gif}", flush=True)
        print(f"  summary: {summary_path}", flush=True)

    overall_path = out_root / "comparison_summary.json"
    overall_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(f"[done] {overall_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
