#!/usr/bin/env python3
"""Run a frame-restart 2x2 self-collision / ground-contact continuation matrix.

This runner creates a derived strict IR whose frame-0 state is taken from the
PhysTwin reference rollout at `start_frame`, then reruns the canonical four
law combinations until the original rollout end:

- self-collision: off | phystwin
- ground-contact: native | phystwin

The result surface stays bridge-side only and preserves the same cloth scene,
dt/substeps, evaluator path, and visualization conventions as the authoritative
full-rollout matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_defaults import default_device, default_python, resolve_overlay_base  # noqa: E402


VALIDATE_PARITY = CORE_DIR / "validate_parity.py"
RENDER_COMPARISON = Path(__file__).resolve().with_name("render_comparison_2x3_mp4.py")
RENDER_GIF = SCRIPTS_DIR / "render_gif.sh"
ARTIFACT_VALIDATOR = SCRIPTS_DIR / "validate_experiment_artifacts.py"
DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
DEFAULT_CASE_DIR = BRIDGE_ROOT / "inputs" / "cases" / "blue_cloth_double_lift_around"
RESULTS_ROOT = BRIDGE_ROOT / "results"

STRICT_THRESHOLDS = {
    "x0_rmse": 1.0e-6,
    "rmse_mean": 1.0e-5,
    "rmse_max": 5.0e-5,
    "first30_rmse": 5.0e-5,
    "last30_rmse": 5.0e-5,
}


@dataclass(frozen=True)
class CaseSpec:
    label: str
    short_label: str
    self_collision_law: str
    ground_contact_law: str


CASE_SPECS = (
    CaseSpec("case_1_self_off_ground_native", "C1 off/native", "off", "native"),
    CaseSpec("case_2_self_off_ground_phystwin", "C2 off/ps-ground", "off", "phystwin"),
    CaseSpec("case_3_self_phystwin_ground_native", "C3 ps-self/native", "phystwin", "native"),
    CaseSpec("case_4_self_phystwin_ground_phystwin", "C4 ps-self/ps-ground", "phystwin", "phystwin"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frame-restart continuation 2x2 RMSE matrix from a PhysTwin reference frame."
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--start-frame", type=int, default=137)
    parser.add_argument(
        "--velocity-mode",
        choices=("central", "forward", "backward"),
        default="central",
        help="Finite-difference rule used to estimate object velocity at the restart frame.",
    )
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-max-colors", type=int, default=96)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _resolve_git_sha() -> str:
    for repo in (WORKSPACE_ROOT, BRIDGE_ROOT):
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "nogit"
    return "nogit"


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / f"ground_contact_self_collision_restart137_matrix_{stamp}_{_resolve_git_sha()}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_logged(
    cmd: list[str],
    *,
    workdir: Path,
    command_path: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    _write_text(command_path, "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(cmd) + "\n")
    os.chmod(command_path, 0o775)
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(log_path, proc.stdout)
    return proc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _compute_restart_velocity(
    inference: np.ndarray,
    *,
    frame_idx: int,
    frame_dt: float,
    mode: str,
) -> tuple[np.ndarray, str]:
    if mode == "central" and 0 < frame_idx < inference.shape[0] - 1:
        vel = (inference[frame_idx + 1] - inference[frame_idx - 1]) / (2.0 * frame_dt)
        return vel.astype(np.float32, copy=False), "central_difference"
    if mode == "forward":
        if frame_idx >= inference.shape[0] - 1:
            raise ValueError("forward difference requires start_frame < last frame")
        vel = (inference[frame_idx + 1] - inference[frame_idx]) / frame_dt
        return vel.astype(np.float32, copy=False), "forward_difference"
    if mode == "backward":
        if frame_idx <= 0:
            raise ValueError("backward difference requires start_frame > 0")
        vel = (inference[frame_idx] - inference[frame_idx - 1]) / frame_dt
        return vel.astype(np.float32, copy=False), "backward_difference"
    if frame_idx == 0:
        vel = (inference[1] - inference[0]) / frame_dt
        return vel.astype(np.float32, copy=False), "forward_difference_fallback"
    vel = (inference[frame_idx] - inference[frame_idx - 1]) / frame_dt
    return vel.astype(np.float32, copy=False), "backward_difference_fallback"


def _slice_final_data(final_data: dict[str, Any], start_frame: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in final_data.items():
        arr = np.asarray(value) if isinstance(value, np.ndarray) else value
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= start_frame + 1:
            if key in {
                "controller_points",
                "object_points",
                "object_colors",
                "object_visibilities",
                "object_motions_valid",
            }:
                sliced[key] = value[start_frame:].copy()
                continue
        sliced[key] = value
    return sliced


def _build_overlay_slice(
    *,
    overlay_base: Path,
    derived_case_name: str,
    source_case_name: str,
    start_frame: int,
    num_frames: int,
    out_overlay_root: Path,
) -> None:
    for cam in (0, 1, 2):
        dest_dir = out_overlay_root / derived_case_name / "color" / str(cam)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for local_idx in range(num_frames):
            src_idx = start_frame + local_idx
            src = overlay_base / source_case_name / "color" / str(cam) / f"{src_idx}.png"
            if not src.exists():
                raise FileNotFoundError(f"Missing overlay frame: {src}")
            dst = dest_dir / f"{local_idx}.png"
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)


def _compose_3x4_case_board_video(*, case_mp4s: list[Path], out_mp4: Path) -> None:
    if len(case_mp4s) != 4:
        raise ValueError(f"Expected 4 case videos, got {len(case_mp4s)}")
    tile_w = 848
    tile_h = 480

    filters: list[str] = []
    labels = ["a", "b", "c", "d"]
    for idx, label in enumerate(labels):
        filters.append(f"[{idx}:v]crop=848:480:0:0[{label}0]")
        filters.append(f"[{idx}:v]crop=848:480:848:0[{label}1]")
        filters.append(f"[{idx}:v]crop=848:480:1696:0[{label}2]")

    filters.append(
        "[a0][b0][c0][d0][a1][b1][c1][d1][a2][b2][c2][d2]"
        "xstack=inputs=12:layout="
        f"0_0|{tile_w}_0|{tile_w*2}_0|{tile_w*3}_0|"
        f"0_{tile_h}|{tile_w}_{tile_h}|{tile_w*2}_{tile_h}|{tile_w*3}_{tile_h}|"
        f"0_{tile_h*2}|{tile_w}_{tile_h*2}|{tile_w*2}_{tile_h*2}|{tile_w*3}_{tile_h*2}"
        "[vout]"
    )
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for case_mp4 in case_mp4s:
        cmd.extend(["-i", str(case_mp4)])
    cmd.extend(
        [
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
        ]
    )
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))


def _build_3x4_board_png(
    *,
    snapshots: dict[str, Path],
    out_path: Path,
) -> None:
    case_images = {label: Image.open(path).convert("RGB") for label, path in snapshots.items()}
    try:
        ordered_specs = list(CASE_SPECS)
        tile_w = case_images[ordered_specs[0].label].width // 3
        tile_h = case_images[ordered_specs[0].label].height // 2
        board = Image.new("RGB", (tile_w * 4, tile_h * 3 + 88), (245, 245, 245))
        from PIL import ImageDraw, ImageFont

        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        title_font = ImageFont.truetype(font_path, size=30) if Path(font_path).exists() else ImageFont.load_default()
        label_font = ImageFont.truetype(font_path, size=22) if Path(font_path).exists() else ImageFont.load_default()
        painter = ImageDraw.Draw(board)
        painter.text((20, 16), "Restart@137 3x4 Reference Board", font=title_font, fill=(20, 20, 20))
        painter.text((20, 52), "Rows = View 0/1/2 | Cols = four continuation cases", font=label_font, fill=(70, 70, 70))
        for col, spec in enumerate(ordered_specs):
            img = case_images[spec.label]
            for row in range(3):
                crop = img.crop((row * tile_w, 0, (row + 1) * tile_w, tile_h))
                x0 = col * tile_w
                y0 = 88 + row * tile_h
                board.paste(crop, (x0, y0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        board.save(out_path)
    finally:
        for image in case_images.values():
            image.close()


def _build_case_command(
    args: argparse.Namespace,
    case: CaseSpec,
    *,
    derived_ir: Path,
    inference_slice: Path,
    out_case_dir: Path,
    num_frames: int,
) -> list[str]:
    return [
        str(args.python),
        str(VALIDATE_PARITY),
        "--ir",
        str(derived_ir.resolve()),
        "--out-dir",
        str(out_case_dir.resolve()),
        "--output-prefix",
        case.label,
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--num-frames",
        str(int(num_frames)),
        "--self-contact-mode",
        case.self_collision_law,
        "--ground-contact-law",
        case.ground_contact_law,
        "--custom-self-contact-hops",
        "0",
        "--phystwin-freeze-collision-table",
        "--phystwin-collision-table-capacity",
        "500",
        "--interpolate-controls",
        "--apply-drag",
        "--ground-restitution-mode",
        "approximate-native",
        "--disable-particle-contact-kernel",
        "--no-enable-tri-contact",
        "--no-shape-contacts",
        "--add-ground-plane",
        "--inference",
        str(inference_slice.resolve()),
        "--max-x0-rmse",
        str(STRICT_THRESHOLDS["x0_rmse"]),
        "--max-rmse-mean",
        str(STRICT_THRESHOLDS["rmse_mean"]),
        "--max-rmse-max",
        str(STRICT_THRESHOLDS["rmse_max"]),
        "--max-first30-rmse",
        str(STRICT_THRESHOLDS["first30_rmse"]),
        "--max-last30-rmse",
        str(STRICT_THRESHOLDS["last30_rmse"]),
    ]


def _collect_case_row(case: CaseSpec, report: dict[str, Any], proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    rollout_summary = report["rollout_summary"]
    simulation = rollout_summary["simulation"]
    contacts = rollout_summary["contacts"]
    checks = report["checks"]
    return {
        "case_label": case.label,
        "self_collision_law": case.self_collision_law,
        "ground_contact_law": case.ground_contact_law,
        "actual_self_collision_law": contacts["self_contact_mode"],
        "actual_ground_contact_law": contacts["ground_contact_law"],
        "x0_rmse": checks["x0_rmse"]["value"],
        "rmse_mean": checks["rmse_mean"]["value"],
        "rmse_max": checks["rmse_max"]["value"],
        "first30_rmse": checks["first30_rmse"]["value"],
        "last30_rmse": checks["last30_rmse"]["value"],
        "passed_strict_gate": bool(report["passed"]),
        "frame_count": int(simulation["frames_run"]),
        "dt": float(simulation["sim_dt"]),
        "substeps": int(simulation["substeps_per_frame"]),
        "ir_path": report["ir_path"],
        "importer_command": report["importer_command"],
        "validate_parity_exit_code": int(proc.returncode),
        "rollout_completed_without_nan_inf": bool(simulation["all_particle_positions_finite"]),
        "rollout_report_json": str(Path(report["rollout_json"]).with_name(Path(report["rollout_json"]).stem + "_rollout_report.json")),
        "rollout_json": report["rollout_json"],
        "rollout_npz": report["rollout_npz"],
        "rmse_curve_csv": report["rmse_curve_csv"],
    }


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_label",
        "self_collision_law",
        "ground_contact_law",
        "x0_rmse",
        "rmse_mean",
        "rmse_max",
        "first30_rmse",
        "last30_rmse",
        "frame_count",
        "dt",
        "substeps",
        "passed_strict_gate",
        "rollout_completed_without_nan_inf",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _write_matrix_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Restart@137 RMSE Matrix",
        "",
        "| case | self law | ground law | x0_rmse | rmse_mean | rmse_max | first30 | last30 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_label']}` | `{row['self_collision_law']}` | `{row['ground_contact_law']}` | "
            f"{row['x0_rmse']:.8f} | {row['rmse_mean']:.8f} | {row['rmse_max']:.8f} | "
            f"{row['first30_rmse']:.8f} | {row['last30_rmse']:.8f} |"
        )
    lines.append("")
    _write_text(path, "\n".join(lines))


def _build_fairness_payload(rows: list[dict[str, Any]], restart_info: dict[str, Any]) -> dict[str, Any]:
    fixed_keys = ("ir_path", "frame_count", "dt", "substeps")
    checks: dict[str, Any] = {}
    for key in fixed_keys:
        values = [row[key] for row in rows]
        checks[key] = {"pass": all(v == values[0] for v in values[1:]), "values": values}
    checks["restart_frame"] = {"pass": True, "values": [restart_info["start_frame"]] * len(rows)}
    checks["restart_velocity_mode"] = {"pass": True, "values": [restart_info["velocity_mode"]] * len(rows)}
    checks["same_state_source"] = {"pass": True, "values": ["PhysTwin reference @ frame boundary"] * len(rows)}
    checks["same_scene"] = {"pass": True, "values": [restart_info["source_case_name"]] * len(rows)}
    checks["same_initial_condition"] = {"pass": True, "values": [restart_info["derived_ir"]] * len(rows)}
    checks["only_law_axes_differ"] = {"pass": True}
    checks["law_labels_match_actual_paths"] = {
        "pass": all(
            row["self_collision_law"] == row["actual_self_collision_law"]
            and row["ground_contact_law"] == row["actual_ground_contact_law"]
            for row in rows
        )
    }
    return {
        "pass": all(v["pass"] for v in checks.values() if isinstance(v, dict) and "pass" in v),
        "checks": checks,
    }


def _write_fairness_md(path: Path, fairness: dict[str, Any], restart_info: dict[str, Any]) -> None:
    lines = [
        "# fairness_check",
        "",
        f"- pass: `{fairness['pass']}`",
        f"- restart frame: `{restart_info['start_frame']}`",
        f"- velocity estimate: `{restart_info['velocity_formula']}`",
        f"- continuation frames: `{restart_info['num_frames']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in fairness["checks"].items():
        if isinstance(value, dict) and "pass" in value:
            lines.append(f"- `{key}`: pass=`{value['pass']}` values=`{value.get('values')}`")
    lines.append("")
    _write_text(path, "\n".join(lines))


def _plot_rmse_curves(path_png: Path, path_csv: Path, rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(11, 5.8))
    curve_rows: list[tuple[int, str, float]] = []
    for idx, row in enumerate(rows):
        curve = np.loadtxt(Path(row["rmse_curve_csv"]), delimiter=",", skiprows=1, dtype=np.float32)
        if curve.ndim == 1:
            curve = curve[None, :]
        frames = curve[:, 0].astype(np.int32)
        values = curve[:, 1].astype(np.float32)
        plt.plot(frames, values, linewidth=2.2, label=row["case_label"])
        for f, v in zip(frames.tolist(), values.tolist()):
            curve_rows.append((int(f), row["case_label"], float(v)))
    plt.xlabel("Continuation frame index")
    plt.ylabel("RMSE vs PhysTwin reference")
    plt.title("Restart@137 continuation RMSE curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_png, dpi=180)
    plt.close()

    with path_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "case_label", "rmse"])
        writer.writerows(curve_rows)


def _root_readme(args: argparse.Namespace, restart_info: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Restart@137 Ground Contact / Self-Collision Matrix",
            "",
            "- task surface: `self_collision_transfer`",
            "- scene: `blue_cloth_double_lift_around`",
            f"- authoritative parent surface: `ground_contact_self_collision_repro_fix_20260404_200830_aa5e607`",
            f"- restart frame: `{restart_info['start_frame']}`",
            f"- restart state source: `PhysTwin reference rollout @ frame {restart_info['start_frame']}`",
            f"- object velocity estimate: `{restart_info['velocity_formula']}`",
            f"- continuation frames: `{restart_info['num_frames']}`",
            "",
            "Outputs:",
            "- derived restart IR + sliced inference",
            "- fair 2x2 continuation RMSE matrix",
            "- four labeled 2x3 comparison videos",
            "- one 3x4 labeled board video",
            "- continuation RMSE curve plot",
            "",
        ]
    )


def _history_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# sim/history",
        "",
        "This restart bundle is derived from the authoritative stable cloth+ground matrix surface.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['case_label']}`: `{row['rollout_report_json']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_dir()
    if out_dir.exists() and not args.skip_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_case_dir = args.case_dir.resolve()
    source_case_name = source_case_dir.name
    overlay_base = resolve_overlay_base(None)
    ir_path = args.ir.resolve()
    ir = np.load(ir_path, allow_pickle=False)
    ir_dict = {k: ir[k] for k in ir.files}

    inference = np.asarray(_load_pickle(source_case_dir / "inference.pkl"), dtype=np.float32)
    controller_traj = np.asarray(ir_dict["controller_traj"], dtype=np.float32)
    controller_idx = np.asarray(ir_dict["controller_idx"], dtype=np.int64).ravel()
    n_obj = int(np.asarray(ir_dict["num_object_points"]).ravel()[0])
    total_frames = int(inference.shape[0])
    start_frame = int(args.start_frame)
    if not (0 <= start_frame < total_frames - 1):
        raise ValueError(f"start_frame must satisfy 0 <= start_frame < {total_frames - 1}, got {start_frame}")

    sim_dt = float(np.asarray(ir_dict["sim_dt"]).reshape(-1)[0])
    sim_substeps = int(np.asarray(ir_dict["sim_substeps"]).reshape(-1)[0])
    frame_dt = sim_dt * sim_substeps
    num_frames = total_frames - start_frame

    obj_v0, velocity_formula = _compute_restart_velocity(
        inference,
        frame_idx=start_frame,
        frame_dt=frame_dt,
        mode=str(args.velocity_mode),
    )

    derived_case_name = f"{source_case_name}_restart{start_frame}"
    derived_case_dir = out_dir / "derived_inputs" / "case_dir" / derived_case_name
    derived_case_dir.mkdir(parents=True, exist_ok=True)
    derived_overlay_root = out_dir / "derived_inputs" / "overlay_base"
    derived_ir_path = out_dir / "derived_inputs" / f"{derived_case_name}_strict_ir.npz"
    inference_slice_path = out_dir / "derived_inputs" / f"{derived_case_name}_inference.pkl"

    x0 = np.asarray(ir_dict["x0"], dtype=np.float32).copy()
    v0 = np.asarray(ir_dict["v0"], dtype=np.float32).copy()
    x0[:n_obj] = inference[start_frame]
    v0[:n_obj] = obj_v0
    if controller_idx.size > 0:
        x0[controller_idx] = controller_traj[start_frame]
        v0[controller_idx] = 0.0

    derived_ir = dict(ir_dict)
    derived_ir["x0"] = x0.astype(np.float32, copy=False)
    derived_ir["v0"] = v0.astype(np.float32, copy=False)
    derived_ir["controller_traj"] = controller_traj[start_frame:].astype(np.float32, copy=False)
    derived_ir["case_name"] = np.asarray(derived_case_name)
    np.savez_compressed(derived_ir_path, **derived_ir)

    _dump_pickle(inference_slice_path, inference[start_frame:].astype(np.float32, copy=False))

    metadata_path = source_case_dir / "metadata.json"
    calibrate_path = source_case_dir / "calibrate.pkl"
    final_data_path = source_case_dir / "final_data.pkl"
    shutil.copy2(metadata_path, derived_case_dir / "metadata.json")
    shutil.copy2(calibrate_path, derived_case_dir / "calibrate.pkl")
    sliced_final_data = _slice_final_data(_load_pickle(final_data_path), start_frame=start_frame)
    _dump_pickle(derived_case_dir / "final_data.pkl", sliced_final_data)
    _build_overlay_slice(
        overlay_base=overlay_base,
        derived_case_name=derived_case_name,
        source_case_name=source_case_name,
        start_frame=start_frame,
        num_frames=num_frames,
        out_overlay_root=derived_overlay_root,
    )

    restart_info = {
        "start_frame": start_frame,
        "source_case_name": source_case_name,
        "derived_case_name": derived_case_name,
        "frame_dt": frame_dt,
        "velocity_mode": str(args.velocity_mode),
        "velocity_formula": velocity_formula,
        "num_frames": num_frames,
        "derived_ir": str(derived_ir_path.resolve()),
        "inference_slice": str(inference_slice_path.resolve()),
    }
    _write_json(derived_case_dir / "restart_info.json", restart_info)

    root_cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--ir",
        str(ir_path),
        "--case-dir",
        str(source_case_dir),
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--start-frame",
        str(start_frame),
        "--velocity-mode",
        str(args.velocity_mode),
        "--gif-width",
        str(args.gif_width),
        "--gif-fps",
        str(args.gif_fps),
        "--gif-max-colors",
        str(args.gif_max_colors),
        "--out-dir",
        str(out_dir),
    ]
    _write_text(out_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(root_cmd) + "\n")
    os.chmod(out_dir / "command.sh", 0o775)
    _write_text(out_dir / "README.md", _root_readme(args, restart_info))
    _write_json(out_dir / "restart_spec.json", restart_info)

    rows: list[dict[str, Any]] = []
    board_case_mp4s: list[Path] = []
    board_snapshots: dict[str, Path] = {}

    for spec in CASE_SPECS:
        case_dir = out_dir / spec.label
        report_path = case_dir / f"{spec.label}_rollout_report.json"
        if args.skip_existing and report_path.exists():
            report = _load_json(report_path)
            proc = subprocess.CompletedProcess([], 0 if bool(report.get("passed")) else 1, "", "")
        else:
            case_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_case_command(
                args,
                spec,
                derived_ir=derived_ir_path,
                inference_slice=inference_slice_path,
                out_case_dir=case_dir,
                num_frames=num_frames,
            )
            proc = _run_logged(
                cmd,
                workdir=WORKSPACE_ROOT,
                command_path=case_dir / "command.sh",
                log_path=case_dir / "run.log",
            )
            if proc.returncode not in (0, 1) or not report_path.exists():
                raise RuntimeError(f"{spec.label} failed; see {case_dir / 'run.log'}")
            report = _load_json(report_path)

        row = _collect_case_row(spec, report, proc)
        rows.append(row)

        mp4_path = case_dir / f"{spec.label}_cmp_2x3_labeled.mp4"
        gif_path = case_dir / f"{spec.label}_cmp_2x3_labeled.gif"
        snapshot_path = case_dir / f"{spec.label}_cmp_2x3_frame0.png"
        if not (args.skip_existing and mp4_path.exists() and gif_path.exists() and snapshot_path.exists()):
            case_visual_cmd = [
                str(args.python),
                str(RENDER_COMPARISON),
                "--report",
                str(report_path),
                "--case-dir",
                str(derived_case_dir),
                "--inference-pkl",
                str(inference_slice_path),
                "--overlay-base",
                str(derived_overlay_root),
                "--newton-label",
                f"{spec.short_label} | Newton Bridge",
                "--phystwin-label",
                "PhysTwin Reference | Restart@137",
                "--out-mp4",
                str(mp4_path),
            ]
            render_proc = _run_logged(
                case_visual_cmd,
                workdir=WORKSPACE_ROOT,
                command_path=case_dir / "visual_command.sh",
                log_path=case_dir / "visual.log",
            )
            if render_proc.returncode != 0:
                raise RuntimeError(f"Visualization failed for {spec.label}; see {case_dir / 'visual.log'}")

            gif_cmd = [
                str(RENDER_GIF),
                str(mp4_path),
                str(gif_path),
                str(int(args.gif_width)),
                str(int(args.gif_fps)),
                str(int(args.gif_max_colors)),
            ]
            gif_proc = _run_logged(
                gif_cmd,
                workdir=WORKSPACE_ROOT,
                command_path=case_dir / "gif_command.sh",
                log_path=case_dir / "gif.log",
            )
            if gif_proc.returncode != 0:
                raise RuntimeError(f"GIF render failed for {spec.label}; see {case_dir / 'gif.log'}")

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp4_path),
                    "-vf",
                    "select=eq(n\\,0)",
                    "-vframes",
                    "1",
                    str(snapshot_path),
                ],
                check=True,
                cwd=str(WORKSPACE_ROOT),
            )
        board_case_mp4s.append(mp4_path)
        board_snapshots[spec.label] = snapshot_path
        _write_text(
            case_dir / "README.md",
            "\n".join(
                [
                    f"# {spec.label}",
                    "",
                    f"- self_collision_law: `{spec.self_collision_law}`",
                    f"- ground_contact_law: `{spec.ground_contact_law}`",
                    f"- restart frame: `{start_frame}`",
                    f"- report: `{report_path}`",
                    f"- compare mp4: `{case_dir / f'{spec.label}_cmp_2x3_labeled.mp4'}`",
                    "",
                ]
            ),
        )

    fairness = _build_fairness_payload(rows, restart_info)
    _write_matrix_csv(out_dir / "rmse_matrix.csv", rows)
    _write_matrix_md(out_dir / "rmse_matrix.md", rows)
    _write_json(
        out_dir / "rmse_matrix_summary.json",
        {
            "start_frame": start_frame,
            "rows": rows,
            "best_case_by_rmse_mean": min(rows, key=lambda row: float(row["rmse_mean"]))["case_label"],
        },
    )
    _write_json(out_dir / "summary.json", {"start_frame": start_frame, "rows": rows, "fairness_pass": fairness["pass"]})
    _write_json(
        out_dir / "manifest.json",
        {
            "run_id": out_dir.name,
            "source_case_name": source_case_name,
            "derived_case_name": derived_case_name,
            "start_frame": start_frame,
            "velocity_formula": velocity_formula,
            "artifacts": {
                "rmse_matrix_csv": str((out_dir / "rmse_matrix.csv").resolve()),
                "rmse_curves_png": str((out_dir / "restart137_rmse_curves.png").resolve()),
            },
        },
    )
    _write_fairness_md(out_dir / "fairness_check.md", fairness, restart_info)
    _plot_rmse_curves(
        out_dir / "restart137_rmse_curves.png",
        out_dir / "restart137_rmse_curves.csv",
        rows,
    )
    _write_text(out_dir / "sim" / "history" / "README.md", _history_readme(rows))

    board_mp4 = out_dir / "restart137_reference_board_3x4_labeled.mp4"
    board_gif = out_dir / "restart137_reference_board_3x4_labeled.gif"
    board_png = out_dir / "restart137_reference_board_3x4_labeled.png"
    _compose_3x4_case_board_video(case_mp4s=board_case_mp4s, out_mp4=board_mp4)
    _run_logged(
        [
            str(RENDER_GIF),
            str(board_mp4),
            str(board_gif),
            str(int(args.gif_width)),
            str(int(args.gif_fps)),
            str(int(args.gif_max_colors)),
        ],
        workdir=WORKSPACE_ROOT,
        command_path=out_dir / "board_gif_command.sh",
        log_path=out_dir / "board_gif.log",
    )
    _build_3x4_board_png(snapshots=board_snapshots, out_path=board_png)

    validator_proc = subprocess.run(
        [
            str(args.python),
            str(ARTIFACT_VALIDATOR),
            str(out_dir),
            "--require-video",
            "--require-gif",
            "--summary-field",
            "rows",
        ],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(out_dir / "artifact_validation.log", validator_proc.stdout)
    if validator_proc.returncode != 0:
        raise RuntimeError(f"Artifact validation failed for {out_dir}; see artifact_validation.log")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
