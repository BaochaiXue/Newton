#!/usr/bin/env python3
"""Run the controlled box self-collision decision matrix.

This runner executes the existing cloth-on-box self-contact demo under a fixed
scene configuration and aggregates the results into decision artifacts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_core_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_defaults = _load_core_module("self_collision_matrix_path_defaults", CORE_DIR / "path_defaults.py")

DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
DEMO_SCRIPT = BRIDGE_ROOT / "demos" / "demo_cloth_box_drop_with_self_contact.py"


@dataclass(frozen=True)
class ModeSpec:
    alias: str
    self_contact_mode: str
    custom_hops: int | None = None
    include_in_figure: bool = True


MODE_SPECS = [
    ModeSpec(alias="off", self_contact_mode="off", include_in_figure=False),
    ModeSpec(alias="native", self_contact_mode="native"),
    ModeSpec(alias="custom_h1", self_contact_mode="custom", custom_hops=1),
    ModeSpec(alias="custom_h2", self_contact_mode="custom", custom_hops=2),
    ModeSpec(alias="phystwin", self_contact_mode="phystwin"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the box self-collision decision matrix.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=path_defaults.default_python())
    parser.add_argument("--device", default=path_defaults.default_device())
    parser.add_argument("--prefix", default="cloth_box_decision")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--sim-dt", type=float, default=1.0 / 600.0)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--drop-height", type=float, default=0.5)
    parser.add_argument("--target-total-mass", type=float, default=0.1)
    parser.add_argument("--contact-dist-scale", type=float, default=1.0)
    parser.add_argument("--on-contact-dist-scale", type=float, default=1.0)
    parser.add_argument("--rigid-mass", type=float, default=4000.0)
    parser.add_argument("--shape-contact-scale", type=float, default=None)
    parser.add_argument("--shape-contact-damping-multiplier", type=float, default=1.0)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--slowdown", type=float, default=2.0)
    parser.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_font(size: int = 18, *, bold: bool = False):
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _append_bool_flag(cmd: list[str], flag_name: str, enabled: bool) -> None:
    cmd.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_case_command(args: argparse.Namespace, mode: ModeSpec, case_base_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(DEMO_SCRIPT),
        "--ir",
        str(args.ir.resolve()),
        "--out-dir",
        str(case_base_dir.resolve()),
        "--prefix",
        args.prefix,
        "--device",
        str(args.device),
        "--frames",
        str(int(args.frames)),
        "--sim-dt",
        str(float(args.sim_dt)),
        "--substeps",
        str(int(args.substeps)),
        "--drop-height",
        str(float(args.drop_height)),
        "--target-total-mass",
        str(float(args.target_total_mass)),
        "--contact-dist-scale",
        str(float(args.contact_dist_scale)),
        "--on-contact-dist-scale",
        str(float(args.on_contact_dist_scale)),
        "--rigid-mass",
        str(float(args.rigid_mass)),
        "--render-fps",
        str(float(args.render_fps)),
        "--slowdown",
        str(float(args.slowdown)),
        "--screen-width",
        str(int(args.screen_width)),
        "--screen-height",
        str(int(args.screen_height)),
        "--self-contact-mode",
        mode.self_contact_mode,
    ]
    if mode.custom_hops is not None:
        cmd.extend(["--custom-self-contact-hops", str(int(mode.custom_hops))])
    if args.shape_contact_scale is not None:
        cmd.extend(["--shape-contact-scale", str(float(args.shape_contact_scale))])
    cmd.extend(
        [
            "--shape-contact-damping-multiplier",
            str(float(args.shape_contact_damping_multiplier)),
        ]
    )
    _append_bool_flag(cmd, "dynamic-box", False)
    _append_bool_flag(cmd, "add-ground-plane", False)
    _append_bool_flag(cmd, "shape-contacts", True)
    _append_bool_flag(cmd, "viewer-headless", True)
    _append_bool_flag(cmd, "skip-render", bool(args.skip_render))
    return cmd


def _find_unique(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern!r} under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one match for {pattern!r} under {root}, got {matches}")
    return matches[0]


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _thumbnail_mp4(mp4_path: Path, out_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            "thumbnail,scale=720:-1",
            "-frames:v",
            "1",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def _evaluate_candidate(summary: dict[str, Any], off_wall_time: float | None) -> tuple[dict[str, Any], float]:
    radius = float(summary.get("particle_radius_median_m", 0.0) or 0.0)
    frames = max(int(summary.get("frames", 0) or 0), 1)
    wall_time = float(summary.get("wall_time_sec", 0.0) or 0.0)
    checks = {
        "finite": bool(summary.get("all_particle_positions_finite", False)),
        "nan_inf": int(summary.get("nan_inf_frame_count", 0) or 0) == 0,
        "peak_p95_overlap": radius > 0.0
        and float(summary.get("peak_nonexcluded_self_contact_p95_overlap_m_over_time", float("inf"))) <= 0.5 * radius,
        "peak_max_overlap": radius > 0.0
        and float(summary.get("peak_nonexcluded_self_contact_max_overlap_m_over_time", float("inf"))) <= 1.0 * radius,
        "persistent_overlap": int(summary.get("persistent_overlap_frames", frames + 1) or 0) <= int(0.05 * frames),
        "box_p99": radius > 0.0
        and float(summary.get("final_penetration_p99_box_m", float("inf")) or float("inf")) <= 0.5 * radius,
        "box_max": radius > 0.0
        and float(summary.get("max_penetration_depth_box_m", float("inf")) or float("inf")) <= 1.0 * radius,
        "wall_ratio": (
            off_wall_time is not None
            and off_wall_time > 1.0e-12
            and wall_time / off_wall_time <= 2.0
        ),
    }

    violation_score = 0.0
    if not checks["finite"]:
        violation_score += 1000.0
    if not checks["nan_inf"]:
        violation_score += 1000.0

    def _excess(value: float, limit: float) -> float:
        if limit <= 1.0e-12:
            return 1000.0
        return max(0.0, value / limit - 1.0)

    if radius > 0.0:
        violation_score += _excess(
            float(summary.get("peak_nonexcluded_self_contact_p95_overlap_m_over_time", 0.0)),
            0.5 * radius,
        )
        violation_score += _excess(
            float(summary.get("peak_nonexcluded_self_contact_max_overlap_m_over_time", 0.0)),
            1.0 * radius,
        )
        violation_score += _excess(
            float(summary.get("final_penetration_p99_box_m", 0.0) or 0.0),
            0.5 * radius,
        )
        violation_score += _excess(
            float(summary.get("max_penetration_depth_box_m", 0.0) or 0.0),
            1.0 * radius,
        )
    if off_wall_time is not None and off_wall_time > 1.0e-12:
        violation_score += _excess(wall_time, 2.0 * off_wall_time)
    persistent_limit = max(1, int(0.05 * frames))
    violation_score += max(
        0.0,
        float(int(summary.get("persistent_overlap_frames", 0) or 0) - persistent_limit) / persistent_limit,
    )
    return checks, float(violation_score)


def _select_provisional_mode(rows: list[dict[str, Any]]) -> tuple[str | None, str]:
    preference = ["native", "custom_h1", "custom_h2", "phystwin"]
    by_alias = {str(row["case_alias"]): row for row in rows}
    for alias in preference:
        row = by_alias.get(alias)
        if row and bool(row.get("passes_provisional_thresholds", False)):
            return alias, "passes provisional thresholds with minimum adaptation"

    candidates = [row for row in rows if str(row.get("case_alias")) != "off"]
    if not candidates:
        return None, "no decision candidates were produced"
    best = min(candidates, key=lambda row: float(row.get("violation_score", 1.0e9)))
    return str(best["case_alias"]), "best provisional score despite failing one or more thresholds"


def _render_comparison_figure(rows: list[dict[str, Any]], out_path: Path) -> Path:
    candidates = [row for row in rows if bool(row.get("include_in_figure", False))]
    panel_w, panel_h = 760, 470
    title_h = 150
    gap = 28
    canvas = Image.new("RGB", (panel_w * 2 + gap * 3, title_h + panel_h * 2 + gap * 3), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(34, bold=True)
    body_font = _load_font(20, bold=False)
    small_font = _load_font(17, bold=False)

    draw.text((40, 28), "Self-Collision Decision Matrix", font=title_font, fill=(24, 58, 96))
    draw.text((40, 80), "Candidate modes only: native vs custom_h1 vs custom_h2 vs phystwin", font=body_font, fill=(71, 85, 105))

    for idx, row in enumerate(candidates[:4]):
        thumb = row.get("thumbnail_png")
        x0 = gap + (idx % 2) * (panel_w + gap)
        y0 = title_h + gap + (idx // 2) * (panel_h + gap)
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=20, fill=(255, 255, 255), outline=(181, 194, 209), width=2)
        if thumb and Path(str(thumb)).exists():
            with Image.open(str(thumb)).convert("RGB") as img:
                fit_w = panel_w - 30
                fit_h = panel_h - 130
                scale = min(fit_w / img.width, fit_h / img.height)
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size)
                paste_x = x0 + 15 + (fit_w - new_size[0]) // 2
                paste_y = y0 + 15 + (fit_h - new_size[1]) // 2
                canvas.paste(img, (paste_x, paste_y))
        pass_flag = bool(row.get("passes_provisional_thresholds", False))
        status = "PASS" if pass_flag else "FAIL"
        status_color = (22, 163, 74) if pass_flag else (220, 38, 38)
        draw.text((x0 + 18, y0 + panel_h - 98), f"{row['case_alias']}  [{status}]", font=body_font, fill=status_color)
        draw.text(
            (x0 + 18, y0 + panel_h - 70),
            f"peak p95 overlap = {float(row.get('peak_nonexcluded_self_contact_p95_overlap_m_over_time', 0.0)) * 1000.0:.2f} mm",
            font=small_font,
            fill=(31, 41, 55),
        )
        draw.text(
            (x0 + 18, y0 + panel_h - 46),
            f"peak max overlap = {float(row.get('peak_nonexcluded_self_contact_max_overlap_m_over_time', 0.0)) * 1000.0:.2f} mm",
            font=small_font,
            fill=(31, 41, 55),
        )
        draw.text(
            (x0 + 18, y0 + panel_h - 22),
            f"box p99 = {float(row.get('final_penetration_p99_box_m', 0.0) or 0.0) * 1000.0:.2f} mm",
            font=small_font,
            fill=(31, 41, 55),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(value) for key, value in row.items()})


def _write_decision_md(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    selected_mode: str | None,
    selected_reason: str,
    figure_path: Path,
    selected_mp4: Path | None,
) -> None:
    lines = [
        "# Self-Collision Decision",
        "",
        "## Current State",
        "",
        "- Final A/B/C decision: ",
        f"- Provisional selected mode: `{selected_mode}`" if selected_mode else "- Provisional selected mode: none",
        f"- Selection reason: {selected_reason}",
        "- Decision note: final A/B/C remains blank until bunny sanity check, OFF-baseline regression, and PhysTwin-style equivalence evidence are attached.",
        "",
        "## Matrix Summary",
        "",
        "| mode | provisional pass | peak p95 overlap (mm) | peak max overlap (mm) | persistent frames | box p99 (mm) | wall ratio vs off |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if str(row.get("case_alias")) == "off":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_alias"]),
                    "yes" if bool(row.get("passes_provisional_thresholds", False)) else "no",
                    f"{float(row.get('peak_nonexcluded_self_contact_p95_overlap_m_over_time', 0.0)) * 1000.0:.3f}",
                    f"{float(row.get('peak_nonexcluded_self_contact_max_overlap_m_over_time', 0.0)) * 1000.0:.3f}",
                    str(int(row.get("persistent_overlap_frames", 0) or 0)),
                    f"{float(row.get('final_penetration_p99_box_m', 0.0) or 0.0) * 1000.0:.3f}",
                    f"{float(row.get('wall_time_ratio_vs_off', 0.0) or 0.0):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- comparison figure: `{figure_path}`",
            f"- provisional selected mp4: `{selected_mp4}`" if selected_mp4 is not None else "- provisional selected mp4: none",
            "",
        ]
    )
    _write_text(path, "\n".join(lines) + "\n")


def _run_case(cmd: list[str], *, case_base_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    case_base_dir.mkdir(parents=True, exist_ok=True)
    command_path = case_base_dir / "command.sh"
    log_path = case_base_dir / "run.log"
    _write_text(command_path, _quote_cmd(cmd) + "\n")
    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(log_path, proc.stdout)
    summary_path = _find_unique(case_base_dir, "*_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, summary_path, log_path


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        out_dir / "command.sh",
        _quote_cmd([sys.executable, *sys.argv]) + "\n",
    )
    _write_text(
        out_dir / "README.md",
        "\n".join(
            [
                "# Self-Collision Decision Matrix",
                "",
                "This directory contains the controlled cloth-on-box self-collision decision matrix.",
                "",
                "Contents:",
                "- per-mode subdirectories with `command.sh`, `run.log`, `scene.npz`, and `*_summary.json`",
                "- `self_collision_decision_table.csv`",
                "- `self_collision_decision.md`",
                "- `self_collision_decision_4panel.png`",
                "- `self_collision_provisional_selected_mode.mp4` when a provisional mode has a render artifact",
                "",
                "Final A/B/C decision remains blank until bunny sanity check, OFF-baseline regression, and equivalence evidence are attached.",
                "",
            ]
        ),
    )

    rows: list[dict[str, Any]] = []
    off_wall_time: float | None = None
    case_outputs: dict[str, dict[str, Path | dict[str, Any]]] = {}

    for spec in MODE_SPECS:
        case_base_dir = out_dir / spec.alias
        cmd = _build_case_command(args, spec, case_base_dir)
        expected_summary = sorted(case_base_dir.rglob("*_summary.json"))
        if args.skip_existing and expected_summary:
            summary_path = _find_unique(case_base_dir, "*_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            log_path = case_base_dir / "run.log"
            command_path = case_base_dir / "command.sh"
        else:
            print(f"Running {spec.alias}...", flush=True)
            summary, summary_path, log_path = _run_case(cmd, case_base_dir=case_base_dir)
            command_path = case_base_dir / "command.sh"

        mp4_path = None
        mp4_matches = sorted(case_base_dir.rglob("*.mp4"))
        if mp4_matches:
            mp4_path = mp4_matches[0]
        case_outputs[spec.alias] = {
            "summary": summary,
            "summary_path": summary_path,
            "log_path": log_path,
            "command_path": command_path,
            "mp4_path": mp4_path,
        }
        if spec.alias == "off":
            off_wall_time = float(summary.get("wall_time_sec", 0.0) or 0.0)

    for spec in MODE_SPECS:
        payload = case_outputs[spec.alias]
        summary = dict(payload["summary"])  # shallow copy
        row = {
            "case_alias": spec.alias,
            "self_contact_mode": spec.self_contact_mode,
            "custom_self_contact_hops": spec.custom_hops,
            "include_in_figure": spec.include_in_figure,
            **summary,
            "summary_json_path": str(payload["summary_path"]),
            "run_log_path": str(payload["log_path"]),
            "command_path": str(payload["command_path"]),
            "video_mp4_path": "" if payload["mp4_path"] is None else str(payload["mp4_path"]),
        }
        if spec.alias == "off":
            row["wall_time_ratio_vs_off"] = 1.0
            row["passes_provisional_thresholds"] = False
            row["violation_score"] = 0.0
        else:
            checks, score = _evaluate_candidate(summary, off_wall_time)
            row["wall_time_ratio_vs_off"] = (
                float(summary.get("wall_time_sec", 0.0) or 0.0) / off_wall_time
                if off_wall_time and off_wall_time > 1.0e-12
                else None
            )
            row["passes_provisional_thresholds"] = bool(all(checks.values()))
            row["violation_score"] = float(score)
            for key, value in checks.items():
                row[f"check_{key}"] = bool(value)
        if payload["mp4_path"] is not None and spec.include_in_figure:
            thumb_path = out_dir / spec.alias / "thumbnail.png"
            try:
                _thumbnail_mp4(Path(str(payload["mp4_path"])), thumb_path)
                row["thumbnail_png"] = str(thumb_path)
            except Exception:
                row["thumbnail_png"] = ""
        rows.append(row)

    selected_mode, selected_reason = _select_provisional_mode(rows)
    selected_mp4 = None
    if selected_mode is not None:
        selected_payload = case_outputs.get(selected_mode, {})
        if selected_payload.get("mp4_path") is not None:
            selected_mp4 = out_dir / "self_collision_provisional_selected_mode.mp4"
            shutil.copy2(Path(str(selected_payload["mp4_path"])), selected_mp4)

    csv_path = out_dir / "self_collision_decision_table.csv"
    md_path = out_dir / "self_collision_decision.md"
    fig_path = out_dir / "self_collision_decision_4panel.png"
    _write_matrix_csv(csv_path, rows)
    _render_comparison_figure(rows, fig_path)
    _write_decision_md(
        md_path,
        rows,
        selected_mode=selected_mode,
        selected_reason=selected_reason,
        figure_path=fig_path,
        selected_mp4=selected_mp4,
    )

    print(f"Decision CSV: {csv_path}", flush=True)
    print(f"Decision MD: {md_path}", flush=True)
    print(f"Comparison Figure: {fig_path}", flush=True)
    if selected_mp4 is not None:
        print(f"Selected MP4: {selected_mp4}", flush=True)
    print(f"Provisional selected mode: {selected_mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
