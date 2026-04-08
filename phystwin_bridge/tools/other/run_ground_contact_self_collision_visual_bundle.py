#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_defaults import default_device, default_python  # noqa: E402


RESULTS_ROOT = BRIDGE_ROOT / "results"
RENDER_COMPARISON = Path(__file__).resolve().with_name("render_comparison_2x3_mp4.py")
RENDER_GIF = SCRIPTS_DIR / "render_gif.sh"
ARTIFACT_VALIDATOR = SCRIPTS_DIR / "validate_experiment_artifacts.py"
DEFAULT_MATRIX_ROOT = (
    BRIDGE_ROOT
    / "results"
    / "ground_contact_self_collision_repro_fix_20260404_200830_aa5e607"
    / "per_run"
    / "run_01"
    / "matrix"
)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    short_label: str
    self_law: str
    ground_law: str


CASES = (
    CaseSpec("case_1_self_off_ground_native", "C1 off/native", "off", "native"),
    CaseSpec("case_2_self_off_ground_phystwin", "C2 off/ps-ground", "off", "phystwin"),
    CaseSpec("case_3_self_phystwin_ground_native", "C3 ps-self/native", "phystwin", "native"),
    CaseSpec("case_4_self_phystwin_ground_phystwin", "C4 ps-self/ps-ground", "phystwin", "phystwin"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the 4 stable self-collision matrix cases into labeled 2x3 comparison videos and a 3x4 board."
    )
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--snapshot-frame", type=int, default=137)
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
    return RESULTS_ROOT / f"ground_contact_self_collision_visual_bundle_{stamp}_{_resolve_git_sha()}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_logged(cmd: list[str], *, workdir: Path, command_path: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
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


def _extract_snapshot(mp4_path: Path, out_png: Path, frame_index: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))


def _font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _build_3x4_board(
    *,
    snapshots: dict[str, Path],
    out_path: Path,
    case_specs: tuple[CaseSpec, ...],
) -> None:
    case_images = {label: Image.open(path).convert("RGB") for label, path in snapshots.items()}
    try:
        tile_w = case_images[case_specs[0].label].width // 3
        tile_h = case_images[case_specs[0].label].height // 2
        board_w = tile_w * len(case_specs)
        header_h = 90
        board_h = header_h + tile_h * 3
        board = Image.new("RGBA", (board_w, board_h), (245, 245, 245, 255))
        draw = ImageDraw.Draw(board)
        title_font = _font(30)
        label_font = _font(22)
        small_font = _font(18)

        draw.text((20, 18), "Stable 3x4 Reference Board", font=title_font, fill=(20, 20, 20))
        draw.text(
            (20, 54),
            "Rows = views 0/1/2 from Newton Bridge rollout snapshot | Cols = 4 matrix cases",
            font=small_font,
            fill=(70, 70, 70),
        )

        for col, spec in enumerate(case_specs):
            x0 = col * tile_w
            img = case_images[spec.label]
            for row in range(3):
                crop = img.crop((row * tile_w, 0, (row + 1) * tile_w, tile_h))
                y0 = header_h + row * tile_h
                board.paste(crop.convert("RGBA"), (x0, y0))

                overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle((10, 10, tile_w - 10, 58), fill=(0, 0, 0, 150))
                overlay_draw.text((24, 18), f"{spec.short_label} | View {row}", font=label_font, fill=(255, 255, 255, 255))
                board.alpha_composite(overlay, (x0, y0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        board.convert("RGB").save(out_path)
    finally:
        for img in case_images.values():
            img.close()


def _compose_3x4_case_board_video(
    *,
    case_mp4s: list[Path],
    out_mp4: Path,
) -> None:
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
    filter_complex = ";".join(filters)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for case_mp4 in case_mp4s:
        cmd.extend(["-i", str(case_mp4)])
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
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))


def _root_readme(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Ground Contact / Self-Collision Visual Bundle",
            "",
            "- source matrix root: `{}`".format(args.matrix_root.resolve()),
            "- scene: `blue_cloth_double_lift_around`",
            "- outputs:",
            "  - four labeled 2x3 comparison videos",
            "  - four labeled deck GIFs",
            "  - one 3x4 labeled reference board image",
            "",
        ]
    )


def _history_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# sim/history",
        "",
        "This visual bundle is derived from the stable 2x2 matrix root.",
        "Per-case rollout sources:",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['case_label']}`: `{row['rollout_report_json']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    matrix_root = args.matrix_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_text(out_dir / "README.md", _root_readme(args))
    root_cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--matrix-root",
        str(matrix_root),
        "--python",
        str(args.python),
        "--snapshot-frame",
        str(int(args.snapshot_frame)),
        "--gif-width",
        str(int(args.gif_width)),
        "--gif-fps",
        str(int(args.gif_fps)),
        "--gif-max-colors",
        str(int(args.gif_max_colors)),
        "--out-dir",
        str(out_dir),
    ]
    _write_text(out_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(root_cmd) + "\n")
    os.chmod(out_dir / "command.sh", 0o775)

    rows: list[dict[str, Any]] = []
    snapshot_paths: dict[str, Path] = {}

    for spec in CASES:
        case_dir = out_dir / spec.label
        case_dir.mkdir(parents=True, exist_ok=True)
        report_path = matrix_root / spec.label / f"{spec.label}_rollout_report.json"
        report = _load_json(report_path)
        mp4_path = case_dir / f"{spec.label}_cmp_2x3_labeled.mp4"
        gif_path = case_dir / f"{spec.label}_cmp_2x3_labeled.gif"
        snapshot_png = case_dir / f"{spec.label}_snapshot_frame{int(args.snapshot_frame)}.png"

        if not (args.skip_existing and mp4_path.exists()):
            cmd = [
                str(args.python),
                str(RENDER_COMPARISON),
                "--report",
                str(report_path),
                "--newton-label",
                f"{spec.short_label} | Newton",
                "--phystwin-label",
                f"{spec.short_label} | Ref",
                "--camera-indices",
                "0,1,2",
                "--out-mp4",
                str(mp4_path),
            ]
            proc = _run_logged(
                cmd,
                workdir=WORKSPACE_ROOT,
                command_path=case_dir / "command.sh",
                log_path=case_dir / "run.log",
            )
            if proc.returncode != 0:
                raise RuntimeError(f"render_comparison_2x3_mp4 failed for {spec.label}")

        if not (args.skip_existing and gif_path.exists()):
            subprocess.run(
                [
                    str(RENDER_GIF),
                    str(mp4_path),
                    str(gif_path),
                    str(int(args.gif_width)),
                    str(int(args.gif_fps)),
                    str(int(args.gif_max_colors)),
                ],
                check=True,
                cwd=str(WORKSPACE_ROOT),
            )

        _extract_snapshot(mp4_path, snapshot_png, int(args.snapshot_frame))
        snapshot_paths[spec.label] = snapshot_png

        rows.append(
            {
                "case_label": spec.label,
                "short_label": spec.short_label,
                "self_collision_law": spec.self_law,
                "ground_contact_law": spec.ground_law,
                "rollout_report_json": str(report_path),
                "mp4": str(mp4_path),
                "gif": str(gif_path),
                "snapshot_png": str(snapshot_png),
                "rmse_mean": float(report["checks"]["rmse_mean"]["value"]),
            }
        )
        _write_text(
            case_dir / "README.md",
            "\n".join(
                [
                    f"# {spec.label}",
                    "",
                    f"- self_collision_law: `{spec.self_law}`",
                    f"- ground_contact_law: `{spec.ground_law}`",
                    f"- mp4: `{mp4_path}`",
                    f"- gif: `{gif_path}`",
                    f"- snapshot: `{snapshot_png}`",
                    "",
                ]
            ),
        )

    board_path = out_dir / "ground_contact_self_collision_reference_board_3x4_labeled.png"
    _build_3x4_board(snapshots=snapshot_paths, out_path=board_path, case_specs=CASES)

    manifest = {
        "run_type": "ground_contact_self_collision_visual_bundle",
        "matrix_root": str(matrix_root),
        "cases": rows,
        "reference_board_png": str(board_path),
    }
    board_video_mp4 = out_dir / "ground_contact_self_collision_reference_board_3x4_labeled.mp4"
    board_video_gif = out_dir / "ground_contact_self_collision_reference_board_3x4_labeled.gif"
    _compose_3x4_case_board_video(
        case_mp4s=[Path(row["mp4"]) for row in rows],
        out_mp4=board_video_mp4,
    )
    subprocess.run(
        [
            str(RENDER_GIF),
            str(board_video_mp4),
            str(board_video_gif),
            str(int(args.gif_width)),
            str(int(args.gif_fps)),
            str(int(args.gif_max_colors)),
        ],
        check=True,
        cwd=str(WORKSPACE_ROOT),
    )
    manifest["reference_board_mp4"] = str(board_video_mp4)
    manifest["reference_board_gif"] = str(board_video_gif)
    _write_text(out_dir / "sim" / "history" / "README.md", _history_readme(rows))
    _write_json(out_dir / "summary.json", manifest)
    _write_json(out_dir / "manifest.json", manifest)
    _write_text(
        out_dir / "matrix_visual_summary.md",
        "\n".join(
            [
                "# 4-case Visual Bundle",
                "",
                "| case | self law | ground law | rmse_mean | mp4 | gif |",
                "| --- | --- | --- | ---: | --- | --- |",
                *[
                    f"| {row['case_label']} | {row['self_collision_law']} | {row['ground_contact_law']} | {row['rmse_mean']:.9g} | `{Path(row['mp4']).name}` | `{Path(row['gif']).name}` |"
                    for row in rows
                ],
                "",
                f"- reference board: `{board_path.name}`",
                "",
            ]
        ),
    )

    validator = subprocess.run(
        [str(args.python), str(ARTIFACT_VALIDATOR), str(out_dir)],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(out_dir / "artifact_validation.log", validator.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
