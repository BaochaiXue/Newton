#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_defaults import default_python  # noqa: E402


RESULTS_ROOT = BRIDGE_ROOT / "results"
ARTIFACT_VALIDATOR = SCRIPTS_DIR / "validate_experiment_artifacts.py"
DEFAULT_MATRIX_ROOT = (
    BRIDGE_ROOT
    / "results"
    / "ground_contact_self_collision_repro_fix_20260404_200830_aa5e607"
    / "per_run"
    / "run_01"
    / "matrix"
)

CASES = (
    ("case_1_self_off_ground_native", "C1 self off / ground native", "#4c78a8"),
    ("case_2_self_off_ground_phystwin", "C2 self off / ground PhysTwin-style", "#72b7b2"),
    ("case_3_self_phystwin_ground_native", "C3 self PhysTwin-style / ground native", "#f58518"),
    ("case_4_self_phystwin_ground_phystwin", "C4 self PhysTwin-style / ground PhysTwin-style", "#e45756"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the four stable RMSE curves for the cloth + implicit-ground 2x2 matrix.")
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--out-dir", type=Path, default=None)
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
    return RESULTS_ROOT / f"ground_contact_self_collision_rmse_curves_{stamp}_{_resolve_git_sha()}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _load_curve(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        return np.asarray(data["rmse_per_frame"], dtype=np.float32)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    matrix_root = args.matrix_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--matrix-root",
        str(matrix_root),
        "--python",
        str(args.python),
        "--out-dir",
        str(out_dir),
    ]
    _write_text(out_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote(cmd) + "\n")
    os.chmod(out_dir / "command.sh", 0o775)

    records = []
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=220)
    first_positive = 40
    first_persistent = 107
    peak_window = 137
    ax.axvline(first_positive, color="#666666", linestyle="--", linewidth=1.2)
    ax.axvline(first_persistent, color="#2c7c2c", linestyle="--", linewidth=1.2)
    ax.axvline(peak_window, color="#8b0000", linestyle="--", linewidth=1.2)

    max_frames = 0
    for case_label, label, color in CASES:
        npz_path = matrix_root / case_label / f"{case_label}.npz"
        report_path = matrix_root / case_label / f"{case_label}_rollout_report.json"
        curve = _load_curve(npz_path)
        report = _load_report(report_path)
        x = np.arange(curve.shape[0])
        ax.plot(x, curve, label=label, linewidth=2.2, color=color)
        max_frames = max(max_frames, int(curve.shape[0]))
        records.append(
            {
                "case_label": case_label,
                "display_label": label,
                "rmse_mean": float(report["checks"]["rmse_mean"]["value"]),
                "rmse_max": float(report["checks"]["rmse_max"]["value"]),
                "first30_rmse": float(report["checks"]["first30_rmse"]["value"]),
                "last30_rmse": float(report["checks"]["last30_rmse"]["value"]),
                "curve_path": str(npz_path),
            }
        )

    ax.text(first_positive + 2, ax.get_ylim()[1] * 0.92, "first sign flip\nframe 40", color="#666666", fontsize=10)
    ax.text(first_persistent + 2, ax.get_ylim()[1] * 0.78, "persistent gap\nframe 107", color="#2c7c2c", fontsize=10)
    ax.text(peak_window + 2, ax.get_ylim()[1] * 0.98, "peak case4-case3 gap\nframe 137", color="#8b0000", fontsize=10, va="top")
    ax.set_title("Stable RMSE Curves For The 4-Case Cloth + Implicit-Ground Matrix", fontsize=18, fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("RMSE vs PhysTwin reference")
    ax.set_xlim(0, max_frames - 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    fig_path = out_dir / "ground_contact_self_collision_rmse_curves.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / "ground_contact_self_collision_rmse_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *[case for case, _label, _color in CASES]])
        curves = [
            _load_curve(matrix_root / case_label / f"{case_label}.npz")
            for case_label, _label, _color in CASES
        ]
        for frame in range(curves[0].shape[0]):
            writer.writerow([frame, *[float(curve[frame]) for curve in curves]])

    readme = "\n".join(
        [
            "# Ground Contact / Self-Collision RMSE Curves",
            "",
            f"- matrix root: `{matrix_root}`",
            "- plot: `ground_contact_self_collision_rmse_curves.png`",
            "- csv: `ground_contact_self_collision_rmse_curves.csv`",
            "- marker lines:",
            "  - frame 40: first sign flip between case 4 and case 3",
            "  - frame 107: first persistent positive gap",
            "  - frame 137: peak positive gap window",
            "",
        ]
    )
    _write_text(out_dir / "README.md", readme)
    _write_text(
        out_dir / "sim" / "history" / "README.md",
        "\n".join(
            [
                "# sim/history",
                "",
                "This plot is derived from the stable matrix root:",
                f"- `{matrix_root}`",
                "",
            ]
        ),
    )

    summary = {
        "run_type": "ground_contact_self_collision_rmse_curves",
        "matrix_root": str(matrix_root),
        "plot_png": str(fig_path),
        "curve_csv": str(csv_path),
        "cases": records,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "manifest.json", summary)

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
