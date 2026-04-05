#!/usr/bin/env python3
"""Repeated reproducibility audit for the controlled 2x2 cloth+ground matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_defaults import default_device, default_python  # noqa: E402


MATRIX_RUNNER = Path(__file__).resolve().parent / "run_ground_contact_self_collision_rmse_matrix.py"
DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
RESULTS_ROOT = BRIDGE_ROOT / "results"
ARTIFACT_VALIDATOR = WORKSPACE_ROOT / "scripts" / "validate_experiment_artifacts.py"
DEFAULT_BEFORE_ROOT = (
    BRIDGE_ROOT
    / "results"
    / "self_collision_case3_vs_case4_diagnosis_20260404_162159_6cb033a"
)

CASE_LABELS = (
    "case_1_self_off_ground_native",
    "case_2_self_off_ground_phystwin",
    "case_3_self_phystwin_ground_native",
    "case_4_self_phystwin_ground_phystwin",
)

DETERMINISM_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated full-matrix reruns and audit reproducibility."
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-frames", type=int, default=302)
    parser.add_argument("--substeps-per-frame", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--before-root", type=Path, default=DEFAULT_BEFORE_ROOT)
    return parser.parse_args()


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return RESULTS_ROOT / f"ground_contact_self_collision_repro_fix_{stamp}_{_resolve_git_sha()}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _determinism_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(DETERMINISM_ENV)
    return env


def _python_env_snapshot(python_bin: str, device: str, env: dict[str, str]) -> dict[str, Any]:
    script = """
import json, os, platform, socket, sys
import numpy as np
import scipy
import warp as wp

wp.init()
device_name = None
try:
    dev = wp.get_device(sys.argv[1])
    device_name = str(dev)
except Exception as exc:
    device_name = f"unresolved:{exc}"

payload = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "hostname": socket.gethostname(),
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "warp_version": wp.__version__,
    "warp_device": device_name,
}
print(json.dumps(payload))
"""
    proc = subprocess.run(
        [str(python_bin), "-c", script, str(device)],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "error": "python_env_snapshot_failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except Exception:
        return {
            "error": "python_env_snapshot_parse_failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


def _gpu_snapshot() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version",
            "--format=csv,noheader",
        ],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _build_matrix_command(args: argparse.Namespace, matrix_out_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(MATRIX_RUNNER),
        "--ir",
        str(args.ir.resolve()),
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--num-frames",
        str(int(args.num_frames)),
        "--out-dir",
        str(matrix_out_dir.resolve()),
    ]
    if args.substeps_per_frame is not None:
        cmd.extend(["--substeps-per-frame", str(int(args.substeps_per_frame))])
    if args.sim_dt is not None:
        cmd.extend(["--sim-dt", str(float(args.sim_dt))])
    return cmd


def _run_logged(cmd: list[str], *, env: dict[str, str], command_path: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
    ]
    for key, value in DETERMINISM_ENV.items():
        script_lines.append(f"export {key}={shlex.quote(value)}")
    script_lines.append(_quote_cmd(cmd))
    _write_text(command_path, "\n".join(script_lines) + "\n")
    os.chmod(command_path, 0o775)
    proc = subprocess.run(
        cmd,
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    _write_text(log_path, proc.stdout)
    return proc


def _load_rows(matrix_root: Path) -> list[dict[str, Any]]:
    payload = _load_json(matrix_root / "rmse_matrix_summary.json")
    return payload["rows"]


def _ranking(rows: list[dict[str, Any]]) -> list[str]:
    return [
        row["case_label"]
        for row in sorted(rows, key=lambda item: (float(item["rmse_mean"]), item["case_label"]))
    ]


def _collect_run_rows(run_index: int, rows: list[dict[str, Any]], ranking: list[str]) -> list[dict[str, Any]]:
    ranking_pos = {label: idx + 1 for idx, label in enumerate(ranking)}
    ranking_string = " > ".join(ranking)
    out: list[dict[str, Any]] = []
    for row in rows:
        rollout_npz = Path(row["rollout_npz"])
        rmse_curve_csv = Path(row["rmse_curve_csv"])
        out.append(
            {
                "run_index": run_index,
                "case_label": row["case_label"],
                "self_collision_law": row["self_collision_law"],
                "ground_contact_law": row["ground_contact_law"],
                "rmse_mean": float(row["rmse_mean"]),
                "rmse_max": float(row["rmse_max"]),
                "first30_rmse": float(row["first30_rmse"]),
                "last30_rmse": float(row["last30_rmse"]),
                "x0_rmse": float(row["x0_rmse"]),
                "dt": float(row["dt"]),
                "substeps": int(row["substeps"]),
                "frame_count": int(row["frame_count"]),
                "ranking_position": ranking_pos[row["case_label"]],
                "ranking_string": ranking_string,
                "rollout_npz_sha256": _sha256(rollout_npz),
                "rmse_curve_sha256": _sha256(rmse_curve_csv),
                "rollout_npz": str(rollout_npz),
                "rmse_curve_csv": str(rmse_curve_csv),
            }
        )
    return out


def _drift_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "span": float(max(values) - min(values)),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _before_payload(before_root: Path) -> dict[str, Any] | None:
    if not before_root.exists():
        return None
    path = before_root / "summary.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    return payload.get("reproducibility")


def _summary_markdown(
    *,
    ranking_strings: list[str],
    fairness_passes: list[bool],
    per_case_stats: dict[str, Any],
    verdict: str,
) -> str:
    lines = [
        "# Repro Audit Summary",
        "",
        f"- verdict: `{verdict}`",
        f"- ranking invariant across reruns: `{len(set(ranking_strings)) == 1}`",
        f"- fairness pass invariant: `{all(fairness_passes)}`",
        "",
        "## Ranking Per Rerun",
        "",
    ]
    for idx, ranking in enumerate(ranking_strings, start=1):
        lines.append(f"- run_{idx:02d}: `{ranking}`")
    lines.extend(["", "## Per-Case Drift", ""])
    for case_label in CASE_LABELS:
        item = per_case_stats[case_label]
        rmse = item["rmse_mean"]
        lines.append(
            f"- `{case_label}`: rmse_mean span=`{rmse['span']:.9g}`, std=`{rmse['std']:.9g}`, "
            f"unique rollout hashes=`{item['unique_rollout_npz_sha256']}`, "
            f"unique rmse curve hashes=`{item['unique_rmse_curve_sha256']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _ranking_markdown(
    *,
    ranking_strings: list[str],
    per_case_stats: dict[str, Any],
    verdict: str,
) -> str:
    lines = [
        "# Ranking Stability Report",
        "",
        f"- final verdict: `{verdict}`",
        f"- unique rankings observed: `{len(set(ranking_strings))}`",
        "",
    ]
    for idx, ranking in enumerate(ranking_strings, start=1):
        lines.append(f"- run_{idx:02d}: `{ranking}`")
    lines.extend(["", "## Bitwise Stability", ""])
    for case_label in CASE_LABELS:
        item = per_case_stats[case_label]
        lines.append(
            f"- `{case_label}`: rollout bitwise stable=`{item['unique_rollout_npz_sha256'] == 1}`, "
            f"rmse curve bitwise stable=`{item['unique_rmse_curve_sha256'] == 1}`"
        )
    lines.append("")
    return "\n".join(lines)


def _before_after_markdown(before: dict[str, Any] | None, per_case_stats: dict[str, Any], ranking_strings: list[str]) -> str:
    lines = [
        "# Before / After Compare",
        "",
    ]
    if before is not None:
        lines.extend(
            [
                "## Before Fix",
                "",
                f"- case 3 rmse_mean range: `{before['case3_rmse_mean_range']}`",
                f"- case 4 rmse_mean range: `{before['case4_rmse_mean_range']}`",
                f"- ranking stable: `{before['ranking_stable']}`",
                "",
            ]
        )
    lines.extend(["## After Fix", ""])
    for case_label in ("case_3_self_phystwin_ground_native", "case_4_self_phystwin_ground_phystwin"):
        stats = per_case_stats[case_label]["rmse_mean"]
        lines.append(
            f"- `{case_label}` rmse_mean range: `[{stats['min']:.12g}, {stats['max']:.12g}]` "
            f"(span `{stats['span']:.9g}`)"
        )
    lines.append(f"- ranking set: `{sorted(set(ranking_strings))}`")
    lines.append("")
    return "\n".join(lines)


def _repro_plan_markdown(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Reproducibility Plan",
            "",
            "1. Fix deterministic environment variables at the top-level runner.",
            "2. Reuse the canonical controlled 2x2 matrix runner with path-explicit outputs.",
            "3. Repeat the full 302-frame matrix at least five times.",
            "4. Capture ranking, per-case metrics, rollout hashes, and environment snapshots.",
            "5. Compare against the prior instability evidence from the case3-vs-case4 diagnosis root.",
            "",
            "Determinism env:",
            *(f"- `{k}={v}`" for k, v in DETERMINISM_ENV.items()),
            "",
            f"Repeats requested: `{args.repeats}`",
            f"IR: `{args.ir.resolve()}`",
            f"Device: `{args.device}`",
            "",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_index",
        "case_label",
        "self_collision_law",
        "ground_contact_law",
        "x0_rmse",
        "rmse_mean",
        "rmse_max",
        "first30_rmse",
        "last30_rmse",
        "dt",
        "substeps",
        "frame_count",
        "ranking_position",
        "ranking_string",
        "rollout_npz_sha256",
        "rmse_curve_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> int:
    args = parse_args()
    if args.repeats < 5:
        raise ValueError(f"repeats must be at least 5, got {args.repeats}")

    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _determinism_env()

    root_cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--ir",
        str(args.ir.resolve()),
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--num-frames",
        str(int(args.num_frames)),
        "--repeats",
        str(int(args.repeats)),
    ]
    if args.substeps_per_frame is not None:
        root_cmd.extend(["--substeps-per-frame", str(int(args.substeps_per_frame))])
    if args.sim_dt is not None:
        root_cmd.extend(["--sim-dt", str(float(args.sim_dt))])

    _write_text(out_dir / "README.md", "# Ground Contact / Self-Collision Repro Fix Audit\n\nRepeated full-matrix reruns under a fixed deterministic environment.\n")
    _write_text(out_dir / "reproducibility_plan.md", _repro_plan_markdown(args))
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
    ]
    for key, value in DETERMINISM_ENV.items():
        script_lines.append(f"export {key}={shlex.quote(value)}")
    script_lines.append(_quote_cmd(root_cmd))
    _write_text(out_dir / "command.sh", "\n".join(script_lines) + "\n")
    os.chmod(out_dir / "command.sh", 0o775)

    env_snapshot = {
        "captured_at": datetime.now().isoformat(),
        "git_sha": _resolve_git_sha(),
        "hostname": socket.gethostname(),
        "determinism_env": DETERMINISM_ENV,
        "python_env": _python_env_snapshot(str(args.python), str(args.device), env),
        "gpu": _gpu_snapshot(),
    }
    _write_json(out_dir / "environment_snapshot.json", env_snapshot)

    run_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    ranking_strings: list[str] = []
    fairness_passes: list[bool] = []

    for run_index in range(1, args.repeats + 1):
        run_dir = out_dir / "per_run" / f"run_{run_index:02d}"
        matrix_dir = run_dir / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        matrix_cmd = _build_matrix_command(args, matrix_dir)
        proc = _run_logged(
            matrix_cmd,
            env=env,
            command_path=run_dir / "command.sh",
            log_path=run_dir / "run.log",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"matrix runner failed for run_{run_index:02d} with exit code {proc.returncode}")

        rows = _load_rows(matrix_dir)
        fairness = _load_json(matrix_dir / "summary.json")["fairness_check_passed"]
        ranking = _ranking(rows)
        ranking_string = " > ".join(ranking)
        ranking_strings.append(ranking_string)
        fairness_passes.append(bool(fairness))
        run_rows.extend(_collect_run_rows(run_index, rows, ranking))

        run_summary = {
            "run_index": run_index,
            "matrix_root": str(matrix_dir),
            "ranking": ranking,
            "ranking_string": ranking_string,
            "fairness_check_passed": bool(fairness),
        }
        run_summaries.append(run_summary)
        _write_json(run_dir / "run_summary.json", run_summary)
        _write_json(run_dir / "environment_snapshot.json", env_snapshot)

    per_case_stats: dict[str, Any] = {}
    for case_label in CASE_LABELS:
        items = [row for row in run_rows if row["case_label"] == case_label]
        per_case_stats[case_label] = {
            "rmse_mean": _drift_stats([float(item["rmse_mean"]) for item in items]),
            "rmse_max": _drift_stats([float(item["rmse_max"]) for item in items]),
            "first30_rmse": _drift_stats([float(item["first30_rmse"]) for item in items]),
            "last30_rmse": _drift_stats([float(item["last30_rmse"]) for item in items]),
            "x0_rmse": _drift_stats([float(item["x0_rmse"]) for item in items]),
            "unique_rollout_npz_sha256": len({item["rollout_npz_sha256"] for item in items}),
            "unique_rmse_curve_sha256": len({item["rmse_curve_sha256"] for item in items}),
        }

    ranking_invariant = len(set(ranking_strings)) == 1
    metric_bitwise_stable = all(
        item["unique_rollout_npz_sha256"] == 1 and item["unique_rmse_curve_sha256"] == 1
        for item in per_case_stats.values()
    )

    residual_metric_drift = {
        case_label: {
            metric: stats["span"]
            for metric, stats in case_stats.items()
            if isinstance(stats, dict) and "span" in stats
        }
        for case_label, case_stats in per_case_stats.items()
    }

    if ranking_invariant:
        verdict = "REPRODUCIBLE"
    elif all(item["rmse_mean"]["span"] <= 1.0e-9 for item in per_case_stats.values()):
        verdict = "METRICS_STABLE_BUT_RANKING_UNSTABLE"
    else:
        verdict = "NONDETERMINISTIC"

    summary_csv = out_dir / "repro_audit_summary.csv"
    _write_csv(summary_csv, run_rows)
    _write_text(
        out_dir / "repro_audit_summary.md",
        _summary_markdown(
            ranking_strings=ranking_strings,
            fairness_passes=fairness_passes,
            per_case_stats=per_case_stats,
            verdict=verdict,
        ),
    )
    _write_text(
        out_dir / "ranking_stability_report.md",
        _ranking_markdown(
            ranking_strings=ranking_strings,
            per_case_stats=per_case_stats,
            verdict=verdict,
        ),
    )
    _write_text(
        out_dir / "before_after_compare.md",
        _before_after_markdown(_before_payload(args.before_root.resolve()), per_case_stats, ranking_strings),
    )
    _write_text(
        out_dir / "changed_files.md",
        "\n".join(
            [
                "# Changed Files",
                "",
                "- `Newton/phystwin_bridge/demos/self_contact_bridge_kernels.py`",
                "- `Newton/phystwin_bridge/tools/core/phystwin_contact_stack.py`",
                "- `Newton/phystwin_bridge/tools/other/run_ground_contact_self_collision_repro_audit.py`",
                "",
            ]
        ),
    )

    final_payload = {
        "verdict": verdict,
        "ranking_invariant": ranking_invariant,
        "metric_bitwise_stable": metric_bitwise_stable,
        "residual_metric_drift": residual_metric_drift,
        "ranking_strings": ranking_strings,
        "fairness_passes": fairness_passes,
        "per_case_stats": per_case_stats,
        "determinism_env": DETERMINISM_ENV,
        "before_root": str(args.before_root.resolve()),
        "run_summaries": run_summaries,
    }
    _write_json(out_dir / "final_verdict.json", final_payload)
    _write_json(out_dir / "summary.json", final_payload)
    _write_json(
        out_dir / "manifest.json",
        {
            "run_type": "ground_contact_self_collision_repro_audit",
            "created_at": datetime.now().isoformat(),
            "runner": str(Path(__file__).resolve()),
            "workspace_root": str(WORKSPACE_ROOT),
            "ir_path": str(args.ir.resolve()),
            "python": str(args.python),
            "device": str(args.device),
            "repeats": int(args.repeats),
            "determinism_env": DETERMINISM_ENV,
            "artifacts": {
                "readme": str(out_dir / "README.md"),
                "command": str(out_dir / "command.sh"),
                "reproducibility_plan": str(out_dir / "reproducibility_plan.md"),
                "summary_csv": str(summary_csv),
                "summary_md": str(out_dir / "repro_audit_summary.md"),
                "ranking_report": str(out_dir / "ranking_stability_report.md"),
                "before_after_compare": str(out_dir / "before_after_compare.md"),
                "final_verdict": str(out_dir / "final_verdict.json"),
            },
        },
    )

    validator_proc = subprocess.run(
        [str(args.python), str(ARTIFACT_VALIDATOR), str(out_dir)],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    _write_text(out_dir / "artifact_validation.log", validator_proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
