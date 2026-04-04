#!/usr/bin/env python3
"""Run the controlled 2x2 self-collision / ground-contact RMSE matrix.

This runner keeps the cloth scene, IR, dt/substeps, export cadence, and parity
evaluator fixed while independently varying only:

- self-collision law: off | phystwin
- ground-contact law: native | phystwin
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_defaults import default_device, default_python  # noqa: E402


VALIDATE_PARITY = CORE_DIR / "validate_parity.py"
DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
RESULTS_ROOT = BRIDGE_ROOT / "results"
ARTIFACT_VALIDATOR = SCRIPTS_DIR / "validate_experiment_artifacts.py"

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
    self_collision_law: str
    ground_contact_law: str


CASE_SPECS = (
    CaseSpec("case_1_self_off_ground_native", "off", "native"),
    CaseSpec("case_2_self_off_ground_phystwin", "off", "phystwin"),
    CaseSpec("case_3_self_phystwin_ground_native", "phystwin", "native"),
    CaseSpec("case_4_self_phystwin_ground_phystwin", "phystwin", "phystwin"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled cloth+ground 2x2 RMSE matrix."
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-frames", type=int, default=302)
    parser.add_argument("--substeps-per-frame", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


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
    return RESULTS_ROOT / f"ground_contact_self_collision_rmse_matrix_{stamp}_{_resolve_git_sha()}"


def _build_case_command(args: argparse.Namespace, case: CaseSpec, case_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(VALIDATE_PARITY),
        "--ir",
        str(args.ir.resolve()),
        "--out-dir",
        str(case_dir.resolve()),
        "--output-prefix",
        case.label,
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--num-frames",
        str(int(args.num_frames)),
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
    if args.substeps_per_frame is not None:
        cmd.extend(["--substeps-per-frame", str(int(args.substeps_per_frame))])
    if args.sim_dt is not None:
        cmd.extend(["--sim-dt", str(float(args.sim_dt))])
    return cmd


def _case_readme(case: CaseSpec, report_path: Path, proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        [
            f"# {case.label}",
            "",
            f"- self_collision_law: `{case.self_collision_law}`",
            f"- ground_contact_law: `{case.ground_contact_law}`",
            f"- report: `{report_path}`",
            f"- command: `{report_path.parent / 'command.sh'}`",
            f"- log: `{report_path.parent / 'run.log'}`",
            f"- validate_parity_exit_code: `{proc.returncode}`",
            "",
        ]
    )


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
        "strict_thresholds_used": {
            "x0_rmse": checks["x0_rmse"]["threshold"],
            "rmse_mean": checks["rmse_mean"]["threshold"],
            "rmse_max": checks["rmse_max"]["threshold"],
            "first30_rmse": checks["first30_rmse"]["threshold"],
            "last30_rmse": checks["last30_rmse"]["threshold"],
        },
        "frame_count": int(simulation["frames_run"]),
        "dt": float(simulation["sim_dt"]),
        "substeps": int(simulation["substeps_per_frame"]),
        "ir_path": report["ir_path"],
        "importer_command": report["importer_command"],
        "validate_parity_exit_code": int(proc.returncode),
        "rollout_completed_without_nan_inf": bool(simulation["all_particle_positions_finite"]),
        "rollout_report_json": str(Path(report["rollout_json"]).with_name(Path(report["rollout_json"]).stem + "_rollout_report.json")),
        "parity_report_json": str(Path(report["rollout_json"]).with_name(Path(report["rollout_json"]).stem + "_rollout_report.json")),
        "rollout_json": report["rollout_json"],
        "rollout_npz": report["rollout_npz"],
        "rmse_curve_csv": report["rmse_curve_csv"],
    }


def _check_same(rows: list[dict[str, Any]], key: str) -> tuple[bool, list[Any]]:
    values = [row[key] for row in rows]
    baseline = values[0]
    same = all(value == baseline for value in values[1:])
    return same, values


def _fairness_payload(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key in ("ir_path", "frame_count", "dt", "substeps"):
        same, values = _check_same(rows, key)
        checks[key] = {"pass": same, "values": values}

    config_keys = (
        "spring_ke_scale",
        "spring_kd_scale",
        "angular_damping",
        "friction_smoothing",
        "enable_tri_contact",
        "interpolate_controls",
        "strict_physics_checks",
        "apply_drag",
        "drag_damping_scale",
        "shape_contacts",
        "add_ground_plane",
        "disable_particle_contact_kernel",
        "ground_restitution_mode",
        "phystwin_freeze_collision_table",
        "phystwin_collision_table_capacity",
    )
    rollout_configs = []
    for row in rows:
        report = _load_json(Path(row["parity_report_json"]))
        rollout_configs.append(report["rollout_summary"]["config"])
    for key in config_keys:
        values = [cfg[key] for cfg in rollout_configs]
        checks[f"config.{key}"] = {"pass": all(v == values[0] for v in values[1:]), "values": values}

    law_labels_match = all(
        row["self_collision_law"] == row["actual_self_collision_law"]
        and row["ground_contact_law"] == row["actual_ground_contact_law"]
        for row in rows
    )
    checks["law_labels_match_actual_paths"] = {"pass": law_labels_match}

    only_axes_differ = all(
        item["pass"] for item in checks.values()
    )
    return {
        "pass": bool(only_axes_differ),
        "scene": "blue_cloth_double_lift_around implicit z=0 ground",
        "intended_axes": {
            "self_collision_law": ["off", "phystwin"],
            "ground_contact_law": ["native", "phystwin"],
        },
        "checks": checks,
        "notes": [
            "All four cases use the same strict IR, same frame count, same dt/substeps, same drag setting, same export cadence, and the same validate_parity evaluator.",
            "The only intended semantic differences are self-collision law and ground-contact law.",
        ],
    }


def _fairness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# fairness_check",
        "",
        f"- pass: `{payload['pass']}`",
        f"- scene: `{payload['scene']}`",
        "",
        "## Checks",
        "",
    ]
    for key, item in payload["checks"].items():
        lines.append(f"- `{key}`: pass=`{item['pass']}`")
        if "values" in item:
            lines.append(f"  values=`{item['values']}`")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in payload["notes"]])
    lines.append("")
    return "\n".join(lines)


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_label",
        "self_collision_law",
        "ground_contact_law",
        "actual_self_collision_law",
        "actual_ground_contact_law",
        "x0_rmse",
        "rmse_mean",
        "rmse_max",
        "first30_rmse",
        "last30_rmse",
        "passed_strict_gate",
        "frame_count",
        "dt",
        "substeps",
        "ir_path",
        "rollout_completed_without_nan_inf",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.9g}"


def _find_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row["case_label"] == label:
            return row
    raise KeyError(label)


def _matrix_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c1 = _find_row(rows, "case_1_self_off_ground_native")
    c2 = _find_row(rows, "case_2_self_off_ground_phystwin")
    c3 = _find_row(rows, "case_3_self_phystwin_ground_native")
    c4 = _find_row(rows, "case_4_self_phystwin_ground_phystwin")

    q1 = float(c2["rmse_mean"] - c1["rmse_mean"])
    q2 = float(c3["rmse_mean"] - c1["rmse_mean"])
    q3 = float(c4["rmse_mean"] - c2["rmse_mean"])
    ground_effect_with_self_on = float(c4["rmse_mean"] - c3["rmse_mean"])
    self_effect_avg_abs = 0.5 * (abs(q2) + abs(q3))
    ground_effect_avg_abs = 0.5 * (abs(q1) + abs(ground_effect_with_self_on))
    interaction = float((c4["rmse_mean"] - c2["rmse_mean"]) - (c3["rmse_mean"] - c1["rmse_mean"]))

    if self_effect_avg_abs > ground_effect_avg_abs:
        dominant_axis = "self_collision_law"
    elif ground_effect_avg_abs > self_effect_avg_abs:
        dominant_axis = "ground_contact_law"
    else:
        dominant_axis = "tie"

    best = min(rows, key=lambda row: float(row["rmse_mean"]))
    return {
        "q1_ground_switch_with_self_off": q1,
        "q2_self_switch_with_ground_native": q2,
        "q3_self_switch_with_ground_phystwin": q3,
        "q4_axis_contribution_more": dominant_axis,
        "ground_effect_with_self_on": ground_effect_with_self_on,
        "self_effect_avg_abs": self_effect_avg_abs,
        "ground_effect_avg_abs": ground_effect_avg_abs,
        "q5_interaction_effect_rmse_mean": interaction,
        "best_case_by_rmse_mean": {
            "case_label": best["case_label"],
            "rmse_mean": best["rmse_mean"],
        },
    }


def _matrix_markdown(rows: list[dict[str, Any]], fairness: dict[str, Any], analysis: dict[str, Any]) -> str:
    header = [
        "# Controlled 2x2 RMSE Matrix",
        "",
        "| case | self law | ground law | x0_rmse | rmse_mean | rmse_max | first30_rmse | last30_rmse | strict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        header.append(
            "| {case_label} | {self_collision_law} | {ground_contact_law} | {x0} | {mean} | {maxv} | {first30} | {last30} | {strict} |".format(
                case_label=row["case_label"],
                self_collision_law=row["self_collision_law"],
                ground_contact_law=row["ground_contact_law"],
                x0=_fmt(row["x0_rmse"]),
                mean=_fmt(row["rmse_mean"]),
                maxv=_fmt(row["rmse_max"]),
                first30=_fmt(row["first30_rmse"]),
                last30=_fmt(row["last30_rmse"]),
                strict=row["passed_strict_gate"],
            )
        )

    lines = header + [
        "",
        "## Scientific Questions",
        "",
        f"- Q1: with self OFF, switching ground native -> phystwin changes `rmse_mean` by `{analysis['q1_ground_switch_with_self_off']:.9g}`.",
        f"- Q2: with ground native fixed, switching self OFF -> phystwin changes `rmse_mean` by `{analysis['q2_self_switch_with_ground_native']:.9g}`.",
        f"- Q3: with ground phystwin fixed, switching self OFF -> phystwin changes `rmse_mean` by `{analysis['q3_self_switch_with_ground_phystwin']:.9g}`.",
        f"- Q4: larger average absolute contribution currently comes from `{analysis['q4_axis_contribution_more']}`.",
        f"- Q5: interaction effect (difference-in-differences on `rmse_mean`) is `{analysis['q5_interaction_effect_rmse_mean']:.9g}`.",
        "",
        "## Fairness",
        "",
        f"- fair controlled comparison: `{fairness['pass']}`",
        f"- best case by `rmse_mean`: `{analysis['best_case_by_rmse_mean']['case_label']}` with `{analysis['best_case_by_rmse_mean']['rmse_mean']:.9g}`",
        "",
    ]
    return "\n".join(lines)


def _root_readme(args: argparse.Namespace, out_dir: Path) -> str:
    return "\n".join(
        [
            "# Ground Contact / Self-Collision RMSE Matrix",
            "",
            "- scene: `blue_cloth_double_lift_around` with implicit `z=0` ground",
            f"- ir: `{args.ir.resolve()}`",
            f"- frames: `{int(args.num_frames)}`",
            f"- substeps_per_frame override: `{args.substeps_per_frame}`",
            f"- sim_dt override: `{args.sim_dt}`",
            "- fixed across all cases:",
            "  - same IR / initial state",
            "  - same validate_parity path",
            "  - same drag setting (`apply_drag=true`)",
            "  - same `enable_tri_contact=false`",
            "  - same `shape_contacts=false` and `add_ground_plane=true`",
            "- intended varying axes:",
            "  - `self_collision_law`: `off | phystwin`",
            "  - `ground_contact_law`: `native | phystwin`",
            "",
        ]
    )


def _root_history_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# sim/history",
        "",
        "This matrix root aggregates four controlled rollout cases. The actual rollout",
        "arrays live in the case subdirectories listed below.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['case_label']}`: `{row['rollout_npz']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_log_lines = [
        f"run_started_at={datetime.now().isoformat()}",
        f"workspace_root={WORKSPACE_ROOT}",
        f"ir={args.ir.resolve()}",
        f"python={args.python}",
        f"device={args.device}",
    ]

    root_command = [
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
    ]
    if args.substeps_per_frame is not None:
        root_command.extend(["--substeps-per-frame", str(int(args.substeps_per_frame))])
    if args.sim_dt is not None:
        root_command.extend(["--sim-dt", str(float(args.sim_dt))])

    _write_text(out_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(root_command) + "\n")
    os.chmod(out_dir / "command.sh", 0o775)
    _write_text(out_dir / "README.md", _root_readme(args, out_dir))

    rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    for case in CASE_SPECS:
        case_dir = out_dir / case.label
        case_dir.mkdir(parents=True, exist_ok=True)
        report_path = case_dir / f"{case.label}_rollout_report.json"
        if args.skip_existing and report_path.exists():
            proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
        else:
            cmd = _build_case_command(args, case, case_dir)
            proc = _run_logged(
                cmd,
                workdir=WORKSPACE_ROOT,
                command_path=case_dir / "command.sh",
                log_path=case_dir / "run.log",
            )
            matrix_log_lines.append(f"{case.label}: validate_parity_exit_code={proc.returncode}")

        if not report_path.exists():
            raise FileNotFoundError(f"Missing report for {case.label}: {report_path}")

        report = _load_json(report_path)
        row = _collect_case_row(case, report, proc)
        rows.append(row)
        case_records.append(
            {
                "case_label": case.label,
                "case_dir": str(case_dir),
                "validate_parity_exit_code": int(proc.returncode),
                "report_json": str(report_path),
            }
        )
        _write_text(case_dir / "README.md", _case_readme(case, report_path, proc))

    fairness = _fairness_payload(args, rows)
    fairness_md_path = out_dir / "fairness_check.md"
    _write_text(fairness_md_path, _fairness_markdown(fairness))

    rmse_csv_path = out_dir / "rmse_matrix.csv"
    _write_matrix_csv(rmse_csv_path, rows)

    analysis = _matrix_analysis(rows)
    rmse_md_path = out_dir / "rmse_matrix.md"
    _write_text(rmse_md_path, _matrix_markdown(rows, fairness, analysis))

    summary_json_path = out_dir / "rmse_matrix_summary.json"
    summary_payload = {
        "scene": "blue_cloth_double_lift_around implicit z=0 ground",
        "strict_thresholds": STRICT_THRESHOLDS,
        "fairness_check_passed": fairness["pass"],
        "analysis": analysis,
        "rows": rows,
    }
    _write_json(
        summary_json_path,
        summary_payload,
    )
    _write_json(out_dir / "summary.json", summary_payload)
    _write_text(out_dir / "sim" / "history" / "README.md", _root_history_readme(rows))

    manifest_path = out_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "run_type": "ground_contact_self_collision_rmse_matrix",
            "created_at": datetime.now().isoformat(),
            "runner": str(Path(__file__).resolve()),
            "workspace_root": str(WORKSPACE_ROOT),
            "ir_path": str(args.ir.resolve()),
            "python": str(args.python),
            "device": str(args.device),
            "case_labels": [case.label for case in CASE_SPECS],
            "cases": case_records,
            "artifacts": {
                "readme": str(out_dir / "README.md"),
                "command": str(out_dir / "command.sh"),
                "matrix_runner_log": str(out_dir / "matrix_runner.log"),
                "fairness_check": str(fairness_md_path),
                "rmse_matrix_csv": str(rmse_csv_path),
                "rmse_matrix_md": str(rmse_md_path),
                "rmse_matrix_summary_json": str(summary_json_path),
            },
        },
    )

    validator_log_path = out_dir / "artifact_validation.log"
    validator_proc = subprocess.run(
        [str(args.python), str(ARTIFACT_VALIDATOR), str(out_dir)],
        cwd=str(WORKSPACE_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(validator_log_path, validator_proc.stdout)
    matrix_log_lines.append(f"artifact_validator_exit_code={validator_proc.returncode}")
    _write_text(out_dir / "matrix_runner.log", "\n".join(matrix_log_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
