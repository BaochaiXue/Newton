#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
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


path_defaults = _load_core_module("strict_self_collision_path_defaults", CORE_DIR / "path_defaults.py")

VALIDATE_PARITY = CORE_DIR / "validate_parity.py"
DEFAULT_IMPORTER = BRIDGE_ROOT / "tools" / "other" / "newton_import_ir_phystwin.py"
RENDER_COMPARISON = BRIDGE_ROOT / "tools" / "other" / "render_comparison_2x3_mp4.py"
VIDEO_QC = SCRIPTS_DIR / "validate_bridge_video_qc.py"
DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict self-collision parity on the blue-cloth PhysTwin case.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=path_defaults.default_python())
    parser.add_argument("--device", default=path_defaults.default_device())
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--importer", type=Path, default=DEFAULT_IMPORTER)
    parser.add_argument("--num-frames", type=int, default=302)
    parser.add_argument("--custom-self-contact-hops", type=int, default=0)
    parser.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ground-restitution-mode",
        choices=["strict-native", "approximate-native"],
        default="approximate-native",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_logged(
    cmd: list[str],
    *,
    workdir: Path,
    command_path: Path,
    log_path: Path,
    check: bool,
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
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {_quote_cmd(cmd)}")
    return proc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_thresholds() -> dict[str, float]:
    return {
        "x0_rmse": 1.0e-6,
        "rmse_mean": 1.0e-5,
        "rmse_max": 5.0e-5,
        "first30_rmse": 5.0e-5,
        "last30_rmse": 5.0e-5,
    }


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parity_dir = out_dir / "strict_self_collision"
    support_dir = out_dir / "support_demo"
    qc_dir = out_dir / "support_demo_qc"
    summary_path = out_dir / "strict_self_collision_parity_summary.json"
    report_md = out_dir / "strict_self_collision_parity_report.md"
    readme_path = out_dir / "README.md"

    thresholds = _strict_thresholds()
    report_json = parity_dir / "strict_self_collision_parity_rollout_report.json"
    if not (args.skip_existing and report_json.exists()):
        validate_cmd = [
            str(args.python),
            str(VALIDATE_PARITY),
            "--importer",
            str(args.importer.resolve()),
            "--ir",
            str(args.ir.resolve()),
            "--out-dir",
            str(parity_dir),
            "--output-prefix",
            "strict_self_collision_parity",
            "--device",
            str(args.device),
            "--num-frames",
            str(int(args.num_frames)),
            "--self-contact-mode",
            "phystwin",
            "--custom-self-contact-hops",
            str(int(args.custom_self_contact_hops)),
            "--interpolate-controls",
            "--apply-drag",
            "--ground-restitution-mode",
            str(args.ground_restitution_mode),
            "--disable-particle-contact-kernel",
            "--max-x0-rmse",
            str(thresholds["x0_rmse"]),
            "--max-rmse-mean",
            str(thresholds["rmse_mean"]),
            "--max-rmse-max",
            str(thresholds["rmse_max"]),
            "--max-first30-rmse",
            str(thresholds["first30_rmse"]),
            "--max-last30-rmse",
            str(thresholds["last30_rmse"]),
        ]
        validate_cmd.append("--shape-contacts" if args.shape_contacts else "--no-shape-contacts")
        validate_cmd.append("--add-ground-plane" if args.add_ground_plane else "--no-add-ground-plane")
        _run_logged(
            validate_cmd,
            workdir=WORKSPACE_ROOT,
            command_path=parity_dir / "command.sh",
            log_path=parity_dir / "run.log",
            check=False,
        )

    if not report_json.exists():
        raise FileNotFoundError(f"Missing parity report JSON: {report_json}")

    report = _load_json(report_json)

    support_mp4 = support_dir / "parity_support_demo.mp4"
    if not (args.skip_existing and support_mp4.exists()):
        render_cmd = [
            str(args.python),
            str(RENDER_COMPARISON),
            "--report",
            str(report_json),
            "--newton-label",
            "Newton phystwin",
            "--phystwin-label",
            "PhysTwin reference",
            "--out-mp4",
            str(support_mp4),
        ]
        _run_logged(
            render_cmd,
            workdir=WORKSPACE_ROOT,
            command_path=support_dir / "command.sh",
            log_path=support_dir / "run.log",
            check=True,
        )

    qc_json = qc_dir / "video_qc.json"
    if not (args.skip_existing and qc_json.exists()):
        qc_cmd = [
            str(args.python),
            str(VIDEO_QC),
            "--video",
            str(support_mp4),
            "--out-dir",
            str(qc_dir),
            "--scene-profile",
            "parity_2x3",
        ]
        _run_logged(
            qc_cmd,
            workdir=WORKSPACE_ROOT,
            command_path=qc_dir / "command.sh",
            log_path=qc_dir / "run.log",
            check=True,
        )

    qc_payload = _load_json(qc_json)
    checks = report["checks"]
    payload = {
        "passed": bool(report["passed"]) and bool(qc_payload["verdict"] == "PASS"),
        "case_name": report["case_name"],
        "report_json": str(report_json),
        "rmse_mean": checks["rmse_mean"]["value"],
        "x0_rmse": checks["x0_rmse"]["value"],
        "rmse_max": checks["rmse_max"]["value"],
        "first30_rmse": checks["first30_rmse"]["value"],
        "last30_rmse": checks["last30_rmse"]["value"],
        "thresholds": thresholds,
        "self_contact_mode": "phystwin",
        "custom_self_contact_hops": int(args.custom_self_contact_hops),
        "rollout_npz": report["rollout_npz"],
        "support_demo_mp4": str(support_mp4),
        "support_demo_qc_json": str(qc_json),
        "support_demo_qc_verdict": qc_payload["verdict"],
        "status": "PASS" if bool(report["passed"]) and bool(qc_payload["verdict"] == "PASS") else "BLOCKING_STRICT_GATE",
    }
    _write_text(summary_path, json.dumps(payload, indent=2) + "\n")

    report_lines = [
        "# Strict Self-Collision Parity Report",
        "",
        f"- status: `{payload['status']}`",
        f"- case: `{payload['case_name']}`",
        f"- self_contact_mode: `{payload['self_contact_mode']}`",
        f"- custom_self_contact_hops: `{payload['custom_self_contact_hops']}`",
        f"- x0_rmse: `{payload['x0_rmse']}` (threshold `{thresholds['x0_rmse']}`)",
        f"- rmse_mean: `{payload['rmse_mean']}` (threshold `{thresholds['rmse_mean']}`)",
        f"- rmse_max: `{payload['rmse_max']}` (threshold `{thresholds['rmse_max']}`)",
        f"- first30_rmse: `{payload['first30_rmse']}` (threshold `{thresholds['first30_rmse']}`)",
        f"- last30_rmse: `{payload['last30_rmse']}` (threshold `{thresholds['last30_rmse']}`)",
        f"- rollout report: `{report_json}`",
        f"- support demo: `{support_mp4}`",
        f"- support demo QC: `{qc_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines) + "\n")

    readme_lines = [
        "# Strict Self-Collision Parity",
        "",
        "- In-scope parity target: `blue_cloth_double_lift_around` PhysTwin self-collision case",
        "- Bridge-side self-contact mode: `phystwin`",
        "- Contact scope: pairwise self-collision + implicit `z=0` ground plane only",
        "- Summary JSON: `strict_self_collision_parity_summary.json`",
        "- Report: `strict_self_collision_parity_report.md`",
        "- Support demo: `support_demo/parity_support_demo.mp4`",
        "- QC JSON: `support_demo_qc/video_qc.json`",
    ]
    _write_text(readme_path, "\n".join(readme_lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
