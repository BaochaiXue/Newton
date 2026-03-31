#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"


def _load_core_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_defaults = _load_core_module("final_campaign_path_defaults", CORE_DIR / "path_defaults.py")

DEFAULT_MATRIX_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
DEFAULT_PARITY_IR = BRIDGE_ROOT / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz"
RUN_SELF_COLLISION_MATRIX = BRIDGE_ROOT / "tools" / "other" / "run_self_collision_matrix.py"
VERIFY_EQUIVALENCE = BRIDGE_ROOT / "tools" / "other" / "verify_phystwin_self_collision_equivalence.py"
VALIDATE_PARITY = BRIDGE_ROOT / "tools" / "core" / "validate_parity.py"
RENDER_PARITY_VIDEO = BRIDGE_ROOT / "tools" / "other" / "render_comparison_2x3_mp4.py"
VALIDATE_BRIDGE_VIDEO_QC = SCRIPTS_DIR / "validate_bridge_video_qc.py"
RUN_BUNNY_DIAG = SCRIPTS_DIR / "run_bunny_force_diag.sh"
RUN_PROFILE = SCRIPTS_DIR / "run_realtime_profile.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final bridge-side self-collision campaign.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=path_defaults.default_python())
    parser.add_argument("--device", default=path_defaults.default_device())
    parser.add_argument("--matrix-ir", type=Path, default=DEFAULT_MATRIX_IR)
    parser.add_argument("--parity-ir", type=Path, default=DEFAULT_PARITY_IR)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_logged(cmd: list[str], *, workdir: Path, command_path: Path, log_path: Path) -> None:
    _write_text(command_path, "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(cmd) + "\n")
    os.chmod(command_path, 0o775)
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_text(log_path, proc.stdout)


def _find_unique(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern!r} under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one match for {pattern!r} under {root}, got {matches}")
    return matches[0]


def _first_mp4(root: Path) -> Path:
    matches = sorted(root.rglob("*.mp4"))
    if not matches:
        raise FileNotFoundError(f"No mp4 found under {root}")
    return matches[0]


def _resolve_final_demo_candidate(
    campaign_root: Path,
    *,
    fallback_mp4: Path,
    fallback_summary: Path,
) -> tuple[Path, Path]:
    candidates = [
        (
            campaign_root
            / "selected_candidates"
            / "phystwin_topdown"
            / "self_phystwin"
            / "cloth_box_final_phystwin_m10.mp4",
            campaign_root
            / "selected_candidates"
            / "phystwin_topdown"
            / "self_phystwin"
            / "cloth_box_final_phystwin_m10_summary.json",
        ),
        (
            campaign_root
            / "selected_candidates"
            / "phystwin_camclose_a"
            / "self_phystwin"
            / "cloth_box_final_phystwin_m10.mp4",
            campaign_root
            / "selected_candidates"
            / "phystwin_camclose_a"
            / "self_phystwin"
            / "cloth_box_final_phystwin_m10_summary.json",
        ),
        (fallback_mp4, fallback_summary),
    ]
    for mp4_path, summary_path in candidates:
        if mp4_path.exists() and summary_path.exists():
            return mp4_path, summary_path
    raise FileNotFoundError("No final cloth-box phystwin demo candidate found.")


def _version_payload(python_bin: str) -> dict[str, Any]:
    payload = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device_gpu": None,
        "warp_version": None,
        "numpy_version": None,
        "scipy_version": None,
    }
    try:
        payload["device_gpu"] = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .splitlines()[0]
            .strip()
        )
    except Exception:
        payload["device_gpu"] = None

    probe = (
        "import json, numpy, scipy\n"
        "out={'numpy_version': numpy.__version__, 'scipy_version': scipy.__version__}\n"
        "try:\n"
        " import warp as wp\n"
        " out['warp_version']=getattr(wp,'__version__',None)\n"
        "except Exception:\n"
        " out['warp_version']=None\n"
        "print(json.dumps(out))\n"
    )
    try:
        out = subprocess.check_output([str(python_bin), "-c", probe], text=True).strip()
        payload.update(json.loads(out))
    except Exception:
        pass
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _write_readme(path: Path, title: str, bullets: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.extend([f"- {item}" for item in bullets])
    _write_text(path, "\n".join(lines) + "\n")


def _run_matrix(args: argparse.Namespace, campaign_root: Path) -> tuple[Path, Path, Path]:
    out_dir = campaign_root / "matrix"
    csv_path = out_dir / "self_collision_decision_table.csv"
    if not (args.skip_existing and csv_path.exists()):
        cmd = [
            str(args.python),
            str(RUN_SELF_COLLISION_MATRIX),
            "--out-dir",
            str(out_dir),
            "--python",
            str(args.python),
            "--device",
            str(args.device),
        ]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "campaign_command.sh",
            log_path=out_dir / "campaign_run.log",
        )
    phystwin_dir = out_dir / "phystwin"
    summary_path = _find_unique(phystwin_dir, "*_summary.json")
    mp4_path = _first_mp4(phystwin_dir)
    _write_readme(
        out_dir / "README.md",
        "Matrix Outputs",
        [
            f"decision csv: {csv_path}",
            f"phystwin summary: {summary_path}",
            f"phystwin video: {mp4_path}",
            "The matrix runner now writes only a provisional selected mp4 name; final selection happens after QC.",
        ],
    )
    return out_dir, summary_path, mp4_path


def _run_equivalence(args: argparse.Namespace, campaign_root: Path) -> Path:
    out_dir = campaign_root / "equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "verify_phystwin_self_collision_equivalence.json"
    if not (args.skip_existing and out_json.exists()):
        cmd = [
            str(args.python),
            str(VERIFY_EQUIVALENCE),
            "--out",
            str(out_json),
            "--device",
            str(args.device),
            "--num-cases",
            "10000",
            "--tol-max-abs-dv",
            "1e-5",
            "--tol-median-rel-dv",
            "1e-4",
        ]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "command.sh",
            log_path=out_dir / "run.log",
        )
    payload = _load_json(out_json)
    _write_readme(
        out_dir / "README.md",
        "PhysTwin Exactness",
        [
            f"pass: {payload.get('pass')}",
            f"max_abs_dv: {payload.get('max_abs_dv')}",
            f"median_rel_dv: {payload.get('median_rel_dv')}",
        ],
    )
    return out_json


def _run_bunny_diag(campaign_root: Path) -> Path:
    out_dir = campaign_root / "native_failure" / "bunny_diagnostic"
    summary_path = out_dir / "self_off" / "force_diagnostic" / "force_diag_trigger_summary.json"
    if not summary_path.exists():
        cmd = [str(RUN_BUNNY_DIAG), str(out_dir)]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "campaign_command.sh",
            log_path=out_dir / "campaign_run.log",
        )
    return summary_path


def _run_profiler(campaign_root: Path) -> tuple[Path, Path]:
    out_dir = campaign_root / "native_failure" / "profiler"
    json_path = out_dir / "cloth_bunny_playground_profile.json"
    csv_path = out_dir / "cloth_bunny_playground_profile.csv"
    if not json_path.exists():
        cmd = [str(RUN_PROFILE), str(out_dir), "--mode", "off", "--rigid-shape", "bunny"]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "campaign_command.sh",
            log_path=out_dir / "campaign_run.log",
        )
    return json_path, csv_path


def _run_strict_parity(args: argparse.Namespace, campaign_root: Path) -> tuple[Path, Path]:
    out_dir = campaign_root / "parity" / "rope_strict"
    report_path = out_dir / "strict_parity_rollout_report.json"
    if not (args.skip_existing and report_path.exists()):
        cmd = [
            str(args.python),
            str(VALIDATE_PARITY),
            "--ir",
            str(args.parity_ir.resolve()),
            "--out-dir",
            str(out_dir),
            "--output-prefix",
            "strict_parity",
            "--strict-physics-checks",
            "--interpolate-controls",
            "--apply-drag",
            "--add-ground-plane",
            "--no-shape-contacts",
            "--no-particle-contacts",
            "--disable-particle-contact-kernel",
            "--max-x0-rmse",
            "1e-6",
            "--max-rmse-mean",
            "1e-5",
            "--max-rmse-max",
            "5e-5",
            "--max-first30-rmse",
            "5e-5",
            "--max-last30-rmse",
            "5e-5",
        ]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "command.sh",
            log_path=out_dir / "run.log",
        )
    payload = _load_json(report_path)
    checks = payload["checks"]
    strict_summary = {
        "passed": bool(payload["passed"]),
        "case_name": str(payload["case_name"]),
        "report_json": str(report_path),
        "rmse_mean": checks["rmse_mean"]["value"],
        "x0_rmse": checks["x0_rmse"]["value"],
        "rmse_max": checks["rmse_max"]["value"],
        "first30_rmse": checks["first30_rmse"]["value"],
        "last30_rmse": checks["last30_rmse"]["value"],
        "thresholds": {
            "max_x0_rmse": checks["x0_rmse"]["threshold"],
            "max_rmse_mean": checks["rmse_mean"]["threshold"],
            "max_rmse_max": checks["rmse_max"]["threshold"],
            "max_first30_rmse": checks["first30_rmse"]["threshold"],
            "max_last30_rmse": checks["last30_rmse"]["threshold"],
        },
    }
    summary_path = campaign_root / "parity" / "strict_parity_summary.json"
    _write_text(summary_path, json.dumps(strict_summary, indent=2) + "\n")
    report_md = campaign_root / "parity" / "strict_parity_report.md"
    _write_text(
        report_md,
        "\n".join(
            [
                "# Strict Parity Report",
                "",
                f"- passed: `{strict_summary['passed']}`",
                f"- case: `{strict_summary['case_name']}`",
                f"- x0_rmse: `{strict_summary['x0_rmse']}`",
                f"- rmse_mean: `{strict_summary['rmse_mean']}`",
                f"- rmse_max: `{strict_summary['rmse_max']}`",
                f"- first30_rmse: `{strict_summary['first30_rmse']}`",
                f"- last30_rmse: `{strict_summary['last30_rmse']}`",
                f"- source report: `{report_path}`",
                "",
            ]
        )
        + "\n",
    )
    return report_path, summary_path


def _render_parity_support(args: argparse.Namespace, campaign_root: Path, report_path: Path) -> Path:
    out_dir = campaign_root / "parity" / "support_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "strict_parity_cmp_2x3_labeled.mp4"
    if not (args.skip_existing and mp4_path.exists()):
        cmd = [
            str(args.python),
            str(RENDER_PARITY_VIDEO),
            "--report",
            str(report_path),
            "--out-mp4",
            str(mp4_path),
            "--newton-label",
            "Newton",
            "--phystwin-label",
            "PhysTwin",
        ]
        _run_logged(
            cmd,
            workdir=WORKSPACE_ROOT,
            command_path=out_dir / "command.sh",
            log_path=out_dir / "run.log",
        )
    _write_readme(out_dir / "README.md", "Parity Support Demo", [f"mp4: {mp4_path}", f"report: {report_path}"])
    return mp4_path


def _run_video_qc(
    *,
    video_path: Path,
    out_dir: Path,
    scene_profile: str,
    summary_json: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    cmd = [
        str(sys.executable),
        str(VALIDATE_BRIDGE_VIDEO_QC),
        "--video",
        str(video_path),
        "--out-dir",
        str(out_dir),
        "--scene-profile",
        scene_profile,
    ]
    if summary_json is not None:
        cmd.extend(["--summary-json", str(summary_json)])
    _run_logged(
        cmd,
        workdir=WORKSPACE_ROOT,
        command_path=out_dir / "command.sh",
        log_path=out_dir / "run.log",
    )
    qc_json = out_dir / "video_qc.json"
    return qc_json, _load_json(qc_json)


def _copy_selected(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_native_failure_readme(campaign_root: Path, matrix_dir: Path, bunny_summary: Path, profile_json: Path) -> None:
    report_path = campaign_root / "subagents" / "native_failure" / "report.md"
    readme = campaign_root / "native_failure" / "README.md"
    _write_text(
        readme,
        "\n".join(
            [
                "# Native Failure Evidence",
                "",
                f"- source evidence report: `{report_path}`",
                f"- controlled matrix: `{matrix_dir}`",
                f"- bunny diagnostic summary: `{bunny_summary}`",
                f"- profiler json: `{profile_json}`",
                "",
                "Conclusion target: native Newton collision / native self-collision are insufficient as the final claim; bridge-side `phystwin` remains the promoted exact path.",
                "",
            ]
        ),
    )


def _write_video_qc_summary(campaign_root: Path, cloth_payload: dict[str, Any], parity_payload: dict[str, Any]) -> Path:
    path = campaign_root / "video_qc" / "final_video_qc_report.md"
    _write_text(
        path,
        "\n".join(
            [
                "# Final Video QC",
                "",
                "## Cloth + Box + Self-Collision ON",
                f"- verdict: `{cloth_payload['verdict']}`",
                f"- json: `{campaign_root / 'video_qc' / 'cloth_box_phystwin' / 'video_qc.json'}`",
                f"- contact sheet: `{cloth_payload['contact_sheet']}`",
                "",
                "## Parity Support Demo",
                f"- verdict: `{parity_payload['verdict']}`",
                f"- json: `{campaign_root / 'video_qc' / 'parity_support' / 'video_qc.json'}`",
                f"- contact sheet: `{parity_payload['contact_sheet']}`",
                "",
            ]
        ),
    )
    return path


def _write_final_status(campaign_root: Path, manifest: dict[str, Any]) -> Path:
    path = campaign_root / "FINAL_STATUS.md"
    _write_text(
        path,
        "\n".join(
            [
                "# Final Campaign Status",
                "",
                f"- selected final mode: `{manifest.get('selected_final_mode')}`",
                f"- final acceptance: `{manifest.get('final_acceptance')}`",
                f"- strict parity summary: `{campaign_root / 'parity' / 'strict_parity_summary.json'}`",
                f"- exactness json: `{campaign_root / 'equivalence' / 'verify_phystwin_self_collision_equivalence.json'}`",
                f"- final demo: `{campaign_root / 'selected' / 'self_collision_on_cloth_box_phystwin.mp4'}`",
                f"- parity support demo: `{campaign_root / 'selected' / 'parity_support_demo.mp4'}`",
                f"- video qc report: `{campaign_root / 'video_qc' / 'final_video_qc_report.md'}`",
                "",
            ]
        ),
    )
    return path


def main() -> int:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    manifest_path = campaign_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest.update(_version_payload(str(args.python)))

    matrix_dir, phystwin_summary_path, phystwin_mp4 = _run_matrix(args, campaign_root)
    equivalence_json = _run_equivalence(args, campaign_root)
    bunny_summary = _run_bunny_diag(campaign_root)
    profile_json, _profile_csv = _run_profiler(campaign_root)
    parity_report_json, strict_parity_summary = _run_strict_parity(args, campaign_root)
    parity_support_mp4 = _render_parity_support(args, campaign_root, parity_report_json)
    cloth_demo_mp4, cloth_demo_summary = _resolve_final_demo_candidate(
        campaign_root,
        fallback_mp4=phystwin_mp4,
        fallback_summary=phystwin_summary_path,
    )

    cloth_qc_json, cloth_qc_payload = _run_video_qc(
        video_path=cloth_demo_mp4,
        out_dir=campaign_root / "video_qc" / "cloth_box_phystwin",
        scene_profile="cloth_box",
        summary_json=cloth_demo_summary,
    )
    parity_qc_json, parity_qc_payload = _run_video_qc(
        video_path=parity_support_mp4,
        out_dir=campaign_root / "video_qc" / "parity_support",
        scene_profile="parity_2x3",
        summary_json=None,
    )

    selected_dir = campaign_root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    if cloth_qc_payload["verdict"] == "PASS":
        _copy_selected(phystwin_mp4, selected_dir / "self_collision_on_cloth_box_phystwin.mp4")
    if parity_qc_payload["verdict"] == "PASS":
        _copy_selected(parity_support_mp4, selected_dir / "parity_support_demo.mp4")

    _write_native_failure_readme(campaign_root, matrix_dir, bunny_summary, profile_json)
    video_qc_report = _write_video_qc_summary(campaign_root, cloth_qc_payload, parity_qc_payload)

    parity_payload = _load_json(strict_parity_summary)
    equivalence_payload = _load_json(equivalence_json)

    final_acceptance = (
        "PASS"
        if (
            bool(equivalence_payload.get("pass"))
            and bool(parity_payload.get("passed"))
            and cloth_qc_payload.get("verdict") == "PASS"
            and parity_qc_payload.get("verdict") == "PASS"
            and (selected_dir / "self_collision_on_cloth_box_phystwin.mp4").exists()
            and (selected_dir / "parity_support_demo.mp4").exists()
        )
        else "PARTIAL"
    )
    manifest.update(
        {
            "campaign_finished_local": datetime.now().astimezone().isoformat(),
            "selected_final_mode": "phystwin",
            "final_acceptance": final_acceptance,
            "equivalence_json": str(equivalence_json),
            "strict_parity_summary": str(strict_parity_summary),
            "cloth_box_qc_json": str(cloth_qc_json),
            "parity_qc_json": str(parity_qc_json),
            "selected_demo": str(selected_dir / "self_collision_on_cloth_box_phystwin.mp4"),
            "selected_parity_demo": str(selected_dir / "parity_support_demo.mp4"),
            "video_qc_report": str(video_qc_report),
            "selected_demo_source": str(cloth_demo_mp4),
            "selected_demo_summary_source": str(cloth_demo_summary),
        }
    )
    _save_manifest(manifest_path, manifest)
    _write_final_status(campaign_root, manifest)

    print(json.dumps({"campaign_root": str(campaign_root), "final_acceptance": final_acceptance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
