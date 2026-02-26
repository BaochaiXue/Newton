#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from path_defaults import default_device, default_python


DEFAULT_THRESHOLDS = {
    "max_x0_rmse": 1e-6,
    "max_rmse_mean": 0.05,
    "max_rmse_max": 0.1,
    "max_first30_rmse": 0.1,
    "max_last30_rmse": 0.1,
}


def _default_importer() -> Path:
    return Path(__file__).resolve().with_name("newton_import_ir.py")


def _default_threshold_config() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "inputs" / "configs" / "parity_thresholds.yaml"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parity-mode rollout and validate consistency against PhysTwin inference."
    )
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="parity_validate")

    parser.add_argument("--importer", type=Path, default=_default_importer())
    parser.add_argument("--python", default=default_python())

    parser.add_argument("--mode", choices=["standard", "parity"], default="parity")
    parser.add_argument(
        "--solver",
        choices=["xpbd", "semi_implicit"],
        default="semi_implicit",
    )
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--xpbd-soft-body-relaxation", type=float, default=0.9)
    parser.add_argument("--xpbd-soft-contact-relaxation", type=float, default=0.9)
    parser.add_argument("--xpbd-rigid-contact-relaxation", type=float, default=0.8)
    parser.add_argument("--xpbd-angular-damping", type=float, default=0.0)
    parser.add_argument(
        "--xpbd-enable-restitution",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--semi-spring-ke-scale", type=float, default=1.0)
    parser.add_argument("--semi-spring-kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--semi-disable-particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--semi-enable-tri-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semi-angular-damping", type=float, default=0.05)
    parser.add_argument("--semi-friction-smoothing", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=296)
    parser.add_argument("--substeps-per-frame", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")
    parser.add_argument(
        "--frame-sync",
        choices=["legacy", "phystwin"],
        default="phystwin",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Warp device string. Defaults to NEWTON_DEVICE env var or cuda:0.",
    )
    parser.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--particle-contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--allow-coupled-contact-radius",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--inference", type=Path, default=None)
    parser.add_argument(
        "--gt-final-data",
        type=Path,
        default=None,
        help="Path to final_data.pkl used as GT source for Chamfer comparison.",
    )
    parser.add_argument(
        "--gt-export",
        type=Path,
        default=None,
        help="Optional path to save extracted GT object points/masks as .npz.",
    )
    parser.add_argument(
        "--gt-use-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use object_motions_valid and object_visibilities mask from final_data when available.",
    )

    parser.add_argument("--parity-collision-radius", type=float, default=1e-5)
    parser.add_argument(
        "--parity-disable-particle-collision",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--parity-controller-inactive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parity-interpolate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parity-apply-drag",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parity-apply-ground-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parity-gravity-from-reverse-z",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parity-gravity-mag", type=float, default=9.8)
    parser.add_argument(
        "--strict-phystwin",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--threshold-config", type=Path, default=_default_threshold_config())
    parser.add_argument(
        "--threshold-case",
        default=None,
        help="Override case key when reading case-specific thresholds.",
    )
    parser.add_argument("--max-x0-rmse", type=float, default=None)
    parser.add_argument("--max-rmse-mean", type=float, default=None)
    parser.add_argument("--max-rmse-max", type=float, default=None)
    parser.add_argument("--max-first30-rmse", type=float, default=None)
    parser.add_argument("--max-last30-rmse", type=float, default=None)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def _load_ir(ir_path: Path) -> dict[str, np.ndarray]:
    with np.load(ir_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _resolve_inference_path(args: argparse.Namespace, ir_path: Path, ir: dict[str, np.ndarray]) -> Path | None:
    if args.inference is not None:
        return args.inference.resolve()
    if "case_name" not in ir:
        return None
    case_name = str(np.asarray(ir["case_name"]).reshape(-1)[0])
    candidate = ir_path.parent.parent.parent / "inputs" / "cases" / case_name / "inference.pkl"
    return candidate if candidate.exists() else None


def _resolve_final_data_path(
    args: argparse.Namespace, ir_path: Path, ir: dict[str, np.ndarray]
) -> Path | None:
    if args.gt_final_data is not None:
        return args.gt_final_data.resolve()
    if "case_name" not in ir:
        return None
    case_name = str(np.asarray(ir["case_name"]).reshape(-1)[0])
    candidate = ir_path.parent.parent.parent / "inputs" / "cases" / case_name / "final_data.pkl"
    return candidate if candidate.exists() else None


def _load_inference(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    import pickle

    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle), dtype=np.float32)


def _load_gt_from_final_data(
    path: Path | None, use_mask: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None or not path.exists():
        return None, None
    with path.open("rb") as handle:
        final_data = pickle.load(handle)

    object_points = np.asarray(final_data["object_points"], dtype=np.float32)
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(f"Unexpected GT object_points shape: {object_points.shape}")

    mask = np.ones(object_points.shape[:2], dtype=bool)
    if use_mask:
        if "object_motions_valid" in final_data:
            motions_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
            if motions_valid.shape == mask.shape:
                mask &= motions_valid
        if "object_visibilities" in final_data:
            visibilities = np.asarray(final_data["object_visibilities"], dtype=bool)
            if visibilities.shape == mask.shape:
                mask &= visibilities
    return object_points, mask


def _compute_x0_rmse(ir: dict[str, np.ndarray], inference: np.ndarray | None) -> float | None:
    if inference is None or inference.ndim != 3:
        return None
    num_object_points = int(ir["num_object_points"])
    if inference.shape[1] != num_object_points:
        return None
    x0_object = ir["x0"][:num_object_points].astype(np.float32)
    return float(np.sqrt(np.mean((x0_object - inference[0]) ** 2)))


def _append_bool_flag(cmd: list[str], flag_name: str, enabled: bool | None) -> None:
    if enabled is None:
        return
    cmd.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def _safe_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _compute_chamfer_curve(
    pred: np.ndarray, gt_points: np.ndarray, gt_mask: np.ndarray
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required to compute GT Chamfer metrics.") from exc

    frames = min(pred.shape[0], gt_points.shape[0], gt_mask.shape[0])
    curve = np.full((frames,), np.nan, dtype=np.float32)

    for frame_idx in range(frames):
        pred_points = pred[frame_idx]
        gt_valid = gt_points[frame_idx][gt_mask[frame_idx]]
        if pred_points.shape[0] == 0 or gt_valid.shape[0] == 0:
            continue

        pred_tree = cKDTree(pred_points)
        gt_tree = cKDTree(gt_valid)
        try:
            d_pred_to_gt, _ = gt_tree.query(pred_points, k=1, workers=-1)
            d_gt_to_pred, _ = pred_tree.query(gt_valid, k=1, workers=-1)
        except TypeError:
            d_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
            d_gt_to_pred, _ = pred_tree.query(gt_valid, k=1)
        curve[frame_idx] = 0.5 * (float(np.mean(d_pred_to_gt)) + float(np.mean(d_gt_to_pred)))
    return curve


def _aggregate_curve(curve: np.ndarray) -> dict[str, float | int | None]:
    first = curve[: min(30, curve.size)]
    last = curve[max(0, curve.size - 30) :]
    return {
        "mean": _safe_mean(curve),
        "max": _safe_max(curve),
        "first30": _safe_mean(first),
        "last30": _safe_mean(last),
        "valid_frames": int(np.isfinite(curve).sum()),
    }


def _resolve_case_name(args: argparse.Namespace, ir: dict[str, np.ndarray]) -> str | None:
    if args.threshold_case is not None:
        return str(args.threshold_case)
    if "case_name" in ir:
        return str(np.asarray(ir["case_name"]).reshape(-1)[0])
    return None


def _load_structured_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        try:
            import yaml  # type: ignore

            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse {path} as YAML. Install pyyaml or use JSON."
            ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Threshold config must be a mapping: {path}")
    return loaded


def _apply_threshold_overrides(target: dict[str, float], source: dict, origin: str, applied: list[dict]) -> None:
    for key in DEFAULT_THRESHOLDS:
        if key not in source:
            continue
        target[key] = float(source[key])
        applied.append({"source": origin, "key": key, "value": float(source[key])})


def _resolve_thresholds(
    args: argparse.Namespace, case_name: str | None
) -> tuple[dict[str, float], list[dict], Path | None]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    applied: list[dict] = []
    config_path = args.threshold_config.resolve() if args.threshold_config is not None else None

    if config_path is not None:
        config_data = _load_structured_config(config_path)
        has_sections = "defaults" in config_data or "cases" in config_data
        defaults = config_data.get("defaults", {}) if has_sections else config_data
        if defaults is not None:
            if not isinstance(defaults, dict):
                raise ValueError(f"`defaults` must be a mapping in {config_path}")
            _apply_threshold_overrides(thresholds, defaults, "config.defaults", applied)

        if has_sections and case_name is not None:
            cases = config_data.get("cases", {})
            if cases is not None:
                if not isinstance(cases, dict):
                    raise ValueError(f"`cases` must be a mapping in {config_path}")
                case_overrides = cases.get(case_name)
                if case_overrides is not None:
                    if not isinstance(case_overrides, dict):
                        raise ValueError(
                            f"Case override for '{case_name}' must be a mapping in {config_path}"
                        )
                    _apply_threshold_overrides(
                        thresholds,
                        case_overrides,
                        f"config.cases.{case_name}",
                        applied,
                    )

    cli_overrides = {
        "max_x0_rmse": args.max_x0_rmse,
        "max_rmse_mean": args.max_rmse_mean,
        "max_rmse_max": args.max_rmse_max,
        "max_first30_rmse": args.max_first30_rmse,
        "max_last30_rmse": args.max_last30_rmse,
    }
    for key, value in cli_overrides.items():
        if value is None:
            continue
        thresholds[key] = float(value)
        applied.append({"source": "cli", "key": key, "value": float(value)})

    return thresholds, applied, config_path


def _build_importer_cmd(
    args: argparse.Namespace,
    ir_path: Path,
    out_dir: Path,
    ir: dict[str, np.ndarray],
) -> list[str]:
    cmd = [
        args.python,
        str(args.importer),
        "--ir",
        str(ir_path),
        "--out-dir",
        str(out_dir),
        "--output-prefix",
        args.output_prefix,
        "--mode",
        args.mode,
        "--solver",
        args.solver,
        "--solver-iterations",
        str(args.solver_iterations),
        "--xpbd-soft-body-relaxation",
        str(args.xpbd_soft_body_relaxation),
        "--xpbd-soft-contact-relaxation",
        str(args.xpbd_soft_contact_relaxation),
        "--xpbd-rigid-contact-relaxation",
        str(args.xpbd_rigid_contact_relaxation),
        "--xpbd-angular-damping",
        str(args.xpbd_angular_damping),
        "--semi-spring-ke-scale",
        str(args.semi_spring_ke_scale),
        "--semi-spring-kd-scale",
        str(args.semi_spring_kd_scale),
        "--semi-angular-damping",
        str(args.semi_angular_damping),
        "--semi-friction-smoothing",
        str(args.semi_friction_smoothing),
        "--num-frames",
        str(args.num_frames),
        "--device",
        args.device,
        "--up-axis",
        args.up_axis,
        "--frame-sync",
        args.frame_sync,
        "--parity-collision-radius",
        str(args.parity_collision_radius),
        "--parity-gravity-mag",
        str(args.parity_gravity_mag),
    ]
    if args.substeps_per_frame is not None:
        cmd.extend(["--substeps-per-frame", str(args.substeps_per_frame)])
    if args.sim_dt is not None:
        cmd.extend(["--sim-dt", str(args.sim_dt)])
    if args.gravity is not None:
        cmd.extend(["--gravity", str(args.gravity)])
    if args.shape_contacts:
        cmd.append("--shape-contacts")
    if args.particle_contacts is not None:
        _append_bool_flag(cmd, "particle-contacts", bool(args.particle_contacts))
    _append_bool_flag(
        cmd,
        "allow-coupled-contact-radius",
        bool(args.allow_coupled_contact_radius),
    )
    if args.inference is not None:
        cmd.extend(["--inference", str(args.inference)])

    parity_disable_particle_collision = args.parity_disable_particle_collision
    if parity_disable_particle_collision is None:
        if "self_collision" in ir and bool(np.asarray(ir["self_collision"]).reshape(-1)[0]):
            parity_disable_particle_collision = False
        else:
            parity_disable_particle_collision = True

    _append_bool_flag(
        cmd,
        "parity-disable-particle-collision",
        parity_disable_particle_collision,
    )
    _append_bool_flag(cmd, "parity-controller-inactive", args.parity_controller_inactive)
    _append_bool_flag(cmd, "parity-interpolate-controls", args.parity_interpolate_controls)
    _append_bool_flag(cmd, "parity-apply-drag", args.parity_apply_drag)
    _append_bool_flag(
        cmd,
        "parity-apply-ground-collision",
        args.parity_apply_ground_collision,
    )
    _append_bool_flag(cmd, "parity-gravity-from-reverse-z", args.parity_gravity_from_reverse_z)
    _append_bool_flag(cmd, "strict-phystwin", args.strict_phystwin)
    _append_bool_flag(
        cmd,
        "semi-disable-particle-contact-kernel",
        args.semi_disable_particle_contact_kernel,
    )
    _append_bool_flag(
        cmd,
        "semi-enable-tri-contact",
        args.semi_enable_tri_contact,
    )
    _append_bool_flag(
        cmd,
        "xpbd-enable-restitution",
        args.xpbd_enable_restitution,
    )
    return cmd


def _write_rmse_curve_csv(rmse_curve: np.ndarray, csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("frame,rmse\n")
        for frame_idx, value in enumerate(rmse_curve):
            handle.write(f"{frame_idx},{float(value):.8f}\n")


def _write_gt_curve_csv(
    newton_curve: np.ndarray,
    inference_curve: np.ndarray | None,
    csv_path: Path,
) -> None:
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("frame,newton_vs_gt_chamfer,inference_vs_gt_chamfer\n")
        num_frames = int(newton_curve.shape[0])
        for frame_idx in range(num_frames):
            newton_value = newton_curve[frame_idx]
            inference_value = (
                inference_curve[frame_idx]
                if inference_curve is not None and frame_idx < inference_curve.shape[0]
                else np.nan
            )
            handle.write(
                f"{frame_idx},{float(newton_value):.8f},{float(inference_value):.8f}\n"
            )


def main() -> int:
    args = parse_args()
    ir_path = args.ir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ir = _load_ir(ir_path)
    case_name = _resolve_case_name(args=args, ir=ir)
    inference_path = _resolve_inference_path(args=args, ir_path=ir_path, ir=ir)
    inference = _load_inference(inference_path)
    final_data_path = _resolve_final_data_path(args=args, ir_path=ir_path, ir=ir)
    gt_points, gt_mask = _load_gt_from_final_data(
        path=final_data_path,
        use_mask=bool(args.gt_use_mask),
    )
    x0_rmse = _compute_x0_rmse(ir=ir, inference=inference)
    thresholds, threshold_overrides, threshold_config_path = _resolve_thresholds(
        args=args,
        case_name=case_name,
    )

    importer_cmd = _build_importer_cmd(args=args, ir_path=ir_path, out_dir=out_dir, ir=ir)
    if not args.skip_run:
        subprocess.run(importer_cmd, check=True)

    rollout_json_path = out_dir / f"{args.output_prefix}.json"
    rollout_npz_path = out_dir / f"{args.output_prefix}.npz"
    report_json_path = out_dir / f"{args.output_prefix}_parity_report.json"
    rmse_curve_csv_path = out_dir / f"{args.output_prefix}_rmse_curve.csv"
    gt_curve_csv_path = out_dir / f"{args.output_prefix}_gt_chamfer_curve.csv"

    if not rollout_json_path.exists() or not rollout_npz_path.exists():
        raise FileNotFoundError("Rollout outputs not found. Ensure importer run completed successfully.")

    with rollout_json_path.open("r", encoding="utf-8") as handle:
        rollout_summary = json.load(handle)
    with np.load(rollout_npz_path) as data:
        rmse_curve = data["rmse_per_frame"].astype(np.float32)
        rollout_object = data["particle_q_object"].astype(np.float32)

    _write_rmse_curve_csv(rmse_curve=rmse_curve, csv_path=rmse_curve_csv_path)

    rmse_mean = float(np.mean(rmse_curve)) if rmse_curve.size else None
    rmse_max = float(np.max(rmse_curve)) if rmse_curve.size else None
    first30_rmse = float(np.mean(rmse_curve[: min(30, rmse_curve.size)])) if rmse_curve.size else None
    last30_rmse = float(np.mean(rmse_curve[max(0, rmse_curve.size - 30) :])) if rmse_curve.size else None

    newton_gt_curve = None
    inference_gt_curve = None
    newton_gt_stats = None
    inference_gt_stats = None
    gt_compare_summary = None
    if gt_points is not None and gt_mask is not None:
        if args.gt_export is not None:
            gt_export_path = args.gt_export.resolve()
            gt_export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                gt_export_path,
                object_points=gt_points.astype(np.float32),
                object_mask=gt_mask.astype(np.bool_),
            )
        newton_gt_curve = _compute_chamfer_curve(
            pred=rollout_object,
            gt_points=gt_points,
            gt_mask=gt_mask,
        )
        newton_gt_stats = _aggregate_curve(newton_gt_curve)

        if inference is not None and inference.ndim == 3:
            inference_gt_curve = _compute_chamfer_curve(
                pred=inference.astype(np.float32),
                gt_points=gt_points,
                gt_mask=gt_mask,
            )
            inference_gt_stats = _aggregate_curve(inference_gt_curve)

        _write_gt_curve_csv(
            newton_curve=newton_gt_curve,
            inference_curve=inference_gt_curve,
            csv_path=gt_curve_csv_path,
        )

        if (
            newton_gt_stats is not None
            and inference_gt_stats is not None
            and newton_gt_stats["mean"] is not None
            and inference_gt_stats["mean"] is not None
        ):
            newton_mean = float(newton_gt_stats["mean"])
            inference_mean = float(inference_gt_stats["mean"])
            if abs(newton_mean - inference_mean) < 1e-12:
                better = "tie"
            else:
                better = "newton" if newton_mean < inference_mean else "inference"
            gt_compare_summary = {
                "better_vs_gt": better,
                "mean_delta_inference_minus_newton": inference_mean - newton_mean,
            }

    checks = {
        "x0_rmse": {
            "value": x0_rmse,
            "threshold": float(thresholds["max_x0_rmse"]),
            "pass": x0_rmse is not None and x0_rmse <= thresholds["max_x0_rmse"],
        },
        "rmse_mean": {
            "value": rmse_mean,
            "threshold": float(thresholds["max_rmse_mean"]),
            "pass": rmse_mean is not None and rmse_mean <= thresholds["max_rmse_mean"],
        },
        "rmse_max": {
            "value": rmse_max,
            "threshold": float(thresholds["max_rmse_max"]),
            "pass": rmse_max is not None and rmse_max <= thresholds["max_rmse_max"],
        },
        "first30_rmse": {
            "value": first30_rmse,
            "threshold": float(thresholds["max_first30_rmse"]),
            "pass": first30_rmse is not None and first30_rmse <= thresholds["max_first30_rmse"],
        },
        "last30_rmse": {
            "value": last30_rmse,
            "threshold": float(thresholds["max_last30_rmse"]),
            "pass": last30_rmse is not None and last30_rmse <= thresholds["max_last30_rmse"],
        },
    }

    passed = all(item["pass"] for item in checks.values())
    report = {
        "passed": passed,
        "ir_path": str(ir_path),
        "case_name": case_name,
        "inference_path": str(inference_path) if inference_path is not None else None,
        "gt_final_data_path": str(final_data_path) if final_data_path is not None else None,
        "gt_use_mask": bool(args.gt_use_mask),
        "gt_export_path": str(args.gt_export.resolve()) if args.gt_export is not None else None,
        "rollout_json": str(rollout_json_path),
        "rollout_npz": str(rollout_npz_path),
        "rmse_curve_csv": str(rmse_curve_csv_path),
        "gt_chamfer_curve_csv": (
            str(gt_curve_csv_path) if newton_gt_curve is not None else None
        ),
        "importer_command": importer_cmd,
        "threshold_config": str(threshold_config_path) if threshold_config_path is not None else None,
        "threshold_overrides": threshold_overrides,
        "checks": checks,
        "newton_vs_gt_chamfer": newton_gt_stats,
        "inference_vs_gt_chamfer": inference_gt_stats,
        "gt_comparison": gt_compare_summary,
        "rollout_summary": rollout_summary,
    }

    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
