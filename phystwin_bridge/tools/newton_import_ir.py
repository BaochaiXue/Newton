#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
from path_defaults import default_device


@wp.kernel
def _set_indexed_particle_state(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_idx: wp.array(dtype=wp.int32),
    target_q: wp.array(dtype=wp.vec3),
    target_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    idx = particle_idx[tid]
    particle_q[idx] = target_q[tid]
    particle_qd[idx] = target_qd[tid]


@wp.kernel
def _apply_parity_post_correction(
    prev_particle_q: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    num_object_points: int,
    dt: float,
    drag_damping: float,
    apply_drag: int,
    apply_ground_collision: int,
    reverse_factor: float,
    collide_elas: float,
    collide_fric: float,
):
    tid = wp.tid()
    if tid >= num_object_points:
        return

    x0 = prev_particle_q[tid]
    v0 = particle_qd[tid]
    if apply_drag != 0:
        v0 = v0 * wp.exp(-dt * drag_damping)

    if apply_ground_collision != 0:
        normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

        x_z = x0[2]
        v_z = v0[2]
        next_x_z = (x_z + v_z * dt) * reverse_factor

        if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
            v_normal = wp.dot(v0, normal) * normal
            v_tangent = v0 - v_normal
            v_normal_length = wp.length(v_normal)
            v_tangent_length = wp.max(wp.length(v_tangent), 1e-6)

            clamp_collide_elas = wp.clamp(collide_elas, low=0.0, high=1.0)
            clamp_collide_fric = wp.clamp(collide_fric, low=0.0, high=2.0)

            v_normal_new = -clamp_collide_elas * v_normal
            tangent_scale = wp.max(
                0.0,
                1.0
                - clamp_collide_fric
                * (1.0 + clamp_collide_elas)
                * v_normal_length
                / v_tangent_length,
            )
            v_tangent_new = tangent_scale * v_tangent
            v1 = v_normal_new + v_tangent_new

            toi = -x_z / v_z
            particle_q[tid] = x0 + v0 * toi + v1 * (dt - toi)
            particle_qd[tid] = v1
            return

    particle_q[tid] = x0 + v0 * dt
    particle_qd[tid] = v0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate a PhysTwin IR in Newton and run a short rollout."
    )
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["standard", "parity"], default="standard")
    parser.add_argument(
        "--solver",
        choices=["xpbd", "semi_implicit"],
        default=None,
        help=(
            "Solver backend. If unset, defaults to semi_implicit in parity mode, xpbd otherwise."
        ),
    )
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument(
        "--xpbd-soft-body-relaxation",
        type=float,
        default=0.9,
        help="XPBD soft-body relaxation.",
    )
    parser.add_argument(
        "--xpbd-soft-contact-relaxation",
        type=float,
        default=0.9,
        help="XPBD soft-contact relaxation.",
    )
    parser.add_argument(
        "--xpbd-rigid-contact-relaxation",
        type=float,
        default=0.8,
        help="XPBD rigid-contact relaxation.",
    )
    parser.add_argument(
        "--xpbd-angular-damping",
        type=float,
        default=0.0,
        help="XPBD rigid-body angular damping.",
    )
    parser.add_argument(
        "--xpbd-enable-restitution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable XPBD restitution.",
    )
    parser.add_argument(
        "--semi-spring-ke-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to spring_ke when --solver semi_implicit.",
    )
    parser.add_argument(
        "--semi-spring-kd-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to spring_kd when --solver semi_implicit.",
    )
    parser.add_argument(
        "--semi-disable-particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --solver semi_implicit, disable Newton particle contact kernel by setting "
            "model.particle_grid=None."
        ),
    )
    parser.add_argument(
        "--semi-enable-tri-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --solver semi_implicit, enable/disable triangle contact kernel.",
    )
    parser.add_argument(
        "--semi-angular-damping",
        type=float,
        default=0.05,
        help="Semi-implicit rigid-body angular damping.",
    )
    parser.add_argument(
        "--semi-friction-smoothing",
        type=float,
        default=1.0,
        help="Semi-implicit friction smoothing.",
    )
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=None,
        help="Override IR substeps for faster short rollouts.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=None,
        help="Override IR sim_dt.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=None,
        help="Gravity value along up-axis. If unset and mode=parity, derive from reverse_z.",
    )
    parser.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Warp device string. Defaults to NEWTON_DEVICE env var or cuda:0.",
    )
    parser.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shape contact generation via model.collide(...).",
    )
    parser.add_argument(
        "--particle-contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable particle-particle contacts. If unset, defaults to enabled in standard mode "
            "and disabled in parity mode."
        ),
    )
    parser.add_argument(
        "--allow-coupled-contact-radius",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow shape-contacts with particle-contacts disabled. This is approximate because "
            "the same particle radius controls both contact paths."
        ),
    )
    parser.add_argument(
        "--enable-contacts",
        dest="legacy_enable_contacts",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parity-collision-radius",
        type=float,
        default=1e-5,
        help="Collision radius used in parity mode when particle-particle collisions are disabled.",
    )
    parser.add_argument(
        "--parity-disable-particle-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, set all particle radii to parity-collision-radius to suppress internal particle collisions.",
    )
    parser.add_argument(
        "--parity-controller-inactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, mark controller particles inactive for Newton contact kernels.",
    )
    parser.add_argument(
        "--parity-interpolate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, linearly interpolate controller trajectory within each frame substep.",
    )
    parser.add_argument(
        "--parity-apply-drag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, apply PhysTwin drag damping after each Newton substep.",
    )
    parser.add_argument(
        "--parity-apply-ground-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, apply PhysTwin-style explicit ground collision after each Newton substep.",
    )
    parser.add_argument(
        "--parity-gravity-from-reverse-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In parity mode, derive gravity direction from reverse_z when --gravity is not specified.",
    )
    parser.add_argument(
        "--parity-gravity-mag",
        type=float,
        default=9.8,
        help="Gravity magnitude used by parity mode when deriving from reverse_z.",
    )
    parser.add_argument(
        "--frame-sync",
        choices=["legacy", "phystwin"],
        default="phystwin",
        help=(
            "Rollout frame indexing convention. "
            "'phystwin' keeps frame 0 as initial state and simulates transitions 0->1, 1->2, ... "
            "to match PhysTwin."
        ),
    )
    parser.add_argument(
        "--strict-phystwin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict PhysTwin semantic checks to avoid parameter-mapping mismatches.",
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=None,
        help="Optional PhysTwin baseline inference.pkl for error metrics.",
    )
    parser.add_argument("--output-prefix", default="newton_rollout_short")
    return parser.parse_args()


def _as_scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(-1)[0])


def _as_bool(array: np.ndarray) -> bool:
    return bool(np.asarray(array).reshape(-1)[0])


def _as_string(array: np.ndarray) -> str:
    return str(np.asarray(array).reshape(-1)[0])


def _load_ir(ir_path: Path) -> dict[str, np.ndarray]:
    with np.load(ir_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_device(device_str: str) -> str:
    try:
        wp.get_device(device_str)
        return device_str
    except Exception:
        return "cpu"


def _gravity_vector(up_axis: str, gravity_scalar: float) -> tuple[float, float, float]:
    vector = [0.0, 0.0, 0.0]
    axis_to_index = {"X": 0, "Y": 1, "Z": 2}
    vector[axis_to_index[up_axis]] = float(gravity_scalar)
    return tuple(vector)


def _resolve_gravity(args: argparse.Namespace, ir: dict[str, np.ndarray]) -> tuple[float, tuple[float, float, float]]:
    if args.gravity is not None:
        gravity_scalar = float(args.gravity)
    elif args.mode == "parity" and args.parity_gravity_from_reverse_z:
        reverse_z = _as_bool(ir["reverse_z"]) if "reverse_z" in ir else False
        gravity_scalar = float(args.parity_gravity_mag) * (1.0 if reverse_z else -1.0)
    else:
        gravity_scalar = 0.0
    return gravity_scalar, _gravity_vector(args.up_axis, gravity_scalar)


def _resolve_reverse_factor(ir: dict[str, np.ndarray]) -> float:
    reverse_z = _as_bool(ir["reverse_z"]) if "reverse_z" in ir else False
    return -1.0 if reverse_z else 1.0


def _resolve_parity_drag_damping(ir: dict[str, np.ndarray]) -> float:
    if "drag_damping" in ir:
        return _as_scalar(ir["drag_damping"])
    return 0.0


def _resolve_ground_contact_params(ir: dict[str, np.ndarray]) -> tuple[float, float]:
    collide_elas = _as_scalar(ir.get("contact_collide_elas", np.asarray(0.5, dtype=np.float32)))
    collide_fric = _as_scalar(ir.get("contact_collide_fric", np.asarray(0.3, dtype=np.float32)))
    return float(collide_elas), float(collide_fric)


def _resolve_object_contact_params(ir: dict[str, np.ndarray]) -> tuple[float, float]:
    collide_object_elas = _as_scalar(
        ir.get("contact_collide_object_elas", np.asarray(0.7, dtype=np.float32))
    )
    collide_object_fric = _as_scalar(
        ir.get("contact_collide_object_fric", np.asarray(0.3, dtype=np.float32))
    )
    return float(collide_object_elas), float(collide_object_fric)


def _resolve_solver_name(args: argparse.Namespace) -> str:
    if args.solver is not None:
        return str(args.solver)
    if args.mode == "parity":
        return "semi_implicit"
    return "xpbd"


def _resolve_shape_contacts_enabled(args: argparse.Namespace) -> bool:
    return bool(args.shape_contacts or args.legacy_enable_contacts)


def _resolve_particle_contacts_enabled(
    args: argparse.Namespace, ir: dict[str, np.ndarray]
) -> bool:
    if args.particle_contacts is not None:
        return bool(args.particle_contacts)
    if args.mode == "parity":
        if "self_collision" in ir:
            return bool(_as_bool(ir["self_collision"]))
        return not bool(args.parity_disable_particle_collision)
    return True


def _resolve_ir_version(ir: dict[str, np.ndarray]) -> int:
    if "ir_version" not in ir:
        return 1
    return int(np.asarray(ir["ir_version"]).reshape(-1)[0])


def _validate_phystwin_semantics(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    solver_name: str,
) -> dict[str, float | str | bool]:
    checks: dict[str, float | str | bool] = {}
    strict = bool(args.strict_phystwin)

    if args.semi_spring_ke_scale <= 0.0:
        raise ValueError(
            f"--semi-spring-ke-scale must be > 0, got {args.semi_spring_ke_scale}."
        )
    if args.semi_spring_kd_scale < 0.0:
        raise ValueError(
            f"--semi-spring-kd-scale must be >= 0, got {args.semi_spring_kd_scale}."
        )
    if args.semi_angular_damping < 0.0:
        raise ValueError(
            f"--semi-angular-damping must be >= 0, got {args.semi_angular_damping}."
        )
    if args.semi_friction_smoothing <= 0.0:
        raise ValueError(
            f"--semi-friction-smoothing must be > 0, got {args.semi_friction_smoothing}."
        )
    if args.xpbd_soft_body_relaxation <= 0.0:
        raise ValueError(
            f"--xpbd-soft-body-relaxation must be > 0, got {args.xpbd_soft_body_relaxation}."
        )
    if args.xpbd_soft_contact_relaxation <= 0.0:
        raise ValueError(
            f"--xpbd-soft-contact-relaxation must be > 0, got {args.xpbd_soft_contact_relaxation}."
        )
    if args.xpbd_rigid_contact_relaxation <= 0.0:
        raise ValueError(
            f"--xpbd-rigid-contact-relaxation must be > 0, got {args.xpbd_rigid_contact_relaxation}."
        )
    if args.xpbd_angular_damping < 0.0:
        raise ValueError(
            f"--xpbd-angular-damping must be >= 0, got {args.xpbd_angular_damping}."
        )

    if strict and args.mode == "parity" and solver_name not in {"semi_implicit", "xpbd"}:
        raise ValueError(
            "Strict PhysTwin mode requires semi_implicit or xpbd solver in parity mode. "
            f"Got solver={solver_name!r}. Re-run with --solver semi_implicit (or xpbd) or --no-strict-phystwin."
        )
    if strict and args.frame_sync != "phystwin":
        raise ValueError(
            "Strict PhysTwin mode requires --frame-sync phystwin. "
            f"Got frame_sync={args.frame_sync!r}."
        )
    if strict and args.mode == "parity" and args.gravity is None and not args.parity_gravity_from_reverse_z:
        raise ValueError(
            "Strict PhysTwin mode requires gravity from reverse_z when --gravity is unset. "
            "Re-run with --parity-gravity-from-reverse-z or set --gravity explicitly."
        )
    if strict and args.mode == "parity" and not args.parity_apply_drag:
        raise ValueError(
            "Strict PhysTwin mode requires --parity-apply-drag."
        )
    if strict and args.mode == "parity" and not args.parity_apply_ground_collision:
        raise ValueError(
            "Strict PhysTwin mode requires --parity-apply-ground-collision."
        )
    if (
        strict
        and args.mode == "parity"
        and "self_collision" in ir
        and _as_bool(ir["self_collision"])
        and args.particle_contacts is False
    ):
        raise ValueError(
            "Strict PhysTwin mode with self_collision=true requires --particle-contacts."
        )
    if (
        strict
        and args.mode == "parity"
        and "self_collision" in ir
        and _as_bool(ir["self_collision"])
        and args.parity_disable_particle_collision
    ):
        raise ValueError(
            "Strict PhysTwin mode with self_collision=true forbids --parity-disable-particle-collision."
        )

    spring_edges = np.asarray(ir["spring_edges"])
    spring_count = int(spring_edges.shape[0])
    spring_ke = np.asarray(ir["spring_ke"], dtype=np.float64).reshape(-1)
    if spring_ke.shape[0] != spring_count:
        raise ValueError(
            f"Expected spring_ke length {spring_count}, got {spring_ke.shape[0]}."
        )
    if not np.all(np.isfinite(spring_ke)):
        raise ValueError("Non-finite spring_ke values detected in IR.")

    spring_rest = None
    if "spring_rest_length" in ir:
        spring_rest = np.asarray(ir["spring_rest_length"], dtype=np.float64).reshape(-1)
        if spring_rest.shape[0] != spring_count:
            raise ValueError(
                f"Expected spring_rest_length length {spring_count}, got {spring_rest.shape[0]}."
            )
        if not np.all(np.isfinite(spring_rest)):
            raise ValueError("Non-finite spring_rest_length values detected in IR.")
        if np.any(spring_rest <= 0.0):
            raise ValueError(
                f"Non-positive spring_rest_length detected (min={float(np.min(spring_rest))})."
            )
        checks["spring_rest_length_min"] = float(np.min(spring_rest))
        checks["spring_rest_length_max"] = float(np.max(spring_rest))
    elif strict:
        raise ValueError("Strict PhysTwin mode requires spring_rest_length in IR.")
    else:
        warnings.warn(
            "IR missing spring_rest_length; spring semantic checks are limited.",
            RuntimeWarning,
        )

    spring_ke_mode = "unknown"
    if "spring_ke_mode" in ir:
        spring_ke_mode = _as_string(ir["spring_ke_mode"])
    checks["spring_ke_mode"] = spring_ke_mode

    if spring_ke_mode == "y_over_rest":
        if "spring_y" not in ir:
            raise ValueError(
                "IR says spring_ke_mode=y_over_rest but spring_y is missing."
            )
        if spring_rest is None:
            raise ValueError(
                "IR says spring_ke_mode=y_over_rest but spring_rest_length is missing."
            )
        spring_y = np.asarray(ir["spring_y"], dtype=np.float64).reshape(-1)
        if spring_y.shape[0] != spring_count:
            raise ValueError(
                f"Expected spring_y length {spring_count}, got {spring_y.shape[0]}."
            )
        expected_ke = spring_y / np.maximum(spring_rest, 1e-12)
        abs_err = np.abs(spring_ke - expected_ke)
        rel_err = abs_err / np.maximum(np.abs(expected_ke), 1e-12)
        checks["spring_ke_rel_error_max"] = float(np.max(rel_err))
        checks["spring_ke_rel_error_mean"] = float(np.mean(rel_err))
        if strict and float(np.max(rel_err)) > 1e-4:
            raise ValueError(
                "spring_ke does not match spring_y/rest_length under strict PhysTwin mode "
                f"(max_rel_error={float(np.max(rel_err)):.6e})."
            )
    elif strict:
        raise ValueError(
            "Strict PhysTwin mode requires IR spring_ke_mode=y_over_rest. "
            f"Got {spring_ke_mode!r}. Re-export IR with strict defaults."
        )

    if "controller_idx" in ir:
        controller_idx = np.asarray(ir["controller_idx"], dtype=np.int64).reshape(-1)
        mass = np.asarray(ir["mass"], dtype=np.float64).reshape(-1)
        if controller_idx.size > 0:
            if controller_idx.min() < 0 or controller_idx.max() >= mass.shape[0]:
                raise ValueError("controller_idx out of bounds for mass array.")
            controller_mass_abs_max = float(np.max(np.abs(mass[controller_idx])))
        else:
            controller_mass_abs_max = 0.0
        checks["controller_mass_abs_max"] = controller_mass_abs_max
        if strict and controller_mass_abs_max > 1e-8:
            raise ValueError(
                "Controller particles must be kinematic (mass=0) in strict PhysTwin mode. "
                f"Observed max |mass|={controller_mass_abs_max:.6e}."
            )

    if "self_collision" in ir:
        checks["self_collision"] = bool(_as_bool(ir["self_collision"]))

    return checks


def _uniform_collision_radius(
    ir: dict[str, np.ndarray], num_particles: int, value: float
) -> np.ndarray:
    radius = np.full((num_particles,), float(value), dtype=np.float32)
    if "controller_idx" in ir and "collision_controller_radius" in ir:
        controller_idx = ir["controller_idx"].astype(np.int64, copy=False)
        if controller_idx.size > 0:
            controller_radius = _as_scalar(ir["collision_controller_radius"])
            radius[controller_idx] = float(controller_radius)
    return radius


def _legacy_collision_radius_from_ir(
    ir: dict[str, np.ndarray], num_particles: int
) -> tuple[np.ndarray, str]:
    radius = None
    source = "legacy_fallback_defaults"
    if "radius" in ir:
        radius = ir["radius"].astype(np.float32).copy()
        source = "legacy_radius"

    collision_dist = None
    if "contact_collision_dist" in ir:
        collision_dist = _as_scalar(ir["contact_collision_dist"])

    if radius is not None and collision_dist is not None:
        # Legacy v1 often stored topology radius in `radius`; detect and correct it.
        ratio = float(np.max(radius) / max(float(collision_dist), 1e-12))
        if ratio > 2.5:
            warnings.warn(
                "Legacy IR detected: `radius` looks like topology radius. "
                "Using `contact_collision_dist` for collision radius.",
                RuntimeWarning,
            )
            return _uniform_collision_radius(ir=ir, num_particles=num_particles, value=collision_dist), (
                "legacy_contact_collision_dist"
            )
        return radius, source

    if radius is not None:
        return radius, source

    if collision_dist is not None:
        warnings.warn(
            "Legacy IR missing `radius`; using `contact_collision_dist` for collision radius.",
            RuntimeWarning,
        )
        return _uniform_collision_radius(ir=ir, num_particles=num_particles, value=collision_dist), (
            "legacy_contact_collision_dist"
        )

    object_radius = _as_scalar(
        ir.get(
            "collision_object_radius",
            ir.get("object_radius", np.asarray(0.02, dtype=np.float32)),
        )
    )
    controller_radius = _as_scalar(
        ir.get(
            "collision_controller_radius",
            ir.get("controller_radius", np.asarray(object_radius, dtype=np.float32)),
        )
    )
    radius = np.full((num_particles,), object_radius, dtype=np.float32)
    if "controller_idx" in ir:
        controller_idx = ir["controller_idx"].astype(np.int64, copy=False)
        if controller_idx.size > 0:
            radius[controller_idx] = controller_radius
    return radius, source


def _initial_collision_radius(
    ir: dict[str, np.ndarray], num_particles: int
) -> tuple[np.ndarray, int, str]:
    ir_version = _resolve_ir_version(ir)
    if "collision_radius" in ir:
        return ir["collision_radius"].astype(np.float32).copy(), ir_version, "collision_radius"

    if ir_version >= 2:
        if "radius" in ir:
            warnings.warn(
                "IR v2 missing `collision_radius`; falling back to `radius` for compatibility.",
                RuntimeWarning,
            )
            return ir["radius"].astype(np.float32).copy(), ir_version, "v2_radius_fallback"
        raise KeyError("IR v2 requires `collision_radius`.")

    legacy_radius, source = _legacy_collision_radius_from_ir(
        ir=ir, num_particles=num_particles
    )
    return legacy_radius, ir_version, source


def _build_particle_data(
    ir: dict[str, np.ndarray], args: argparse.Namespace, particle_contacts_enabled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    x0 = ir["x0"].astype(np.float32, copy=False)
    v0 = ir["v0"].astype(np.float32, copy=False)
    mass = ir["mass"].astype(np.float32, copy=False)
    radius, ir_version, radius_source = _initial_collision_radius(
        ir=ir, num_particles=x0.shape[0]
    )
    flags = np.full((x0.shape[0],), int(newton.ParticleFlags.ACTIVE), dtype=np.int32)

    if not particle_contacts_enabled:
        radius.fill(float(args.parity_collision_radius))
    elif args.mode == "parity" and "self_collision" in ir and _as_bool(ir["self_collision"]):
        # PhysTwin self-collision triggers when pair distance < collision_dist.
        # Newton particle contact triggers around (radius_i + radius_j), so use half-distance radius.
        collision_dist = _as_scalar(
            ir.get("contact_collision_dist", np.asarray(0.02, dtype=np.float32))
        )
        object_contact_radius = float(max(collision_dist, 1e-8) * 0.5)
        radius.fill(object_contact_radius)
        if "controller_idx" in ir:
            controller_idx = ir["controller_idx"].astype(np.int64, copy=False)
            if controller_idx.size > 0:
                radius[controller_idx] = float(args.parity_collision_radius)

    if args.mode == "parity":
        if args.parity_controller_inactive and "controller_idx" in ir:
            controller_idx = ir["controller_idx"].astype(np.int64, copy=False)
            if controller_idx.size > 0:
                flags[controller_idx] = 0

    return x0, v0, mass, radius, flags, ir_version, radius_source


def _build_model(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
    solver_name: str,
):
    semantic_checks = _validate_phystwin_semantics(
        ir=ir,
        args=args,
        solver_name=solver_name,
    )
    shape_contacts_enabled = _resolve_shape_contacts_enabled(args)
    particle_contacts_enabled = _resolve_particle_contacts_enabled(args, ir)
    contact_semantics = "exact"
    if shape_contacts_enabled and not particle_contacts_enabled:
        contact_semantics = "approx_coupled_radius"
        if not args.allow_coupled_contact_radius:
            raise ValueError(
                "Unsupported contact combination: --shape-contacts with --no-particle-contacts "
                "shares particle radius semantics in Newton. Re-run with "
                "--allow-coupled-contact-radius to accept this approximation."
            )
        warnings.warn(
            "Using approximate contact semantics: shape contacts enabled while particle contacts are disabled. "
            "Particle radius is coupled across contact paths.",
            RuntimeWarning,
        )

    axis = newton.Axis.from_any(args.up_axis)
    builder = newton.ModelBuilder(up_axis=axis, gravity=0.0)
    x0, v0, mass, radius, flags, ir_version, radius_source = _build_particle_data(
        ir=ir, args=args, particle_contacts_enabled=particle_contacts_enabled
    )

    builder.add_particles(
        pos=[tuple(row.tolist()) for row in x0],
        vel=[tuple(row.tolist()) for row in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
        flags=flags.astype(int).tolist(),
    )

    spring_edges = ir["spring_edges"]
    if spring_edges.ndim != 2 or spring_edges.shape[1] != 2:
        raise ValueError(f"Expected `spring_edges` shape [spring_count, 2], got {spring_edges.shape}")
    spring_count = int(spring_edges.shape[0])

    spring_ke = np.asarray(ir["spring_ke"]).reshape(-1)
    spring_kd = np.asarray(ir["spring_kd"]).reshape(-1)
    if spring_ke.shape[0] != spring_count:
        raise ValueError(f"Expected `spring_ke` length {spring_count}, got {spring_ke.shape[0]}")
    if spring_kd.shape[0] != spring_count:
        raise ValueError(f"Expected `spring_kd` length {spring_count}, got {spring_kd.shape[0]}")
    spring_rest_length = None
    if "spring_rest_length" in ir:
        rest = ir["spring_rest_length"].astype(np.float32, copy=False).reshape(-1)
        if rest.shape[0] == spring_count:
            spring_rest_length = rest
        else:
            warnings.warn(
                "Ignoring `spring_rest_length` because it does not match `spring_edges` "
                f"count: rest={rest.shape[0]}, edges={spring_count}.",
                RuntimeWarning,
            )
    if solver_name == "semi_implicit":
        spring_ke = spring_ke * float(args.semi_spring_ke_scale)
        spring_kd = spring_kd * float(args.semi_spring_kd_scale)
        semantic_checks["semi_spring_ke_scale"] = float(args.semi_spring_ke_scale)
        semantic_checks["semi_spring_kd_scale"] = float(args.semi_spring_kd_scale)
    for spring_idx in range(spring_count):
        i = int(spring_edges[spring_idx, 0])
        j = int(spring_edges[spring_idx, 1])
        builder.add_spring(
            i=i,
            j=j,
            ke=float(spring_ke[spring_idx]),
            kd=float(spring_kd[spring_idx]),
            control=0.0,
        )
        if spring_rest_length is not None:
            builder.spring_rest_length[-1] = float(spring_rest_length[spring_idx])

    model = builder.finalize(device=device)
    if solver_name == "semi_implicit" and args.semi_disable_particle_contact_kernel:
        model.particle_grid = None
        semantic_checks["semi_particle_contact_kernel_disabled"] = True
    else:
        semantic_checks["semi_particle_contact_kernel_disabled"] = False
    gravity_scalar, gravity_vector = _resolve_gravity(args=args, ir=ir)
    model.set_gravity(gravity_vector)
    if "contact_collide_fric" in ir:
        particle_mu = float(np.clip(_as_scalar(ir["contact_collide_fric"]), 0.0, 2.0))
        model.particle_mu = particle_mu
        semantic_checks["particle_mu"] = particle_mu
    return (
        model,
        gravity_scalar,
        gravity_vector,
        radius,
        flags,
        shape_contacts_enabled,
        particle_contacts_enabled,
        ir_version,
        radius_source,
        contact_semantics,
        semantic_checks,
    )


def _make_solver(
    model: newton.Model,
    solver_name: str,
    iterations: int,
    args: argparse.Namespace,
):
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(
            model,
            angular_damping=float(args.semi_angular_damping),
            friction_smoothing=float(args.semi_friction_smoothing),
            enable_tri_contact=bool(args.semi_enable_tri_contact),
        )
    return newton.solvers.SolverXPBD(
        model,
        iterations=iterations,
        soft_body_relaxation=float(args.xpbd_soft_body_relaxation),
        soft_contact_relaxation=float(args.xpbd_soft_contact_relaxation),
        rigid_contact_relaxation=float(args.xpbd_rigid_contact_relaxation),
        angular_damping=float(args.xpbd_angular_damping),
        enable_restitution=bool(args.xpbd_enable_restitution),
    )


def _load_inference(path: Path | None):
    if path is None or not path.exists():
        return None
    import pickle

    with path.open("rb") as handle:
        inference = pickle.load(handle)
    return np.asarray(inference, dtype=np.float32)


def _controller_target_for_substep(
    controller_traj: np.ndarray,
    frame_idx: int,
    substep_idx: int,
    substeps_per_frame: int,
    interpolate_controls: bool,
) -> np.ndarray:
    if not interpolate_controls:
        return controller_traj[frame_idx]
    if frame_idx == 0:
        return controller_traj[0]
    alpha = float(substep_idx + 1) / float(substeps_per_frame)
    prev_ctrl = controller_traj[frame_idx - 1]
    next_ctrl = controller_traj[frame_idx]
    return prev_ctrl + (next_ctrl - prev_ctrl) * alpha


def main() -> int:
    args = parse_args()
    ir_path = args.ir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ir = _load_ir(ir_path)
    resolved_device = _resolve_device(args.device)
    wp.init()
    solver_name = _resolve_solver_name(args)

    (
        model,
        gravity_scalar,
        gravity_vector,
        radius_used,
        flags_used,
        shape_contacts_enabled,
        particle_contacts_enabled,
        ir_version,
        radius_source,
        contact_semantics,
        semantic_checks,
    ) = _build_model(
        ir=ir,
        args=args,
        device=resolved_device,
        solver_name=solver_name,
    )
    solver = _make_solver(model, solver_name, args.solver_iterations, args)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts() if shape_contacts_enabled else None

    controller_idx = ir["controller_idx"].astype(np.int64)
    controller_traj = ir["controller_traj"].astype(np.float32)
    num_object_points = int(ir["num_object_points"])
    if controller_idx.ndim != 1:
        raise ValueError(f"Expected `controller_idx` 1D array, got {controller_idx.shape}")
    if controller_traj.ndim != 3 or controller_traj.shape[-1] != 3:
        raise ValueError(f"Expected `controller_traj` shape [frames, ctrl_count, 3], got {controller_traj.shape}")
    if controller_traj.shape[1] != controller_idx.size:
        raise ValueError(
            "Controller count mismatch between `controller_traj` and `controller_idx`: "
            f"traj_ctrl={controller_traj.shape[1]}, idx_ctrl={controller_idx.size}."
        )
    if controller_idx.size > 0:
        particle_count = int(np.asarray(ir["x0"]).shape[0])
        if controller_idx.min() < 0 or controller_idx.max() >= particle_count:
            raise ValueError(
                f"`controller_idx` out of bounds for particle_count={particle_count}: "
                f"min={int(controller_idx.min())}, max={int(controller_idx.max())}."
            )

    ir_frames = controller_traj.shape[0]
    frames_to_run = max(1, min(args.num_frames, ir_frames))

    sim_dt = float(args.sim_dt) if args.sim_dt is not None else _as_scalar(ir["sim_dt"])
    substeps_per_frame = (
        int(args.substeps_per_frame)
        if args.substeps_per_frame is not None
        else int(np.asarray(ir["sim_substeps"]).reshape(-1)[0])
    )
    substeps_per_frame = max(1, substeps_per_frame)

    rollout_all: list[np.ndarray] = []
    rollout_object: list[np.ndarray] = []

    controller_idx_wp = None
    controller_target_wp = None
    controller_vel_wp = None
    controller_vel_zero = None
    if controller_idx.size > 0:
        controller_idx_wp = wp.array(
            controller_idx.astype(np.int32, copy=False),
            dtype=wp.int32,
            device=model.device,
        )
        controller_target_wp = wp.empty((controller_idx.size,), dtype=wp.vec3, device=model.device)
        controller_vel_wp = wp.empty((controller_idx.size,), dtype=wp.vec3, device=model.device)
        # Keep controller velocities at zero: parity assumes controllers are position-projected kinematics,
        # and feeding finite-difference velocities into spring damping changes the effective PhysTwin behavior.
        controller_vel_zero = np.zeros((controller_idx.size, 3), dtype=np.float32)

    if args.frame_sync == "phystwin":
        q0 = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q0)
        rollout_object.append(q0[:num_object_points])
        frame_indices = range(1, frames_to_run)
    else:
        frame_indices = range(frames_to_run)

    wall_start = time.perf_counter()
    parity_apply_drag = bool(args.mode == "parity" and args.parity_apply_drag)
    parity_apply_ground_collision = bool(
        args.mode == "parity" and args.parity_apply_ground_collision
    )
    parity_post_correction_enabled = bool(parity_apply_drag or parity_apply_ground_collision)
    parity_drag_damping = _resolve_parity_drag_damping(ir)
    parity_reverse_factor = _resolve_reverse_factor(ir)
    parity_collide_elas, parity_collide_fric = _resolve_ground_contact_params(ir)
    parity_collide_object_elas, parity_collide_object_fric = _resolve_object_contact_params(ir)
    semantic_checks["parity_apply_drag"] = parity_apply_drag
    semantic_checks["parity_apply_ground_collision"] = parity_apply_ground_collision
    semantic_checks["parity_drag_damping"] = float(parity_drag_damping)
    semantic_checks["parity_reverse_factor"] = float(parity_reverse_factor)
    semantic_checks["parity_collide_elas"] = float(parity_collide_elas)
    semantic_checks["parity_collide_fric"] = float(parity_collide_fric)
    semantic_checks["parity_collide_object_elas"] = float(parity_collide_object_elas)
    semantic_checks["parity_collide_object_fric"] = float(parity_collide_object_fric)

    for frame_idx in frame_indices:
        for substep_idx in range(substeps_per_frame):
            if controller_idx.size > 0:
                target_ctrl = _controller_target_for_substep(
                    controller_traj=controller_traj,
                    frame_idx=frame_idx,
                    substep_idx=substep_idx,
                    substeps_per_frame=substeps_per_frame,
                    interpolate_controls=(args.mode == "parity" and args.parity_interpolate_controls),
                )
                target_ctrl_f32 = target_ctrl.astype(np.float32, copy=False)
                assert controller_vel_zero is not None
                controller_vel = controller_vel_zero

                assert controller_idx_wp is not None
                assert controller_target_wp is not None
                assert controller_vel_wp is not None
                controller_target_wp.assign(target_ctrl_f32)
                controller_vel_wp.assign(controller_vel)
                wp.launch(
                    kernel=_set_indexed_particle_state,
                    dim=controller_idx.size,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        controller_idx_wp,
                        controller_target_wp,
                        controller_vel_wp,
                    ],
                    device=model.device,
                )

            if contacts is not None:
                model.collide(state_in, contacts)

            assert solver is not None
            solver.step(state_in, state_out, control, contacts, sim_dt)

            if parity_post_correction_enabled:
                wp.launch(
                    kernel=_apply_parity_post_correction,
                    dim=num_object_points,
                    inputs=[
                        state_in.particle_q,
                        state_out.particle_q,
                        state_out.particle_qd,
                        num_object_points,
                        float(sim_dt),
                        float(parity_drag_damping),
                        int(parity_apply_drag),
                        int(parity_apply_ground_collision),
                        float(parity_reverse_factor),
                        float(parity_collide_elas),
                        float(parity_collide_fric),
                    ],
                    device=model.device,
                )
            state_in, state_out = state_out, state_in

        q_frame = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q_frame)
        rollout_object.append(q_frame[:num_object_points])
    wall_elapsed = time.perf_counter() - wall_start

    rollout_all_np = np.stack(rollout_all, axis=0)
    rollout_object_np = np.stack(rollout_object, axis=0)

    inference_path = args.inference
    if inference_path is None:
        case_name = _as_string(ir["case_name"])
        candidate = ir_path.parent.parent.parent / "inputs" / "cases" / case_name / "inference.pkl"
        inference_path = candidate if candidate.exists() else None
    inference = _load_inference(inference_path)

    rmse_per_frame = None
    mean_rmse = None
    max_rmse = None
    compared_frames = 0
    if inference is not None and inference.ndim == 3 and inference.shape[1] == num_object_points:
        compared_frames = min(frames_to_run, inference.shape[0])
        err = rollout_object_np[:compared_frames] - inference[:compared_frames]
        rmse_per_frame = np.sqrt(np.mean(err * err, axis=(1, 2))).astype(np.float32)
        mean_rmse = float(rmse_per_frame.mean())
        max_rmse = float(rmse_per_frame.max())

    output_prefix = args.output_prefix
    npz_path = out_dir / f"{output_prefix}.npz"
    json_path = out_dir / f"{output_prefix}.json"

    np.savez_compressed(
        npz_path,
        particle_q_all=rollout_all_np,
        particle_q_object=rollout_object_np,
        sim_dt=np.asarray(sim_dt, dtype=np.float32),
        substeps_per_frame=np.asarray(substeps_per_frame, dtype=np.int32),
        frames_run=np.asarray(frames_to_run, dtype=np.int32),
        rmse_per_frame=(
            rmse_per_frame if rmse_per_frame is not None else np.zeros((0,), dtype=np.float32)
        ),
    )

    summary = {
        "ir_path": str(ir_path),
        "output_npz": str(npz_path),
        "device_requested": args.device,
        "device_used": resolved_device,
        "mode": args.mode,
        "ir_version": int(ir_version),
        "solver": solver_name,
        "solver_iterations": int(args.solver_iterations),
        "xpbd_soft_body_relaxation": float(args.xpbd_soft_body_relaxation),
        "xpbd_soft_contact_relaxation": float(args.xpbd_soft_contact_relaxation),
        "xpbd_rigid_contact_relaxation": float(args.xpbd_rigid_contact_relaxation),
        "xpbd_angular_damping": float(args.xpbd_angular_damping),
        "xpbd_enable_restitution": bool(args.xpbd_enable_restitution),
        "semi_spring_ke_scale": float(args.semi_spring_ke_scale),
        "semi_spring_kd_scale": float(args.semi_spring_kd_scale),
        "semi_disable_particle_contact_kernel": bool(args.semi_disable_particle_contact_kernel),
        "semi_enable_tri_contact": bool(args.semi_enable_tri_contact),
        "semi_angular_damping": float(args.semi_angular_damping),
        "semi_friction_smoothing": float(args.semi_friction_smoothing),
        "frames_run": int(frames_to_run),
        "substeps_per_frame": int(substeps_per_frame),
        "sim_dt": float(sim_dt),
        "gravity_scalar": float(gravity_scalar),
        "gravity_vector": [float(v) for v in gravity_vector],
        "wall_time_sec": float(wall_elapsed),
        "particles_total": int(rollout_all_np.shape[1]),
        "particles_object": int(num_object_points),
        "particle_radius_min": float(np.min(radius_used)),
        "particle_radius_max": float(np.max(radius_used)),
        "inactive_particle_count": int(np.sum(flags_used == 0)),
        "shape_contacts_enabled": bool(shape_contacts_enabled),
        "particle_contacts_enabled": bool(particle_contacts_enabled),
        "allow_coupled_contact_radius": bool(args.allow_coupled_contact_radius),
        "contact_semantics": contact_semantics,
        "collision_radius_source": radius_source,
        "frame_sync": args.frame_sync,
        "strict_phystwin": bool(args.strict_phystwin),
        "parity_apply_drag": parity_apply_drag,
        "parity_apply_ground_collision": parity_apply_ground_collision,
        "parity_drag_damping": float(parity_drag_damping),
        "parity_reverse_factor": float(parity_reverse_factor),
        "parity_collide_elas": float(parity_collide_elas),
        "parity_collide_fric": float(parity_collide_fric),
        "parity_collide_object_elas": float(parity_collide_object_elas),
        "parity_collide_object_fric": float(parity_collide_object_fric),
        "semantic_checks": semantic_checks,
        "inference_used": str(inference_path) if inference is not None else None,
        "compared_frames": int(compared_frames),
        "rmse_mean": mean_rmse,
        "rmse_max": max_rmse,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
