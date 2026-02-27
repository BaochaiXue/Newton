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
def _apply_phystwin_drag(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    num_object_points: int,
    dt: float,
    drag_damping: float,
):
    """PhysTwin-style drag damping (v <- v*exp(-dt*k)) + matching position correction.

    PhysTwin applies exponential velocity damping each substep before integrating x using
    the damped velocity. Newton's semi-implicit integrator integrates x using the undamped
    v1; we compensate by adjusting x by v1*dt*(1-exp(-dt*k)).
    """
    tid = wp.tid()
    if tid >= num_object_points:
        return

    v1 = particle_qd[tid]
    drag = wp.exp(-dt * drag_damping)
    particle_q[tid] = particle_q[tid] - v1 * dt * (1.0 - drag)
    particle_qd[tid] = v1 * drag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instantiate a PhysTwin IR in Newton and run a native Newton rollout "
            "(XPBD or semi_implicit)."
        )
    )
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="newton_rollout")

    parser.add_argument(
        "--solver",
        choices=["xpbd", "semi_implicit"],
        default="xpbd",
        help="Newton solver backend.",
    )
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument(
        "--spring-ke-scale",
        type=float,
        default=1.0,
        help="Global scale factor applied to IR spring_ke for all solvers.",
    )
    parser.add_argument(
        "--spring-kd-scale",
        type=float,
        default=1.0,
        help="Global scale factor applied to IR spring_kd for all solvers.",
    )
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
        help="Override IR substeps.",
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
        help="Gravity scalar along --up-axis. If unset and --gravity-from-reverse-z is true, derive from IR reverse_z.",
    )
    parser.add_argument(
        "--gravity-mag",
        type=float,
        default=9.8,
        help="Gravity magnitude when deriving gravity from reverse_z.",
    )
    parser.add_argument(
        "--gravity-from-reverse-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --gravity is unset, derive gravity sign from IR reverse_z.",
    )
    parser.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")
    parser.add_argument(
        "--frame-sync",
        choices=["legacy", "phystwin"],
        default="phystwin",
        help=(
            "Rollout frame indexing convention. "
            "'phystwin' keeps frame 0 as initial state and simulates 0->1, 1->2, ..."
        ),
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
        help="Enable shape contact generation via model.collide(...).",
    )
    parser.add_argument(
        "--add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a static Newton ground plane at z=0 using contact params from IR.",
    )
    parser.add_argument(
        "--particle-contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable particle-particle contacts. If unset, follows IR self_collision when available, "
            "otherwise disabled."
        ),
    )
    parser.add_argument(
        "--particle-contact-radius",
        type=float,
        default=1e-5,
        help=(
            "Effective collision radius used when --no-particle-contacts. "
            "Keeps solver kernels active while strongly suppressing internal collisions."
        ),
    )
    parser.add_argument(
        "--object-contact-radius",
        type=float,
        default=None,
        help=(
            "Override contact radius for object particles (0..num_object_points-1). "
            "Useful when disabling particle-particle contacts but still needing stable particle-shape contacts "
            "without a large sphere offset."
        ),
    )
    parser.add_argument(
        "--allow-coupled-contact-radius",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow shape-contacts with particle-contacts disabled. This is approximate because "
            "particle radius affects both paths."
        ),
    )
    parser.add_argument(
        "--controller-inactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated: marking controllers inactive can break spring constraints in XPBD "
            "(inactive particles are skipped by integration and delta application). "
            "Prefer leaving controllers active with mass=0 and tiny radius."
        ),
    )
    parser.add_argument(
        "--interpolate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly interpolate controller trajectories within each frame.",
    )
    parser.add_argument(
        "--strict-physics-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require strict spring/parameter consistency checks when loading IR.",
    )
    parser.add_argument(
        "--apply-drag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply PhysTwin-style drag using IR drag_damping (applied after each Newton substep).",
    )
    parser.add_argument(
        "--drag-damping-scale",
        type=float,
        default=1.0,
        help="Optional scale applied to IR drag_damping when --apply-drag is enabled.",
    )
    parser.add_argument(
        "--particle-mu-override",
        type=float,
        default=None,
        help="Optional override for model.particle_mu.",
    )
    parser.add_argument(
        "--ground-mu-scale",
        type=float,
        default=1.0,
        help="Scale applied to ground-plane friction coefficient loaded from IR.",
    )
    parser.add_argument(
        "--ground-restitution-scale",
        type=float,
        default=1.0,
        help="Scale applied to ground-plane restitution loaded from IR.",
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=None,
        help="Optional PhysTwin baseline inference.pkl for RMSE reporting.",
    )
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


def _resolve_ir_version(ir: dict[str, np.ndarray]) -> int:
    if "ir_version" not in ir:
        return 1
    return int(np.asarray(ir["ir_version"]).reshape(-1)[0])


def _gravity_vector(up_axis: str, gravity_scalar: float) -> tuple[float, float, float]:
    vec = [0.0, 0.0, 0.0]
    axis_to_index = {"X": 0, "Y": 1, "Z": 2}
    vec[axis_to_index[up_axis]] = float(gravity_scalar)
    return tuple(vec)


def _resolve_gravity(args: argparse.Namespace, ir: dict[str, np.ndarray]) -> tuple[float, tuple[float, float, float]]:
    if args.gravity is not None:
        gravity_scalar = float(args.gravity)
    elif bool(args.gravity_from_reverse_z):
        reverse_z = _as_bool(ir["reverse_z"]) if "reverse_z" in ir else False
        gravity_scalar = float(args.gravity_mag) * (1.0 if reverse_z else -1.0)
    else:
        gravity_scalar = 0.0
    return gravity_scalar, _gravity_vector(args.up_axis, gravity_scalar)


def _resolve_particle_contacts_enabled(args: argparse.Namespace, ir: dict[str, np.ndarray]) -> bool:
    if args.particle_contacts is not None:
        return bool(args.particle_contacts)
    if "self_collision" in ir:
        return bool(_as_bool(ir["self_collision"]))
    return False


def _validate_physics_semantics(ir: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, float | str | bool]:
    checks: dict[str, float | str | bool] = {}
    strict = bool(args.strict_physics_checks)

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
    if args.spring_ke_scale <= 0.0:
        raise ValueError(
            f"--spring-ke-scale must be > 0, got {args.spring_ke_scale}."
        )
    if args.spring_kd_scale < 0.0:
        raise ValueError(
            f"--spring-kd-scale must be >= 0, got {args.spring_kd_scale}."
        )
    if args.ground_mu_scale < 0.0:
        raise ValueError(
            f"--ground-mu-scale must be >= 0, got {args.ground_mu_scale}."
        )
    if args.ground_restitution_scale < 0.0:
        raise ValueError(
            f"--ground-restitution-scale must be >= 0, got {args.ground_restitution_scale}."
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
        raise ValueError("Strict physics checks require spring_rest_length in IR.")
    else:
        warnings.warn(
            "IR missing spring_rest_length; spring semantic checks are limited.",
            RuntimeWarning,
        )

    spring_ke_mode = "unknown"
    if "spring_ke_mode" in ir:
        spring_ke_mode = _as_string(ir["spring_ke_mode"])
    checks["spring_ke_mode"] = spring_ke_mode

    if "spring_y" in ir and spring_rest is not None:
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
                "spring_ke does not match spring_y/rest_length under strict checks "
                f"(max_rel_error={float(np.max(rel_err)):.6e})."
            )
    elif strict:
        raise ValueError("Strict physics checks require spring_y and spring_rest_length in IR.")

    if strict and spring_ke_mode != "y_over_rest":
        raise ValueError(
            "Strict physics checks require IR spring_ke_mode=y_over_rest. "
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
                "Controller particles must be kinematic (mass=0) under strict checks. "
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
                "IR v2 missing `collision_radius`; falling back to `radius`.",
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

    # PhysTwin uses `contact_collision_dist` as a pairwise distance threshold (d < dist).
    # Newton particle-particle contact triggers around (radius_i + radius_j), so we map
    # to radius = dist/2. This radius also affects particle-shape contacts, so it must
    # be set consistently even when particle-particle contacts are disabled.
    if "contact_collision_dist" in ir:
        collision_dist = float(_as_scalar(ir["contact_collision_dist"]))
        object_radius = float(max(collision_dist * 0.5, 1e-8))
        radius.fill(object_radius)
        radius_source = "contact_collision_dist_half"
        if "controller_idx" in ir:
            controller_idx = ir["controller_idx"].astype(np.int64, copy=False)
            if controller_idx.size > 0:
                # Controllers are typically inactive for contact kernels; keep them tiny.
                radius[controller_idx] = float(args.particle_contact_radius)

    if args.object_contact_radius is not None:
        override = float(args.object_contact_radius)
        if override <= 0.0:
            raise ValueError(f"--object-contact-radius must be > 0, got {override}.")
        num_object_points = int(np.asarray(ir["num_object_points"]).reshape(-1)[0])
        radius[:num_object_points].fill(override)
        radius_source = "object_contact_radius_override"

    return x0, v0, mass, radius, flags, ir_version, radius_source


def _build_model(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
):
    semantic_checks = _validate_physics_semantics(ir=ir, args=args)
    shape_contacts_enabled = bool(args.shape_contacts)
    particle_contacts_enabled = _resolve_particle_contacts_enabled(args, ir)

    # With the current importer, particle radii always follow IR contact semantics (when available),
    # regardless of whether particle-particle contacts are enabled. Disabling particle contacts only
    # disables the particle-particle kernel, so shape-contacts remain consistent.
    contact_semantics = "exact"

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

    spring_edges = np.asarray(ir["spring_edges"])
    if spring_edges.ndim != 2 or spring_edges.shape[1] != 2:
        raise ValueError(f"Expected spring_edges shape [spring_count, 2], got {spring_edges.shape}")
    spring_count = int(spring_edges.shape[0])

    spring_ke = np.asarray(ir["spring_ke"], dtype=np.float32).reshape(-1)
    spring_kd = np.asarray(ir["spring_kd"], dtype=np.float32).reshape(-1)
    if spring_ke.shape[0] != spring_count:
        raise ValueError(f"Expected spring_ke length {spring_count}, got {spring_ke.shape[0]}")
    if spring_kd.shape[0] != spring_count:
        raise ValueError(f"Expected spring_kd length {spring_count}, got {spring_kd.shape[0]}")

    spring_rest_length = None
    if "spring_rest_length" in ir:
        rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).reshape(-1)
        if rest.shape[0] != spring_count:
            raise ValueError(
                "spring_rest_length count mismatch: "
                f"rest={rest.shape[0]}, edges={spring_count}"
            )
        spring_rest_length = rest

    spring_ke = spring_ke * float(args.spring_ke_scale)
    spring_kd = spring_kd * float(args.spring_kd_scale)
    semantic_checks["spring_ke_scale"] = float(args.spring_ke_scale)
    semantic_checks["spring_kd_scale"] = float(args.spring_kd_scale)

    if args.solver == "semi_implicit":
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

    if bool(args.add_ground_plane):
        ground_cfg = builder.default_shape_cfg.copy()
        if "contact_collide_fric" in ir:
            mu_value = _as_scalar(ir["contact_collide_fric"]) * float(args.ground_mu_scale)
            ground_cfg.mu = float(np.clip(mu_value, 0.0, 2.0))
        if "contact_collide_elas" in ir:
            restitution_value = _as_scalar(ir["contact_collide_elas"]) * float(args.ground_restitution_scale)
            ground_cfg.restitution = float(np.clip(restitution_value, 0.0, 1.0))
        reverse_z = bool(_as_bool(ir["reverse_z"])) if "reverse_z" in ir else False
        # Ground plane at 0 in Newton coordinates. Note particle-shape contacts are sphere-based:
        # particle centers will rest at approximately -radius (or +radius depending on axis).
        if args.up_axis != "Z":
            builder.add_ground_plane(cfg=ground_cfg)
            semantic_checks["ground_plane_reverse_z"] = False
        elif reverse_z:
            # Work around wp.quat_between_vectors ambiguity for opposite vectors (Z -> -Z) by
            # explicitly supplying a transform whose local +Z points along world -Z.
            xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(1.0, 0.0, 0.0, 0.0))
            builder.add_shape_plane(
                body=-1,
                xform=xform,
                width=0.0,
                length=0.0,
                cfg=ground_cfg,
                label="ground_plane_reverse_z",
            )
            semantic_checks["ground_plane_reverse_z"] = True
        else:
            builder.add_ground_plane(cfg=ground_cfg)
            semantic_checks["ground_plane_reverse_z"] = False
        semantic_checks["ground_plane_added"] = True
        semantic_checks["ground_mu"] = float(ground_cfg.mu)
        semantic_checks["ground_restitution"] = float(ground_cfg.restitution)
    else:
        semantic_checks["ground_plane_added"] = False

    model = builder.finalize(device=device)

    semantic_checks["particle_contacts_enabled"] = bool(particle_contacts_enabled)

    # Disable particle-particle contact kernel while keeping particle radii for particle-shape contacts.
    if not particle_contacts_enabled:
        if args.solver == "xpbd":
            model.particle_max_radius = 0.0
            # Also disable hash-grid rebuilds in SolverXPBD (it rebuilds whenever particle_grid is not None).
            model.particle_grid = None
            semantic_checks["xpbd_particle_contacts_disabled"] = True
        else:
            model.particle_grid = None
            semantic_checks["semi_particle_contacts_disabled"] = True
    else:
        semantic_checks["xpbd_particle_contacts_disabled"] = False
        semantic_checks["semi_particle_contacts_disabled"] = False

    # Optional extra guard: allow manually disabling semi-implicit particle contact kernel even if enabled.
    if args.solver == "semi_implicit" and bool(args.semi_disable_particle_contact_kernel):
        model.particle_grid = None
        semantic_checks["semi_particle_contact_kernel_disabled"] = True
    else:
        semantic_checks["semi_particle_contact_kernel_disabled"] = False

    gravity_scalar, gravity_vector = _resolve_gravity(args=args, ir=ir)
    model.set_gravity(gravity_vector)

    if args.particle_mu_override is not None:
        particle_mu = float(np.clip(float(args.particle_mu_override), 0.0, 2.0))
        model.particle_mu = particle_mu
        semantic_checks["particle_mu"] = particle_mu
        semantic_checks["particle_mu_source"] = "override"
    elif "contact_collide_fric" in ir:
        particle_mu = float(np.clip(_as_scalar(ir["contact_collide_fric"]), 0.0, 2.0))
        model.particle_mu = particle_mu
        semantic_checks["particle_mu"] = particle_mu
        semantic_checks["particle_mu_source"] = "ir"

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
    args: argparse.Namespace,
):
    if args.solver == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(
            model,
            angular_damping=float(args.semi_angular_damping),
            friction_smoothing=float(args.semi_friction_smoothing),
            enable_tri_contact=bool(args.semi_enable_tri_contact),
        )
    return newton.solvers.SolverXPBD(
        model,
        iterations=int(args.solver_iterations),
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
    )
    solver = _make_solver(model, args=args)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    shape_contacts_enabled = bool(shape_contacts_enabled)
    particle_contacts_enabled = bool(particle_contacts_enabled)
    contacts_enabled = bool(shape_contacts_enabled or particle_contacts_enabled)
    # Keep a Contacts buffer allocated even when contacts are disabled. XPBD shape contact
    # kernels expect a non-null Contacts object whenever the model has shapes.
    contacts = model.contacts()

    controller_idx = ir["controller_idx"].astype(np.int64)
    controller_traj = ir["controller_traj"].astype(np.float32)
    num_object_points = int(ir["num_object_points"])
    if controller_idx.ndim != 1:
        raise ValueError(f"Expected controller_idx 1D array, got {controller_idx.shape}")
    if controller_traj.ndim != 3 or controller_traj.shape[-1] != 3:
        raise ValueError(f"Expected controller_traj shape [frames, ctrl_count, 3], got {controller_traj.shape}")
    if controller_traj.shape[1] != controller_idx.size:
        raise ValueError(
            "Controller count mismatch between controller_traj and controller_idx: "
            f"traj_ctrl={controller_traj.shape[1]}, idx_ctrl={controller_idx.size}."
        )
    if controller_idx.size > 0:
        particle_count = int(np.asarray(ir["x0"]).shape[0])
        if controller_idx.min() < 0 or controller_idx.max() >= particle_count:
            raise ValueError(
                f"controller_idx out of bounds for particle_count={particle_count}: "
                f"min={int(controller_idx.min())}, max={int(controller_idx.max())}."
            )

    ir_frames = controller_traj.shape[0]
    frames_to_run = max(1, min(int(args.num_frames), ir_frames))

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
        # PhysTwin keeps controller velocities at zero (control points are driven by position only).
        controller_vel_zero = np.zeros((controller_idx.size, 3), dtype=np.float32)

    if args.frame_sync == "phystwin":
        q0 = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q0)
        rollout_object.append(q0[:num_object_points])
        frame_indices = range(1, frames_to_run)
    else:
        frame_indices = range(frames_to_run)

    wall_start = time.perf_counter()
    for frame_idx in frame_indices:
        for substep_idx in range(substeps_per_frame):
            # Newton solvers accumulate forces into state_in.{particle_f,body_f};
            # these buffers must be cleared every substep to avoid force carry-over.
            state_in.clear_forces()
            if controller_idx.size > 0:
                target_ctrl = _controller_target_for_substep(
                    controller_traj=controller_traj,
                    frame_idx=frame_idx,
                    substep_idx=substep_idx,
                    substeps_per_frame=substeps_per_frame,
                    interpolate_controls=bool(args.interpolate_controls),
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

            if contacts_enabled:
                model.collide(state_in, contacts)
            else:
                contacts.clear()

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if bool(args.apply_drag) and "drag_damping" in ir:
                drag_damping = float(_as_scalar(ir["drag_damping"])) * float(args.drag_damping_scale)
                if drag_damping > 0.0:
                    wp.launch(
                        kernel=_apply_phystwin_drag,
                        dim=num_object_points,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            num_object_points,
                            sim_dt,
                            drag_damping,
                        ],
                        device=model.device,
                    )

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
        "ir_version": int(ir_version),
        "solver": args.solver,
        "solver_iterations": int(args.solver_iterations),
        "spring_ke_scale": float(args.spring_ke_scale),
        "spring_kd_scale": float(args.spring_kd_scale),
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
        "gravity_mag": float(args.gravity_mag),
        "gravity_from_reverse_z": bool(args.gravity_from_reverse_z),
        "gravity_vector": [float(v) for v in gravity_vector],
        "wall_time_sec": float(wall_elapsed),
        "particles_total": int(rollout_all_np.shape[1]),
        "particles_object": int(num_object_points),
        "particle_radius_min": float(np.min(radius_used)),
        "particle_radius_max": float(np.max(radius_used)),
        "inactive_particle_count": int(np.sum(flags_used == 0)),
        "shape_contacts_enabled": bool(shape_contacts_enabled),
        "add_ground_plane": bool(args.add_ground_plane),
        "particle_contacts_enabled": bool(particle_contacts_enabled),
        "allow_coupled_contact_radius": bool(args.allow_coupled_contact_radius),
        "object_contact_radius": float(args.object_contact_radius)
        if args.object_contact_radius is not None
        else None,
        "contact_semantics": contact_semantics,
        "collision_radius_source": radius_source,
        "controller_inactive": bool(args.controller_inactive),
        "interpolate_controls": bool(args.interpolate_controls),
        "frame_sync": args.frame_sync,
        "strict_physics_checks": bool(args.strict_physics_checks),
        "particle_mu_override": args.particle_mu_override,
        "ground_mu_scale": float(args.ground_mu_scale),
        "ground_restitution_scale": float(args.ground_restitution_scale),
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
