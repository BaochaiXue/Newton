#!/usr/bin/env python3
"""Two object-only ropes interact on the ground with real rope-rope contact.

This demo targets a simpler multi-rope interaction setup:

- duplicate one object-only rope IR into two ropes in one Newton particle system
- place the lower rope resting on the ground
- drop the upper rope onto the lower rope
- keep Newton particle-particle contact enabled only for cross-rope pairs
- use a fixed close camera for a deterministic offline MP4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from demo_rope_bunny_drop import (
    _apply_drag_correction_ignore_axis,
    _copy_object_only_ir,
    load_ir,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
)
from demo_shared import _pair_penalty_contact_force, apply_viewer_shape_colors, compute_visual_particle_radii

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


ROPE_SPECS = (
    {
        "name": "rope_a",
        "point_color": (0.27, 0.54, 0.88),
        "line_color": (0.60, 0.80, 0.98),
    },
    {
        "name": "rope_b",
        "point_color": (0.96, 0.63, 0.26),
        "line_color": (0.99, 0.78, 0.49),
    },
)


def _rotate_points_z(points: np.ndarray, yaw_deg: float, center_xy: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    if np.isclose(yaw, 0.0):
        return points
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = points.copy()
    xy = rotated[:, :2] - center_xy[None, :]
    rotated[:, :2] = xy @ rot.T + center_xy[None, :]
    return rotated


@wp.kernel
def _eval_cross_rope_contact(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    split_index: int,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return

    rope_i = i >= split_index
    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    f = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        if (index >= split_index) == rope_i:
            continue

        n = x - particle_x[index]
        d = wp.length(n)
        if d < 1.0e-8:
            continue
        err = d - radius - particle_radius[index]
        if err <= k_cohesion:
            n = n / d
            vrel = v - particle_v[index]
            f += _pair_penalty_contact_force(
                n, vrel, err, k_contact, k_damp, k_friction, k_mu
            )

    particle_f[i] = particle_f[i] + f


@wp.kernel
def _eval_ground_contact_range(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    start: int,
    count: int,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    local_id = wp.tid()
    i = start + local_id
    if local_id >= count:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    n = wp.vec3(0.0, 0.0, 1.0)
    c = x[2] - radius
    if c <= 0.0:
        particle_f[i] = particle_f[i] + _pair_penalty_contact_force(
            n, v, c, k_contact, k_damp, k_friction, k_mu
        )


def _default_rope_ir() -> Path:
    return WORKSPACE_ROOT / "tmp" / "rope_double_hand_object_only_test_ir.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One rope rests on the ground while another rope drops onto it with real cross-rope particle contact.")
    p.add_argument("--ir", type=Path, default=_default_rope_ir(), help="Path to rope PhysTwin IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="rope_ground_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--sim-dt", type=float, default=1.0e-4)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument(
        "--drop-height",
        type=float,
        default=0.35,
        help="Vertical gap between the lower rope bottom and the upper rope bottom, in meters.",
    )
    p.add_argument(
        "--lower-rope-bottom-z",
        type=float,
        default=-1.0,
        help="Target lower-rope bottom height in world z. Use negative value for automatic radius-based resting height.",
    )
    p.add_argument(
        "--lower-rope-yaw-deg",
        type=float,
        default=0.0,
        help="World-space yaw rotation [deg] applied to the lower rope before placement.",
    )
    p.add_argument(
        "--upper-rope-yaw-deg",
        type=float,
        default=90.0,
        help="World-space yaw rotation [deg] applied to the upper rope before placement. Default 90 makes a cross.",
    )
    p.add_argument(
        "--object-mass",
        type=float,
        default=1.0,
        help="Fallback per-particle rope mass for ropes without an auto-set total-mass target.",
    )
    p.add_argument(
        "--mass-spring-scale",
        type=float,
        default=None,
        help=(
            "Optional shared scale applied to both ropes before any rope-specific auto-set-weight target. "
            "This exists for compatibility with the object-only rope helpers."
        ),
    )
    p.add_argument(
        "--lower-rope-auto-set-weight",
        type=float,
        default=None,
        help=(
            "Target lower-rope total deformable mass [kg]. If provided, auto-compute one rope-specific "
            "scale so lower-rope mass + spring_ke + spring_kd follow the same ratio."
        ),
    )
    p.add_argument(
        "--upper-rope-auto-set-weight",
        type=float,
        default=None,
        help=(
            "Target upper-rope total deformable mass [kg]. If provided, auto-compute one rope-specific "
            "scale so upper-rope mass + spring_ke + spring_kd follow the same ratio."
        ),
    )
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument(
        "--drag-ignore-gravity-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply drag only orthogonal to gravity so free-fall acceleration is preserved.",
    )
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument("--disable-particle-contact-kernel", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=2.0)
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(-1.55, 1.35, 1.18),
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=-10.0)
    p.add_argument("--camera-yaw", type=float, default=-40.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument(
        "--post-contact-video-seconds",
        type=float,
        default=0.0,
        help=(
            "End the rendered video this many output-video seconds after the first "
            "cross-rope contact. Use <= 0 to disable trimming."
        ),
    )
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.0004)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--rope-line-width", type=float, default=0.01)
    p.add_argument("--spring-stride", type=int, default=4)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _validate_weight_args(args: argparse.Namespace) -> None:
    if args.mass_spring_scale is not None and float(args.mass_spring_scale) <= 0.0:
        raise ValueError(f"--mass-spring-scale must be > 0, got {args.mass_spring_scale}")
    use_auto = (
        args.lower_rope_auto_set_weight is not None
        or args.upper_rope_auto_set_weight is not None
    )
    if use_auto and not np.isclose(float(args.object_mass), 1.0):
        raise ValueError(
            "Per-rope auto-set-weight cannot be used together with a non-default --object-mass. "
            "Use only one source of rope mass control."
        )
    if use_auto and args.mass_spring_scale is not None:
        raise ValueError(
            "Per-rope auto-set-weight cannot be used together with --mass-spring-scale. "
            "The rope-specific weight scales already control mass + spring + contact."
        )
    if use_auto and not np.isclose(float(args.spring_ke_scale), 1.0):
        raise ValueError(
            "Per-rope auto-set-weight cannot be used together with --spring-ke-scale. "
            "The rope-specific weight scales already control spring_ke."
        )
    if use_auto and not np.isclose(float(args.spring_kd_scale), 1.0):
        raise ValueError(
            "Per-rope auto-set-weight cannot be used together with --spring-kd-scale. "
            "The rope-specific weight scales already control spring_kd."
        )
    for flag in ("lower_rope_auto_set_weight", "upper_rope_auto_set_weight"):
        value = getattr(args, flag)
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"--{flag.replace('_', '-')} must be > 0, got {value}")


def _resolve_rope_mass_scales(ir_obj: dict[str, Any], args: argparse.Namespace, n_obj: int) -> tuple[np.ndarray, float]:
    base_mass = np.asarray(ir_obj["mass"], dtype=np.float32).copy()[:n_obj]
    base_total_mass = float(base_mass.sum())
    if base_total_mass <= 0.0:
        raise ValueError(f"Base rope total mass must be > 0, got {base_total_mass}")

    scales = np.ones((len(ROPE_SPECS),), dtype=np.float32)
    targets = (
        args.lower_rope_auto_set_weight,
        args.upper_rope_auto_set_weight,
    )
    for rope_id, target_total in enumerate(targets):
        if target_total is not None:
            scales[rope_id] = np.float32(float(target_total) / base_total_mass)
    return scales, base_total_mass


def _make_single_rope_contact_ir(
    ir_multi: dict[str, Any], rope_range: tuple[int, int]
) -> dict[str, Any]:
    start, end = [int(v) for v in rope_range]
    ir_single: dict[str, Any] = {
        "mass": np.asarray(ir_multi["mass"], dtype=np.float32)[start:end].copy(),
        "num_object_points": np.asarray(end - start, dtype=np.int32),
    }
    for key in (
        "contact_collide_object_elas",
        "contact_collide_object_fric",
        "contact_collide_elas",
        "contact_collide_fric",
    ):
        if key in ir_multi:
            ir_single[key] = np.array(ir_multi[key], copy=True)
    return ir_single


def _map_object_contact_params(ir: dict[str, Any], cfg: newton_import_ir.SimConfig) -> dict[str, float]:
    elas_raw = float(newton_import_ir.ir_scalar(ir, "contact_collide_object_elas"))
    fric_raw = float(newton_import_ir.ir_scalar(ir, "contact_collide_object_fric"))
    mu = float(np.clip(fric_raw, 0.0, newton_import_ir.MU_MAX))
    zeta = float(newton_import_ir._restitution_to_damping_ratio(elas_raw))
    ke = 1.0e3 if cfg.particle_contact_ke is None else float(cfg.particle_contact_ke)
    ke = max(ke, float(newton_import_ir.EPSILON))
    m_ref = float(newton_import_ir._resolve_object_mass_reference(ir))
    m_eff_ref = max(0.5 * m_ref, float(newton_import_ir.EPSILON))
    kd = 2.0 * zeta * np.sqrt(ke * m_eff_ref)
    kf = float(cfg.particle_contact_kf_scale) * kd
    return {"ke": float(ke), "kd": float(kd), "kf": float(kf), "mu": float(mu)}


def _map_ground_contact_params(ir: dict[str, Any], cfg: newton_import_ir.SimConfig) -> dict[str, float]:
    tmp_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    gcfg = tmp_builder.default_shape_cfg.copy()
    checks: dict[str, Any] = {}
    newton_import_ir._configure_ground_contact_material(gcfg, ir, cfg, checks, context="two_ropes_ground")
    return {"ke": float(gcfg.ke), "kd": float(gcfg.kd), "kf": float(gcfg.kf), "mu": float(gcfg.mu)}


def _scale_contact_params(params: dict[str, float], weight_scale: float) -> dict[str, float]:
    alpha = float(weight_scale)
    if np.isclose(alpha, 1.0):
        return dict(params)
    if alpha <= 0.0:
        raise ValueError(f"weight_scale must be > 0, got {alpha}")
    scaled = dict(params)
    scaled["ke"] = float(scaled["ke"] * alpha)
    scaled["kd"] = float(scaled["kd"] * alpha)
    scaled["kf"] = float(scaled["kf"] * alpha)
    return scaled


def _build_multi_rope_ir(ir_obj: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    base_x = np.asarray(ir_obj["x0"], dtype=np.float32).copy()[:n_obj]
    base_v = np.asarray(ir_obj["v0"], dtype=np.float32).copy()[:n_obj]
    base_mass = np.asarray(ir_obj["mass"], dtype=np.float32).copy()[:n_obj]
    base_radius = np.asarray(ir_obj["collision_radius"], dtype=np.float32).copy()[:n_obj]
    rope_mass_scales, base_total_mass = _resolve_rope_mass_scales(ir_obj, args, n_obj)
    base_bbox_min = base_x.min(axis=0)
    base_bbox_max = base_x.max(axis=0)
    base_center_xy = (0.5 * (base_bbox_min[:2] + base_bbox_max[:2])).astype(np.float32)
    rope_yaws = np.asarray(
        [float(args.lower_rope_yaw_deg), float(args.upper_rope_yaw_deg)],
        dtype=np.float32,
    )

    base_edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
    keep = (base_edges[:, 0] < n_obj) & (base_edges[:, 1] < n_obj)
    base_edges = base_edges[keep].astype(np.int32, copy=True)
    spring_data: dict[str, np.ndarray] = {}
    for key in ("spring_ke", "spring_kd", "spring_rest_length", "spring_y", "spring_y_raw"):
        if key in ir_obj:
            spring_data[key] = np.asarray(ir_obj[key], dtype=np.float32).ravel()[keep].copy()

    xs: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    render_edges: list[np.ndarray] = []
    rope_ranges: list[list[int]] = []
    rope_offsets: list[list[float]] = []
    rope_velocities: list[list[float]] = []

    for rope_id, spec in enumerate(ROPE_SPECS):
        start = rope_id * n_obj
        end = start + n_obj

        x = _rotate_points_z(base_x, float(rope_yaws[rope_id]), base_center_xy)
        offset = np.array(
            [0.0, 0.0, 0.0 if rope_id == 0 else float(args.drop_height)],
            dtype=np.float32,
        )
        x += offset[None, :]

        v = np.zeros_like(base_v)
        v[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)[None, :]

        xs.append(x)
        vs.append(v)
        masses.append((base_mass * rope_mass_scales[rope_id]).astype(np.float32, copy=False))
        radii.append(base_radius.copy())
        edges.append(base_edges + start)
        render_edges.append(base_edges[:: max(1, int(args.spring_stride))] + start)
        rope_ranges.append([start, end])
        rope_offsets.append([float(v) for v in offset])
        rope_velocities.append([0.0, 0.0, 0.0])

    ir_multi: dict[str, Any] = {}
    for key, value in ir_obj.items():
        ir_multi[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    ir_multi["x0"] = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    ir_multi["v0"] = np.concatenate(vs, axis=0).astype(np.float32, copy=False)
    ir_multi["mass"] = np.concatenate(masses, axis=0).astype(np.float32, copy=False)
    ir_multi["collision_radius"] = np.concatenate(radii, axis=0).astype(np.float32, copy=False)
    ir_multi["spring_edges"] = np.concatenate(edges, axis=0).astype(np.int32, copy=False)
    for key, arr in spring_data.items():
        if key in ("spring_ke", "spring_kd"):
            ir_multi[key] = np.concatenate(
                [(arr * rope_mass_scales[rope_id]).astype(np.float32, copy=False) for rope_id in range(len(ROPE_SPECS))],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            ir_multi[key] = np.concatenate([arr.copy() for _ in ROPE_SPECS], axis=0).astype(np.float32, copy=False)
    ir_multi["num_object_points"] = np.asarray(len(ROPE_SPECS) * n_obj, dtype=np.int32)
    ir_multi["self_collision"] = np.asarray(True)
    ir_multi["reverse_z"] = np.asarray(False)
    ir_multi.pop("controller_idx", None)
    ir_multi.pop("controller_traj", None)

    meta = {
        "rope_particle_ranges": np.asarray(rope_ranges, dtype=np.int32),
        "rope_initial_offsets": np.asarray(rope_offsets, dtype=np.float32),
        "rope_initial_velocities": np.asarray(rope_velocities, dtype=np.float32),
        "render_edges": [np.asarray(e, dtype=np.int32) for e in render_edges],
        "rope_count": len(ROPE_SPECS),
        "rope_mass_scales": rope_mass_scales.astype(np.float32, copy=False),
        "rope_total_masses": np.asarray([float(m.sum()) for m in masses], dtype=np.float32),
        "rope_object_mass_per_particle": np.asarray([float(m.mean()) for m in masses], dtype=np.float32),
        "base_rope_total_mass": float(base_total_mass),
        "rope_yaws_deg": rope_yaws.astype(np.float32, copy=False),
    }
    return ir_multi, meta


def build_model(ir_obj: dict[str, Any], args: argparse.Namespace, device: str):
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
        add_ground_plane=True,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=True,
        device=device,
    )

    checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)
    particle_contacts = newton_import_ir._resolve_particle_contacts(cfg, ir_obj)

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    _radius, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    newton_import_ir._add_ground_plane(builder, ir_obj, cfg, checks)

    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    rope_center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    base_radius = float(np.asarray(ir_obj["collision_radius"], dtype=np.float32).max())
    lower_rope_bottom_z = (
        base_radius if float(args.lower_rope_bottom_z) < 0.0 else float(args.lower_rope_bottom_z)
    )
    shift = np.array(
        [
            -float(rope_center_xy[0]),
            -float(rope_center_xy[1]),
            lower_rope_bottom_z - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]

    model = builder.finalize(device=device)
    cross_rope_contact_grid = model.particle_grid
    model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)

    rope_ranges = np.asarray(ir_obj["_rope_particle_ranges"], dtype=np.int32)
    rope_a_ir = _make_single_rope_contact_ir(ir_obj, tuple(rope_ranges[0].tolist()))
    rope_b_ir = _make_single_rope_contact_ir(ir_obj, tuple(rope_ranges[1].tolist()))
    rope_mass_scales = np.asarray(ir_obj["_rope_mass_scales"], dtype=np.float32)
    rope_contact_params = [
        _scale_contact_params(_map_object_contact_params(rope_a_ir, cfg), float(rope_mass_scales[0])),
        _scale_contact_params(_map_object_contact_params(rope_b_ir, cfg), float(rope_mass_scales[1])),
    ]
    rope_ground_params = [
        _scale_contact_params(_map_ground_contact_params(rope_a_ir, cfg), float(rope_mass_scales[0])),
        _scale_contact_params(_map_ground_contact_params(rope_b_ir, cfg), float(rope_mass_scales[1])),
    ]
    cross_rope_contact = {
        key: float(0.5 * (rope_contact_params[0][key] + rope_contact_params[1][key]))
        for key in ("ke", "kd", "kf", "mu")
    }

    meta = {
        "rope_shift": shift.astype(np.float32, copy=False),
        "lower_rope_bottom_z": float(lower_rope_bottom_z),
        "upper_rope_gap_z": float(args.drop_height),
        "rope_particle_ranges": np.asarray(ir_obj["_rope_particle_ranges"], dtype=np.int32),
        "rope_initial_offsets": np.asarray(ir_obj["_rope_initial_offsets"], dtype=np.float32),
        "rope_initial_velocities": np.asarray(ir_obj["_rope_initial_velocities"], dtype=np.float32),
        "render_edges": [np.asarray(e, dtype=np.int32) for e in ir_obj["_render_edges"]],
        "rope_count": int(ir_obj["_rope_count"]),
        "rope_mass_scales": np.asarray(ir_obj["_rope_mass_scales"], dtype=np.float32),
        "rope_total_masses": np.asarray(ir_obj["_rope_total_masses"], dtype=np.float32),
        "rope_object_mass_per_particle": np.asarray(ir_obj["_rope_object_mass_per_particle"], dtype=np.float32),
        "base_rope_total_mass": float(np.asarray(ir_obj["_base_rope_total_mass"]).ravel()[0]),
        "rope_yaws_deg": np.asarray(ir_obj["_rope_yaws_deg"], dtype=np.float32),
        "rope_contact_params": rope_contact_params,
        "rope_ground_params": rope_ground_params,
        "cross_rope_contact_params": cross_rope_contact,
        "cross_rope_contact_enabled": True,
    }
    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    return model, meta, n_obj, cross_rope_contact_grid


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
    cross_rope_contact_grid: wp.HashGrid | None,
) -> dict[str, Any]:
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
        add_ground_plane=True,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=True,
        device=device,
    )

    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = None

    sim_dt = float(args.sim_dt)
    substeps = max(1, int(args.substeps))

    particle_grid = cross_rope_contact_grid
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))
    rope_ranges = np.asarray(ir_obj["_rope_particle_ranges"], dtype=np.int32)
    split_index = int(rope_ranges[1, 0])
    rope_a_start, rope_a_end = [int(v) for v in rope_ranges[0]]
    rope_b_start, rope_b_end = [int(v) for v in rope_ranges[1]]
    rope_a_count = rope_a_end - rope_a_start
    rope_b_count = rope_b_end - rope_b_start
    cross_rope_contact = dict(meta["cross_rope_contact_params"])
    rope_ground_params = list(meta["rope_ground_params"])

    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

    n_frames = max(2, int(args.frames))
    particle_q_all: list[np.ndarray] = []
    particle_q_object: list[np.ndarray] = []
    body_q: list[np.ndarray] = []
    body_vel: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in range(n_frames):
        q = state_in.particle_q.numpy().astype(np.float32)
        particle_q_all.append(q.copy())
        particle_q_object.append(q[:n_obj].copy())
        if state_in.body_q is None:
            body_q.append(np.zeros((0, 7), dtype=np.float32))
        else:
            body_q.append(state_in.body_q.numpy().astype(np.float32).copy())
        if state_in.body_qd is None:
            body_vel.append(np.zeros((0, 3), dtype=np.float32))
        else:
            body_vel.append(state_in.body_qd.numpy().astype(np.float32)[:, :3].copy())

        for _ in range(substeps):
            state_in.clear_forces()

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)
                wp.launch(
                    _eval_cross_rope_contact,
                    dim=model.particle_count,
                    inputs=[
                        particle_grid.id,
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.particle_radius,
                        model.particle_flags,
                        split_index,
                        float(cross_rope_contact["ke"]),
                        float(cross_rope_contact["kd"]),
                        float(cross_rope_contact["kf"]),
                        float(cross_rope_contact["mu"]),
                        float(model.particle_cohesion),
                        float(model.particle_max_radius),
                        state_in.particle_f,
                    ],
                    device=device,
                )

            wp.launch(
                _eval_ground_contact_range,
                dim=rope_a_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    rope_a_start,
                    rope_a_count,
                    float(rope_ground_params[0]["ke"]),
                    float(rope_ground_params[0]["kd"]),
                    float(rope_ground_params[0]["kf"]),
                    float(rope_ground_params[0]["mu"]),
                    state_in.particle_f,
                ],
                device=device,
            )
            wp.launch(
                _eval_ground_contact_range,
                dim=rope_b_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    rope_b_start,
                    rope_b_count,
                    float(rope_ground_params[1]["ke"]),
                    float(rope_ground_params[1]["kd"]),
                    float(rope_ground_params[1]["kf"]),
                    float(rope_ground_params[1]["mu"]),
                    state_in.particle_f,
                ],
                device=device,
            )

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis,
                        dim=n_obj,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            n_obj,
                            sim_dt,
                            drag,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        newton_import_ir._apply_drag_correction,
                        dim=n_obj,
                        inputs=[state_in.particle_q, state_in.particle_qd, n_obj, sim_dt, drag],
                        device=device,
                    )

        if frame == 49 or frame == 99 or frame == 199 or frame == 399 or frame == 599 or frame == n_frames - 1:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": np.stack(particle_q_all, axis=0),
        "particle_q_object": np.stack(particle_q_object, axis=0),
        "body_q": np.stack(body_q, axis=0),
        "body_vel": np.stack(body_vel, axis=0),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": float(wall_time),
        "particle_contacts_enabled": True,
    }


def render_video(model: newton.Model, sim_data: dict[str, Any], meta: dict[str, Any], args: argparse.Namespace, device: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps)
    out_mp4 = args.out_dir / f"{args.prefix}.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps_out:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=bool(args.viewer_headless))
    try:
        viewer.set_model(model)
        viewer.show_particles = False
        viewer.show_triangles = True
        viewer.show_visual = True
        viewer.show_static = True
        viewer.show_collision = True
        viewer.show_contacts = False
        viewer.show_ui = False
        viewer.picking_enabled = False
        try:
            viewer.camera.fov = float(args.camera_fov)
        except Exception:
            pass
        cam_pos = np.asarray(args.camera_pos, dtype=np.float32)
        viewer.set_camera(
            wp.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            float(args.camera_pitch),
            float(args.camera_yaw),
        )

        try:
            apply_viewer_shape_colors(viewer, model)
        except Exception:
            pass

        physical_radii = model.particle_radius.numpy().astype(np.float32, copy=False)
        render_radii = compute_visual_particle_radii(
            physical_radii,
            radius_scale=float(args.particle_radius_vis_scale),
            radius_cap=float(args.particle_radius_vis_min),
        )
        rope_ranges = np.asarray(meta["rope_particle_ranges"], dtype=np.int32)

        neighbor_offsets = (
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
            (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
            (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
            (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0, 0, -1), (0, 0, 0), (0, 0, 1),
            (0, 1, -1), (0, 1, 0), (0, 1, 1),
            (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1),
            (1, 1, -1), (1, 1, 0), (1, 1, 1),
        )

        def _first_cross_rope_contact_frame() -> int | None:
            q_obj_all = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
            if q_obj_all.ndim != 3 or q_obj_all.shape[0] == 0:
                return None

            rope_a_start, rope_a_end = [int(v) for v in rope_ranges[0]]
            rope_b_start, rope_b_end = [int(v) for v in rope_ranges[1]]
            rope_a_radius = physical_radii[rope_a_start:rope_a_end]
            rope_b_radius = physical_radii[rope_b_start:rope_b_end]
            max_query_radius = float(np.max(rope_a_radius) + np.max(rope_b_radius) + model.particle_cohesion)
            cell_size = max(max_query_radius, 1.0e-6)

            for frame_idx in range(q_obj_all.shape[0]):
                rope_a_q = q_obj_all[frame_idx, rope_a_start:rope_a_end]
                rope_b_q = q_obj_all[frame_idx, rope_b_start:rope_b_end]

                rope_b_cells = np.floor(rope_b_q / cell_size).astype(np.int32, copy=False)
                buckets: dict[tuple[int, int, int], list[int]] = {}
                for j, cell in enumerate(rope_b_cells):
                    key = (int(cell[0]), int(cell[1]), int(cell[2]))
                    buckets.setdefault(key, []).append(j)

                rope_a_cells = np.floor(rope_a_q / cell_size).astype(np.int32, copy=False)
                for i, cell in enumerate(rope_a_cells):
                    base = (int(cell[0]), int(cell[1]), int(cell[2]))
                    p_i = rope_a_q[i]
                    r_i = float(rope_a_radius[i])
                    for dx, dy, dz in neighbor_offsets:
                        cand_ids = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                        if not cand_ids:
                            continue
                        cand_idx = np.asarray(cand_ids, dtype=np.int32)
                        delta = rope_b_q[cand_idx] - p_i[None, :]
                        dist2 = np.sum(delta * delta, axis=1)
                        threshold = r_i + rope_b_radius[cand_idx] + float(model.particle_cohesion)
                        if np.any(dist2 <= threshold * threshold):
                            return int(frame_idx)
            return None

        rope_radii_wp = []
        rope_color_wp = []
        rope_points_wp = []
        rope_line_starts_wp = []
        rope_line_ends_wp = []
        for rope_idx, spec in enumerate(ROPE_SPECS):
            start, end = rope_ranges[rope_idx]
            rope_count = int(end - start)
            rope_points_wp.append(wp.empty((rope_count,), dtype=wp.vec3, device=device))
            rope_radii_wp.append(wp.array(render_radii[start:end], dtype=wp.float32, device=device))
            rope_color_wp.append(
                wp.full(shape=(rope_count,), value=wp.vec3(*spec["point_color"]), dtype=wp.vec3, device=device)
            )
            rope_edges = np.asarray(meta["render_edges"][rope_idx], dtype=np.int32)
            rope_line_count = int(len(rope_edges))
            rope_line_starts_wp.append(wp.empty((rope_line_count,), dtype=wp.vec3, device=device))
            rope_line_ends_wp.append(wp.empty((rope_line_count,), dtype=wp.vec3, device=device))

        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
        if state.body_qd is not None:
            state.body_qd.zero_()

        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_sim_frames = int(sim_data["particle_q_all"].shape[0])
        sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
        first_contact_frame = _first_cross_rope_contact_frame()
        render_end_frame = n_sim_frames - 1
        if first_contact_frame is not None and float(args.post_contact_video_seconds) > 0.0:
            sim_window = float(args.post_contact_video_seconds) / max(float(args.slowdown), 1.0e-6)
            extra_frames = int(round(sim_window / max(sim_frame_dt, 1.0e-12)))
            render_end_frame = min(n_sim_frames - 1, int(first_contact_frame) + max(0, extra_frames))
        sim_duration = max(float(render_end_frame) * sim_frame_dt, 0.0)
        video_duration = sim_duration * max(float(args.slowdown), 1.0e-6)
        n_out_frames = max(1, int(round(video_duration * fps_out)))
        if n_out_frames == 1 or sim_duration <= 0.0:
            render_indices = np.zeros((1,), dtype=np.int32)
        else:
            sample_times = np.linspace(0.0, sim_duration, n_out_frames, endpoint=True, dtype=np.float64)
            render_indices = np.clip(np.rint(sample_times / sim_frame_dt).astype(np.int32), 0, render_end_frame)

        sim_data["first_cross_rope_contact_frame"] = -1 if first_contact_frame is None else int(first_contact_frame)
        sim_data["render_end_frame"] = int(render_end_frame)
        sim_data["rendered_frame_count"] = int(n_out_frames)
        sim_data["video_duration_target_sec"] = float(video_duration)

        for out_idx, sim_idx in enumerate(render_indices):
            state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
            if state.body_q is not None and sim_data["body_q"].shape[1] > 0:
                state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))

            sim_t = float(sim_idx) * sim_frame_dt
            viewer.begin_frame(sim_t)
            viewer.log_state(state)

            q_obj = sim_data["particle_q_object"][sim_idx].astype(np.float32, copy=False)
            for rope_idx, spec in enumerate(ROPE_SPECS):
                start, end = rope_ranges[rope_idx]
                rope_points_wp[rope_idx].assign(q_obj[start:end])
                viewer.log_points(
                    f"/demo/{spec['name']}_points",
                    rope_points_wp[rope_idx],
                    rope_radii_wp[rope_idx],
                    rope_color_wp[rope_idx],
                    hidden=False,
                )

                rope_edges = np.asarray(meta["render_edges"][rope_idx], dtype=np.int32)
                rope_line_starts_wp[rope_idx].assign(q_obj[rope_edges[:, 0]])
                rope_line_ends_wp[rope_idx].assign(q_obj[rope_edges[:, 1]])
                viewer.log_lines(
                    f"/demo/{spec['name']}_springs",
                    rope_line_starts_wp[rope_idx],
                    rope_line_ends_wp[rope_idx],
                    spec["line_color"],
                    width=float(args.rope_line_width),
                    hidden=False,
                )

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        f"SLOW MOTION {float(args.slowdown):.1f}x",
                        (
                            "Scene: lower rope rests on ground, upper rope drops in a cross layout"
                            if abs(float(meta["rope_yaws_deg"][1] - meta["rope_yaws_deg"][0])) > 1.0e-3
                            else "Scene: lower rope rests on ground, upper rope drops onto it"
                        ),
                        f"frame {out_idx + 1:03d}/{n_out_frames:03d}  t={sim_t:.3f}s",
                    ],
                    font_size=int(args.label_font_size),
                )
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(frame.tobytes())

        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        if ffmpeg_proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        ffmpeg_proc = None
    finally:
        try:
            viewer.close()
        except Exception:
            pass
    print(f"  Video saved: {out_mp4}", flush=True)
    return out_mp4


def save_scene_npz(args: argparse.Namespace, sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int) -> Path:
    scene_npz = args.out_dir / f"{args.prefix}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        particle_q_object=sim_data["particle_q_object"],
        body_q=sim_data["body_q"],
        body_vel=sim_data["body_vel"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        drop_height=np.float32(args.drop_height),
        lower_rope_bottom_z=np.float32(meta["lower_rope_bottom_z"]),
        num_object_points=np.int32(n_obj),
        rope_count=np.int32(meta["rope_count"]),
        rope_particle_ranges=np.asarray(meta["rope_particle_ranges"], dtype=np.int32),
        rope_initial_offsets=np.asarray(meta["rope_initial_offsets"], dtype=np.float32),
        rope_initial_velocities=np.asarray(meta["rope_initial_velocities"], dtype=np.float32),
    )
    return scene_npz


def build_summary(args: argparse.Namespace, ir_obj: dict[str, Any], sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int, out_mp4: Path) -> dict[str, Any]:
    body_speed = (
        np.linalg.norm(sim_data["body_vel"][:, 0, :], axis=1)
        if sim_data["body_vel"].shape[1] > 0
        else np.zeros((sim_data["body_vel"].shape[0],), dtype=np.float32)
    )
    sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    sim_duration = max((sim_data["particle_q_all"].shape[0] - 1) * sim_frame_dt, 0.0)
    video_duration = sim_duration * float(args.slowdown)
    rendered_frame_count = max(1, int(round(video_duration * float(args.render_fps))))
    if "rendered_frame_count" in sim_data:
        rendered_frame_count = int(sim_data["rendered_frame_count"])
    if "video_duration_target_sec" in sim_data:
        video_duration = float(sim_data["video_duration_target_sec"])
    return {
        "experiment": "rope_ground_drop_object_only",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "rope_count": int(meta["rope_count"]),
        "rope_particle_ranges": np.asarray(meta["rope_particle_ranges"], dtype=np.int32).tolist(),
        "rope_initial_offsets": np.asarray(meta["rope_initial_offsets"], dtype=np.float32).tolist(),
        "rope_initial_velocities": np.asarray(meta["rope_initial_velocities"], dtype=np.float32).tolist(),
        "rope_mass_scales": np.asarray(meta["rope_mass_scales"], dtype=np.float32).tolist(),
        "base_rope_total_mass": float(meta["base_rope_total_mass"]),
        "rope_total_masses": np.asarray(meta["rope_total_masses"], dtype=np.float32).tolist(),
        "rope_object_mass_per_particle": np.asarray(meta["rope_object_mass_per_particle"], dtype=np.float32).tolist(),
        "rope_yaws_deg": np.asarray(meta["rope_yaws_deg"], dtype=np.float32).tolist(),
        "cross_rope_contact_params": {k: float(v) for k, v in meta["cross_rope_contact_params"].items()},
        "rope_ground_contact_params": [
            {k: float(v) for k, v in rope_params.items()} for rope_params in meta["rope_ground_params"]
        ],
        "rope_particle_contact_params": [
            {k: float(v) for k, v in rope_params.items()} for rope_params in meta["rope_contact_params"]
        ],
        "drop_height_m": float(args.drop_height),
        "lower_rope_bottom_z": float(meta["lower_rope_bottom_z"]),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "has_ground_plane": True,
        "has_rigid_body": False,
        "reverse_z": bool(newton_import_ir.ir_bool(ir_obj, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "frames": int(args.frames),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_frame_dt),
        "sim_duration_sec": float(sim_duration),
        "wall_time_sec": float(sim_data["wall_time"]),
        "slowdown_factor": float(args.slowdown),
        "render_fps": float(args.render_fps),
        "rendered_frame_count": int(rendered_frame_count),
        "video_duration_target_sec": float(video_duration),
        "first_cross_rope_contact_frame": (
            None
            if int(sim_data.get("first_cross_rope_contact_frame", -1)) < 0
            else int(sim_data["first_cross_rope_contact_frame"])
        ),
        "render_end_frame": int(sim_data.get("render_end_frame", sim_data["particle_q_all"].shape[0] - 1)),
        "post_contact_video_seconds": float(args.post_contact_video_seconds),
        "body_speed_initial": float(body_speed[0]),
        "body_speed_final": float(body_speed[-1]),
        "body_speed_max": float(np.max(body_speed)),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "particle_contacts_enabled": bool(sim_data["particle_contacts_enabled"]),
        "cross_rope_contact_only": True,
        "camera_pos": [float(v) for v in args.camera_pos],
        "camera_pitch": float(args.camera_pitch),
        "camera_yaw": float(args.camera_yaw),
        "camera_fov": float(args.camera_fov),
        "camera_mode": "manual",
        "render_video": str(out_mp4),
    }


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    summary_path = args.out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    _validate_weight_args(args)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    ir = load_ir(args.ir)
    ir_obj = _copy_object_only_ir(ir, args)
    ir_multi, ir_multi_meta = _build_multi_rope_ir(ir_obj, args)
    ir_multi["_rope_particle_ranges"] = ir_multi_meta["rope_particle_ranges"]
    ir_multi["_rope_initial_offsets"] = ir_multi_meta["rope_initial_offsets"]
    ir_multi["_rope_initial_velocities"] = ir_multi_meta["rope_initial_velocities"]
    ir_multi["_render_edges"] = ir_multi_meta["render_edges"]
    ir_multi["_rope_count"] = np.asarray(ir_multi_meta["rope_count"], dtype=np.int32)
    ir_multi["_rope_mass_scales"] = np.asarray(ir_multi_meta["rope_mass_scales"], dtype=np.float32)
    ir_multi["_rope_total_masses"] = np.asarray(ir_multi_meta["rope_total_masses"], dtype=np.float32)
    ir_multi["_rope_object_mass_per_particle"] = np.asarray(ir_multi_meta["rope_object_mass_per_particle"], dtype=np.float32)
    ir_multi["_base_rope_total_mass"] = np.asarray(ir_multi_meta["base_rope_total_mass"], dtype=np.float32)
    ir_multi["_rope_yaws_deg"] = np.asarray(ir_multi_meta["rope_yaws_deg"], dtype=np.float32)

    print(f"Building dual-rope ground-drop model from {args.ir.resolve()}", flush=True)
    model, meta, n_obj, cross_rope_contact_grid = build_model(ir_multi, args, device)

    print("Running simulation...", flush=True)
    sim_data = simulate(model, ir_multi, meta, args, n_obj, device, cross_rope_contact_grid)

    scene_npz = save_scene_npz(args, sim_data, meta, n_obj)
    print(f"  Scene NPZ: {scene_npz}", flush=True)

    out_mp4: Path | None = None
    if not args.skip_render:
        print("Rendering video...", flush=True)
        out_mp4 = render_video(model, sim_data, meta, args, device)

    summary = build_summary(args, ir_multi, sim_data, meta, n_obj, out_mp4 or Path(""))
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
