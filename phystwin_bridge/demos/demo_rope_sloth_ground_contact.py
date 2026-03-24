#!/usr/bin/env python3
"""One rope drops onto a grounded sloth with cross-object particle contact only."""
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
    load_ir,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
)
from demo_shared import _pair_penalty_contact_force, compute_visual_particle_radii

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_ROOT = Path(__file__).resolve().parents[1]


@wp.kernel
def _eval_cross_object_contact(
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

    is_object_b = i >= split_index
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
        if (index >= split_index) == is_object_b:
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


@wp.kernel
def _apply_drag_correction_ignore_axis_range(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    start: int,
    count: int,
    dt: float,
    damping: float,
    axis_unit: wp.vec3,
):
    local_id = wp.tid()
    tid = start + local_id
    if local_id >= count:
        return
    v = particle_qd[tid]
    v_axis = axis_unit * wp.dot(v, axis_unit)
    v_ortho = v - v_axis
    scale = wp.exp(-dt * damping)
    particle_q[tid] = particle_q[tid] - v_ortho * dt * (1.0 - scale)
    particle_qd[tid] = v_axis + v_ortho * scale


def _default_rope_ir() -> Path:
    return WORKSPACE_ROOT / "tmp" / "rope_double_hand_object_only_test_ir.npz"


def _default_sloth_ir() -> Path:
    return (
        BRIDGE_ROOT
        / "ir"
        / "double_lift_sloth_normal"
        / "phystwin_ir_v2_bf_strict.npz"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One rope free-falls onto a grounded sloth with cross-object particle contact."
    )
    p.add_argument("--rope-ir", type=Path, default=_default_rope_ir())
    p.add_argument("--sloth-ir", type=Path, default=_default_sloth_ir())
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="rope_sloth_ground_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument(
        "--drop-height",
        type=float,
        default=0.35,
        help="Rope bottom height above sloth top, in meters.",
    )
    p.add_argument(
        "--sloth-bottom-z",
        type=float,
        default=-1.0,
        help="Target sloth bottom height; negative means auto-rest on ground using radius.",
    )
    p.add_argument("--rope-object-mass", type=float, default=None)
    p.add_argument("--sloth-object-mass", type=float, default=None)
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
        default=(-0.7, 0.65, 0.45),
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=-8.0)
    p.add_argument("--camera-yaw", type=float, default=-38.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.0008)
    p.add_argument(
        "--particle-radius-scale",
        type=float,
        default=1.0,
        help="Physical particle radius scale applied to both rope and sloth after IR/contact-distance mapping.",
    )
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--rope-line-width", type=float, default=0.01)
    p.add_argument("--sloth-line-width", type=float, default=0.004)
    p.add_argument("--rope-spring-stride", type=int, default=4)
    p.add_argument("--sloth-spring-stride", type=int, default=12)
    p.add_argument("--render-sloth-springs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _prepare_object_ir(
    ir: dict[str, np.ndarray], object_mass_override: float | None
) -> tuple[dict[str, Any], int, np.ndarray, np.ndarray, np.ndarray]:
    ir_demo: dict[str, Any] = {}
    for key, value in ir.items():
        ir_demo[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    reverse_z = bool(newton_import_ir.ir_bool(ir_demo, "reverse_z", default=False))
    n_obj = int(np.asarray(ir_demo["num_object_points"]).ravel()[0])

    x0 = np.asarray(ir_demo["x0"], dtype=np.float32).copy()[:n_obj]
    v0 = np.asarray(ir_demo["v0"], dtype=np.float32).copy()[:n_obj]
    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0

    if object_mass_override is None:
        mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()[:n_obj]
    else:
        mass = np.full(n_obj, float(object_mass_override), dtype=np.float32)

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = mass
    ir_demo["collision_radius"] = np.asarray(ir_demo["collision_radius"], dtype=np.float32).copy()[:n_obj]
    ir_demo["num_object_points"] = np.asarray(n_obj, dtype=np.int32)
    ir_demo["reverse_z"] = np.asarray(False)

    edges = np.asarray(ir_demo["spring_edges"], dtype=np.int32)
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    ir_demo["spring_edges"] = edges[keep].astype(np.int32, copy=True)
    for key in ("spring_ke", "spring_kd", "spring_rest_length", "spring_y", "spring_y_raw"):
        if key in ir_demo:
            ir_demo[key] = np.asarray(ir_demo[key], dtype=np.float32).copy().ravel()[keep]

    ir_demo.pop("controller_idx", None)
    ir_demo.pop("controller_traj", None)
    return ir_demo, n_obj, x0, edges[keep].astype(np.int32, copy=True), ir_demo["collision_radius"].copy()


def _per_object_particle_radius(ir: dict[str, Any], radius_scale: float = 1.0) -> np.ndarray:
    radius = np.asarray(ir["collision_radius"], dtype=np.float32).copy()
    if "contact_collision_dist" in ir:
        dist = float(newton_import_ir.ir_scalar(ir, "contact_collision_dist"))
        radius.fill(max(dist * 0.5, float(newton_import_ir.EPSILON)))
    radius *= float(radius_scale)
    return radius


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
    newton_import_ir._configure_ground_contact_material(gcfg, ir, cfg, checks, context="custom_ground")
    return {"ke": float(gcfg.ke), "kd": float(gcfg.kd), "kf": float(gcfg.kf), "mu": float(gcfg.mu)}


def _build_combined_ir(
    rope_ir_raw: dict[str, np.ndarray], sloth_ir_raw: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    rope_ir, rope_n, rope_x0, rope_edges, rope_radii = _prepare_object_ir(
        rope_ir_raw, args.rope_object_mass
    )
    sloth_ir, sloth_n, sloth_x0, sloth_edges, sloth_radii = _prepare_object_ir(
        sloth_ir_raw, args.sloth_object_mass
    )

    rope_bbox_min = rope_x0.min(axis=0)
    rope_bbox_max = rope_x0.max(axis=0)
    rope_center_xy = 0.5 * (rope_bbox_min[:2] + rope_bbox_max[:2])

    sloth_bbox_min = sloth_x0.min(axis=0)
    sloth_bbox_max = sloth_x0.max(axis=0)
    sloth_center_xy = 0.5 * (sloth_bbox_min[:2] + sloth_bbox_max[:2])

    radius_scale = float(args.particle_radius_scale)
    sloth_ground_radius = float(sloth_radii.max()) * radius_scale
    sloth_bottom_z = (
        sloth_ground_radius if float(args.sloth_bottom_z) < 0.0 else float(args.sloth_bottom_z)
    )
    sloth_shift = np.array(
        [
            -float(sloth_center_xy[0]),
            -float(sloth_center_xy[1]),
            sloth_bottom_z - float(sloth_bbox_min[2]),
        ],
        dtype=np.float32,
    )
    sloth_x = sloth_x0 + sloth_shift
    sloth_top_z = float(sloth_x[:, 2].max())

    rope_shift = np.array(
        [
            -float(rope_center_xy[0]),
            -float(rope_center_xy[1]),
            sloth_top_z + float(args.drop_height) - float(rope_bbox_min[2]),
        ],
        dtype=np.float32,
    )
    rope_x = rope_x0 + rope_shift

    rope_v = np.asarray(rope_ir["v0"], dtype=np.float32).copy()
    sloth_v = np.asarray(sloth_ir["v0"], dtype=np.float32).copy()

    combined: dict[str, Any] = {}
    for key, value in sloth_ir.items():
        combined[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    combined["x0"] = np.concatenate([rope_x, sloth_x], axis=0).astype(np.float32, copy=False)
    combined["v0"] = np.concatenate([rope_v, sloth_v], axis=0).astype(np.float32, copy=False)
    combined["mass"] = np.concatenate(
        [
            np.asarray(rope_ir["mass"], dtype=np.float32).copy(),
            np.asarray(sloth_ir["mass"], dtype=np.float32).copy(),
        ],
        axis=0,
    )
    combined["collision_radius"] = np.concatenate([rope_radii, sloth_radii], axis=0).astype(np.float32, copy=False)
    combined["num_object_points"] = np.asarray(rope_n + sloth_n, dtype=np.int32)
    combined["reverse_z"] = np.asarray(False)
    combined["self_collision"] = np.asarray(True)

    combined["spring_edges"] = np.concatenate([rope_edges, sloth_edges + rope_n], axis=0).astype(np.int32, copy=False)
    for key in ("spring_ke", "spring_kd", "spring_rest_length", "spring_y", "spring_y_raw"):
        if key in rope_ir and key in sloth_ir:
            combined[key] = np.concatenate(
                [np.asarray(rope_ir[key], dtype=np.float32).ravel(), np.asarray(sloth_ir[key], dtype=np.float32).ravel()],
                axis=0,
            ).astype(np.float32, copy=False)

    combined.pop("controller_idx", None)
    combined.pop("controller_traj", None)
    combined["_rope_ir"] = rope_ir
    combined["_sloth_ir"] = sloth_ir
    combined["_rope_range"] = np.asarray([0, rope_n], dtype=np.int32)
    combined["_sloth_range"] = np.asarray([rope_n, rope_n + sloth_n], dtype=np.int32)
    combined["_rope_shift"] = rope_shift.astype(np.float32, copy=False)
    combined["_sloth_shift"] = sloth_shift.astype(np.float32, copy=False)
    combined["_sloth_top_z"] = np.asarray(sloth_top_z, dtype=np.float32)
    combined["_sloth_bottom_z"] = np.asarray(sloth_bottom_z, dtype=np.float32)
    combined["_rope_render_edges"] = rope_edges[:: max(1, int(args.rope_spring_stride))].astype(np.int32, copy=False)
    combined["_sloth_render_edges"] = sloth_edges[:: max(1, int(args.sloth_spring_stride))].astype(np.int32, copy=False) + rope_n

    meta = {
        "rope_range": combined["_rope_range"],
        "sloth_range": combined["_sloth_range"],
        "rope_shift": combined["_rope_shift"],
        "sloth_shift": combined["_sloth_shift"],
        "sloth_top_z": float(sloth_top_z),
        "sloth_bottom_z": float(sloth_bottom_z),
        "rope_render_edges": combined["_rope_render_edges"],
        "sloth_render_edges": combined["_sloth_render_edges"],
        "rope_ir": rope_ir,
        "sloth_ir": sloth_ir,
    }
    return combined, meta


def build_model(ir_obj: dict[str, Any], args: argparse.Namespace, device: str):
    rope_ir = ir_obj["_rope_ir"]
    sloth_ir = ir_obj["_sloth_ir"]

    cfg = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
        shape_contacts=False,
        add_ground_plane=False,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=False,
        device=device,
    )

    rope_checks = newton_import_ir.validate_ir_physics(rope_ir, cfg)
    sloth_checks = newton_import_ir.validate_ir_physics(sloth_ir, cfg)

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    builder.add_ground_plane()

    rope_radius = _per_object_particle_radius(rope_ir, float(args.particle_radius_scale))
    sloth_radius = _per_object_particle_radius(sloth_ir, float(args.particle_radius_scale))

    rope_start, rope_end = [int(v) for v in ir_obj["_rope_range"]]
    sloth_start, sloth_end = [int(v) for v in ir_obj["_sloth_range"]]

    rope_x = np.asarray(ir_obj["x0"], dtype=np.float32)[rope_start:rope_end].copy()
    rope_v = np.asarray(ir_obj["v0"], dtype=np.float32)[rope_start:rope_end].copy()
    rope_m = np.asarray(ir_obj["mass"], dtype=np.float32)[rope_start:rope_end].copy()
    rope_flags = np.full(rope_x.shape[0], int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
    builder.add_particles(
        pos=[tuple(row.tolist()) for row in rope_x],
        vel=[tuple(row.tolist()) for row in rope_v],
        mass=rope_m.astype(float).tolist(),
        radius=rope_radius.astype(float).tolist(),
        flags=rope_flags.astype(int).tolist(),
    )

    sloth_x = np.asarray(ir_obj["x0"], dtype=np.float32)[sloth_start:sloth_end].copy()
    sloth_v = np.asarray(ir_obj["v0"], dtype=np.float32)[sloth_start:sloth_end].copy()
    sloth_m = np.asarray(ir_obj["mass"], dtype=np.float32)[sloth_start:sloth_end].copy()
    sloth_flags = np.full(sloth_x.shape[0], int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
    builder.add_particles(
        pos=[tuple(row.tolist()) for row in sloth_x],
        vel=[tuple(row.tolist()) for row in sloth_v],
        mass=sloth_m.astype(float).tolist(),
        radius=sloth_radius.astype(float).tolist(),
        flags=sloth_flags.astype(int).tolist(),
    )

    for ir_single, start in ((rope_ir, 0), (sloth_ir, sloth_start)):
        edges = np.asarray(ir_single["spring_edges"], dtype=np.int32)
        ke = np.asarray(ir_single["spring_ke"], dtype=np.float32).ravel() * float(args.spring_ke_scale)
        kd = np.asarray(ir_single["spring_kd"], dtype=np.float32).ravel() * float(args.spring_kd_scale)
        rest = np.asarray(ir_single["spring_rest_length"], dtype=np.float32).ravel()
        for i in range(edges.shape[0]):
            builder.add_spring(
                i=int(edges[i, 0] + start),
                j=int(edges[i, 1] + start),
                ke=float(ke[i]),
                kd=float(kd[i]),
                control=0.0,
            )
            builder.spring_rest_length[-1] = float(rest[i])

    model = builder.finalize(device=device)
    cross_object_grid = model.particle_grid
    model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, sloth_ir)
    model.set_gravity(gravity_vec)

    meta = {
        "rope_range": np.asarray(ir_obj["_rope_range"], dtype=np.int32),
        "sloth_range": np.asarray(ir_obj["_sloth_range"], dtype=np.int32),
        "rope_shift": np.asarray(ir_obj["_rope_shift"], dtype=np.float32),
        "sloth_shift": np.asarray(ir_obj["_sloth_shift"], dtype=np.float32),
        "sloth_top_z": float(ir_obj["_sloth_top_z"]),
        "sloth_bottom_z": float(ir_obj["_sloth_bottom_z"]),
        "rope_render_edges": np.asarray(ir_obj["_rope_render_edges"], dtype=np.int32),
        "sloth_render_edges": np.asarray(ir_obj["_sloth_render_edges"], dtype=np.int32),
        "rope_radius": rope_radius.astype(np.float32, copy=False),
        "sloth_radius": sloth_radius.astype(np.float32, copy=False),
        "rope_contact": _map_object_contact_params(rope_ir, cfg),
        "sloth_contact": _map_object_contact_params(sloth_ir, cfg),
        "rope_ground": _map_ground_contact_params(rope_ir, cfg),
        "sloth_ground": _map_ground_contact_params(sloth_ir, cfg),
        "rope_drag_damping": float(newton_import_ir.ir_scalar(rope_ir, "drag_damping")) if "drag_damping" in rope_ir else 0.0,
        "sloth_drag_damping": float(newton_import_ir.ir_scalar(sloth_ir, "drag_damping")) if "drag_damping" in sloth_ir else 0.0,
    }

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    return model, meta, n_obj, cross_object_grid


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
    cross_object_grid: wp.HashGrid | None,
    split_index: int,
) -> dict[str, Any]:
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = None

    sim_dt = (
        float(args.sim_dt)
        if args.sim_dt is not None
        else float(newton_import_ir.ir_scalar(ir_obj, "sim_dt"))
    )
    substeps = (
        int(args.substeps)
        if args.substeps is not None
        else int(newton_import_ir.ir_scalar(ir_obj, "sim_substeps"))
    )
    substeps = max(1, substeps)
    particle_grid = cross_object_grid
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))

    rope_start, rope_end = [int(v) for v in meta["rope_range"]]
    sloth_start, sloth_end = [int(v) for v in meta["sloth_range"]]
    rope_count = rope_end - rope_start
    sloth_count = sloth_end - sloth_start

    rope_contact = meta["rope_contact"]
    sloth_contact = meta["sloth_contact"]
    rope_ground = meta["rope_ground"]
    sloth_ground = meta["sloth_ground"]

    rope_drag = float(meta["rope_drag_damping"]) * float(args.drag_damping_scale)
    sloth_drag = float(meta["sloth_drag_damping"]) * float(args.drag_damping_scale)

    gravity_vec = np.array([0.0, 0.0, -float(args.gravity_mag)], dtype=np.float32)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

    n_frames = max(2, int(args.frames))
    particle_q_all: list[np.ndarray] = []
    particle_q_object: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in range(n_frames):
        q = state_in.particle_q.numpy().astype(np.float32)
        particle_q_all.append(q.copy())
        particle_q_object.append(q[:n_obj].copy())

        for _ in range(substeps):
            state_in.clear_forces()

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)
                wp.launch(
                    _eval_cross_object_contact,
                    dim=model.particle_count,
                    inputs=[
                        particle_grid.id,
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.particle_radius,
                        model.particle_flags,
                        int(split_index),
                        float(0.5 * (rope_contact["ke"] + sloth_contact["ke"])),
                        float(0.5 * (rope_contact["kd"] + sloth_contact["kd"])),
                        float(0.5 * (rope_contact["kf"] + sloth_contact["kf"])),
                        float(0.5 * (rope_contact["mu"] + sloth_contact["mu"])),
                        float(model.particle_cohesion),
                        float(model.particle_max_radius),
                        state_in.particle_f,
                    ],
                    device=device,
                )

            wp.launch(
                _eval_ground_contact_range,
                dim=rope_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    rope_start,
                    rope_count,
                    float(rope_ground["ke"]),
                    float(rope_ground["kd"]),
                    float(rope_ground["kf"]),
                    float(rope_ground["mu"]),
                    state_in.particle_f,
                ],
                device=device,
            )
            wp.launch(
                _eval_ground_contact_range,
                dim=sloth_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    sloth_start,
                    sloth_count,
                    float(sloth_ground["ke"]),
                    float(sloth_ground["kd"]),
                    float(sloth_ground["kf"]),
                    float(sloth_ground["mu"]),
                    state_in.particle_f,
                ],
                device=device,
            )

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if rope_drag > 0.0:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis_range,
                        dim=rope_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            rope_start,
                            rope_count,
                            sim_dt,
                            rope_drag,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )
            if sloth_drag > 0.0:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis_range,
                        dim=sloth_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            sloth_start,
                            sloth_count,
                            sim_dt,
                            sloth_drag,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )

        if frame in {49, 99, 199, 399, 599, n_frames - 1}:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": np.stack(particle_q_all, axis=0),
        "particle_q_object": np.stack(particle_q_object, axis=0),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": float(wall_time),
        "particle_contacts_enabled": True,
    }


def render_video(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> Path:
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
            shape_colors = {}
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "ground" in name or "plane" in name:
                    shape_colors[idx] = (0.23, 0.26, 0.31)
            if shape_colors:
                viewer.update_shape_colors(shape_colors)
        except Exception:
            pass

        radii = compute_visual_particle_radii(
            model.particle_radius.numpy(),
            radius_scale=float(args.particle_radius_vis_scale),
            radius_cap=float(args.particle_radius_vis_min),
        )
        rope_start, rope_end = [int(v) for v in meta["rope_range"]]
        sloth_start, sloth_end = [int(v) for v in meta["sloth_range"]]

        rope_points_wp = wp.empty((rope_end - rope_start,), dtype=wp.vec3, device=device)
        rope_radii_wp = wp.array(radii[rope_start:rope_end], dtype=wp.float32, device=device)
        rope_colors_wp = wp.full((rope_end - rope_start,), value=wp.vec3(0.96, 0.63, 0.26), dtype=wp.vec3, device=device)
        rope_line_starts_wp = wp.empty((len(meta["rope_render_edges"]),), dtype=wp.vec3, device=device)
        rope_line_ends_wp = wp.empty((len(meta["rope_render_edges"]),), dtype=wp.vec3, device=device)

        sloth_points_wp = wp.empty((sloth_end - sloth_start,), dtype=wp.vec3, device=device)
        sloth_radii_wp = wp.array(radii[sloth_start:sloth_end], dtype=wp.float32, device=device)
        sloth_colors_wp = wp.full((sloth_end - sloth_start,), value=wp.vec3(0.27, 0.72, 0.88), dtype=wp.vec3, device=device)
        sloth_line_starts_wp = wp.empty((len(meta["sloth_render_edges"]),), dtype=wp.vec3, device=device)
        sloth_line_ends_wp = wp.empty((len(meta["sloth_render_edges"]),), dtype=wp.vec3, device=device)

        state = model.state()
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_sim_frames = int(sim_data["particle_q_all"].shape[0])
        sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
        sim_duration = max(float(n_sim_frames - 1) * sim_frame_dt, 0.0)
        video_duration = sim_duration * max(float(args.slowdown), 1.0e-6)
        n_out_frames = max(1, int(round(video_duration * fps_out)))
        if n_out_frames == 1 or sim_duration <= 0.0:
            render_indices = np.zeros((1,), dtype=np.int32)
        else:
            sample_times = np.linspace(0.0, sim_duration, n_out_frames, endpoint=True, dtype=np.float64)
            render_indices = np.clip(np.rint(sample_times / sim_frame_dt).astype(np.int32), 0, n_sim_frames - 1)

        for out_idx, sim_idx in enumerate(render_indices):
            q_all = sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False)
            state.particle_q.assign(q_all)

            sim_t = float(sim_idx) * sim_frame_dt
            viewer.begin_frame(sim_t)
            viewer.log_state(state)

            rope_q = q_all[rope_start:rope_end]
            rope_points_wp.assign(rope_q)
            viewer.log_points("/demo/rope_points", rope_points_wp, rope_radii_wp, rope_colors_wp, hidden=False)
            rope_edges = np.asarray(meta["rope_render_edges"], dtype=np.int32)
            rope_line_starts_wp.assign(q_all[rope_edges[:, 0]])
            rope_line_ends_wp.assign(q_all[rope_edges[:, 1]])
            viewer.log_lines("/demo/rope_springs", rope_line_starts_wp, rope_line_ends_wp, (0.99, 0.78, 0.49), width=float(args.rope_line_width), hidden=False)

            sloth_q = q_all[sloth_start:sloth_end]
            sloth_points_wp.assign(sloth_q)
            viewer.log_points("/demo/sloth_points", sloth_points_wp, sloth_radii_wp, sloth_colors_wp, hidden=False)
            if bool(args.render_sloth_springs):
                sloth_edges = np.asarray(meta["sloth_render_edges"], dtype=np.int32)
                sloth_line_starts_wp.assign(q_all[sloth_edges[:, 0]])
                sloth_line_ends_wp.assign(q_all[sloth_edges[:, 1]])
                viewer.log_lines("/demo/sloth_springs", sloth_line_starts_wp, sloth_line_ends_wp, (0.57, 0.85, 0.98), width=float(args.sloth_line_width), hidden=False)

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        f"SLOW MOTION {float(args.slowdown):.1f}x",
                        "Scene: rope free-falls onto a grounded sloth",
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
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        drop_height=np.float32(args.drop_height),
        sloth_bottom_z=np.float32(meta["sloth_bottom_z"]),
        num_object_points=np.int32(n_obj),
        rope_range=np.asarray(meta["rope_range"], dtype=np.int32),
        sloth_range=np.asarray(meta["sloth_range"], dtype=np.int32),
        rope_shift=np.asarray(meta["rope_shift"], dtype=np.float32),
        sloth_shift=np.asarray(meta["sloth_shift"], dtype=np.float32),
    )
    return scene_npz


def build_summary(
    args: argparse.Namespace,
    ir_obj: dict[str, Any],
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
    out_mp4: Path,
) -> dict[str, Any]:
    sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    sim_duration = max((sim_data["particle_q_all"].shape[0] - 1) * sim_frame_dt, 0.0)
    video_duration = sim_duration * float(args.slowdown)
    rendered_frame_count = max(1, int(round(video_duration * float(args.render_fps))))
    return {
        "experiment": "rope_sloth_ground_drop_object_only",
        "rope_ir_path": str(args.rope_ir.resolve()),
        "sloth_ir_path": str(args.sloth_ir.resolve()),
        "object_only": True,
        "rope_range": np.asarray(meta["rope_range"], dtype=np.int32).tolist(),
        "sloth_range": np.asarray(meta["sloth_range"], dtype=np.int32).tolist(),
        "drop_height_m": float(args.drop_height),
        "sloth_bottom_z": float(meta["sloth_bottom_z"]),
        "rope_object_mass_per_particle": float(np.asarray(ir_obj["mass"], dtype=np.float32)[meta["rope_range"][0]:meta["rope_range"][1]].mean()),
        "sloth_object_mass_per_particle": float(np.asarray(ir_obj["mass"], dtype=np.float32)[meta["sloth_range"][0]:meta["sloth_range"][1]].mean()),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "has_ground_plane": True,
        "has_rigid_body": False,
        "particle_contacts_enabled": bool(sim_data["particle_contacts_enabled"]),
        "cross_object_contact_only": True,
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
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
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
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    rope_ir_raw = load_ir(args.rope_ir)
    sloth_ir_raw = load_ir(args.sloth_ir)
    ir_multi, meta = _build_combined_ir(rope_ir_raw, sloth_ir_raw, args)

    print(
        f"Building rope+grounded-sloth model from {args.rope_ir.resolve()} and {args.sloth_ir.resolve()}",
        flush=True,
    )
    model, meta, n_obj, cross_object_grid = build_model(ir_multi, args, device)
    split_index = int(meta["sloth_range"][0])

    print("Running simulation...", flush=True)
    sim_data = simulate(model, ir_multi, meta, args, n_obj, device, cross_object_grid, split_index)

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
