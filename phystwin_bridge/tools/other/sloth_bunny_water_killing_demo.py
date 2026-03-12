#!/usr/bin/env python3
"""Dedicated killing demo: object-only PhysTwin sloth drops into Newton MPM water with two floating bunnies.

Rules for this script:
- The sloth softbody must come from the PhysTwin -> Newton importer path.
- Only object particles + object springs are imported; controller particles are dropped.
- The only intentional physical override is object particle mass = 0.001 kg.
- Newton core is not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from demo_common import (
    CORE_DIR,
    add_dense_particle_forces,
    alpha_shape_surface_mesh,
    apply_drag_range,
    camera_position,
    compute_body_forces,
    ground_grid,
    load_core_module,
    load_ir_checked,
    load_surface_points,
    model_particle_collider_body_ids,
    overlay_text_lines_rgb,
    spawn_mpm_particle_block,
    subtract_body_force,
)

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

path_defaults = load_core_module("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = load_core_module("newton_import_ir", CORE_DIR / "newton_import_ir.py")

import newton  # noqa: E402


@dataclass
class SoftSpec:
    name: str
    case_name: str
    ir_path: Path
    particle_start: int
    particle_count: int
    drag_damping: float
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    mass_sum: float
    placement_shift: np.ndarray
    reverse_z: bool
    collision_radius_mean: float
    collider_mu: float
    color_rgb: tuple[float, float, float]
    strict_export: bool
    spring_ke_mode: str
    spring_ke_rel_error_max: float


@dataclass
class SoftProxy:
    collider_id: int
    interp_particle_ids_global: np.ndarray
    interp_particle_weights: np.ndarray
    mesh_points: wp.array
    mesh_velocities: wp.array
    mesh: Any
    query_points_world: np.ndarray
    collider_thickness: float
    collider_mu: float


@dataclass
class DemoMeta:
    soft_spec: SoftSpec
    bunny_mesh_vertices_local: np.ndarray
    bunny_mesh_tri_indices: np.ndarray
    bunny_mesh_render_edges: np.ndarray
    bunny_mesh_scale: float
    bunny_mesh_asset_path: str
    rigid_labels: list[str]
    rigid_cluster_center: np.ndarray
    pool_half_extent: float
    pool_height: float
    pool_wall_thickness: float
    ground_z: float
    water_surface_z: float
    scene_label: str


@dataclass
class WaterSystem:
    model: newton.Model
    solver: Any
    state_0: newton.State
    state_1: newton.State
    render_points: wp.array
    render_radii: wp.array
    render_colors: wp.array
    collider_impulses: wp.array
    collider_impulse_pos: wp.array
    collider_impulse_ids: wp.array
    collider_body_id: wp.array
    body_water_forces: wp.array
    soft_particle_forces: wp.array
    soft_proxy: SoftProxy
    frame_dt: float
    active_impulse_count: int
    use_particle_render: bool


@dataclass
class ViewSpec:
    name: str
    label: str
    yaw_deg: float
    pitch_deg: float
    distance: float


def _output_stem(prefix: str) -> str:
    return prefix.strip().replace(" ", "_")


def _view_specs() -> list[ViewSpec]:
    return [
        ViewSpec("wide", "Hero Wide", yaw_deg=-34.0, pitch_deg=-12.0, distance=5.2),
        ViewSpec("impact", "Impact Close", yaw_deg=-14.0, pitch_deg=-10.0, distance=4.1),
    ]


def _parse_view_names(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("--views must contain at least one view name")
    known = {spec.name for spec in _view_specs()}
    unknown = [item for item in items if item not in known]
    if unknown:
        raise ValueError(f"Unknown view(s): {unknown}. Known: {sorted(known)}")
    return items


def load_bunny_mesh(asset_name: str, prim_path: str):
    try:
        import newton.examples
        import newton.usd
        from pxr import Usd
    except Exception as exc:
        raise RuntimeError("Bunny mesh requires newton.examples, newton.usd, and pxr") from exc

    asset_path = Path(newton.examples.get_asset(asset_name))
    stage = Usd.Stage.Open(str(asset_path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path} in {asset_path}")
    mesh = newton.usd.get_mesh(prim)
    points = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if indices.size % 3 != 0:
        raise ValueError(f"Bunny mesh indices must be multiple of 3, got {indices.size}")
    tri_indices = indices.reshape(-1, 3)
    e01 = tri_indices[:, [0, 1]]
    e12 = tri_indices[:, [1, 2]]
    e20 = tri_indices[:, [2, 0]]
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0).astype(np.int32, copy=False)
    return mesh, points, tri_indices, all_edges, str(asset_path)


def quat_to_rotmat(q_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _bunny_quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(float(yaw_deg))
    sz = math.sin(0.5 * yaw)
    cz = math.cos(0.5 * yaw)
    tilt = math.radians(90.0)
    sx = math.sin(0.5 * tilt)
    cx = math.cos(0.5 * tilt)
    return (sx * cz, sx * sz, cx * sz, cx * cz)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dedicated killing demo: importer-driven object-only sloth + 2 bunny rigids + Newton MPM water."
    )
    p.add_argument(
        "--sloth-ir",
        type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_sloth_normal/"
            "20260305_163731_full_parity/double_lift_sloth_normal_ir.npz"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="sloth_bunny_water_killing")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--sim-dt", type=float, default=1.1111111e-4)
    p.add_argument("--substeps", type=int, default=300)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--render-only-scene", type=Path, default=None)

    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--ground-extent", type=float, default=8.0)

    p.add_argument("--pool-half-extent", type=float, default=1.0)
    p.add_argument("--pool-height", type=float, default=2.0)
    p.add_argument("--pool-wall-thickness", type=float, default=0.08)
    p.add_argument("--water-fill-height", type=float, default=1.5)

    p.add_argument("--sloth-target-xy", type=float, nargs=2, default=(0.0, 0.0))
    p.add_argument("--sloth-bottom-z", type=float, default=3.0)
    p.add_argument("--particle-mass", type=float, default=0.001)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument("--relax-steps", type=int, default=240)
    p.add_argument("--relax-damping", type=float, default=60.0)

    p.add_argument("--bunny-count", type=int, default=2)
    p.add_argument("--bunny-spacing", type=float, default=0.46)
    p.add_argument("--rigid-mass", type=float, default=180.0)
    p.add_argument("--rigid-scale", type=float, default=0.16)
    p.add_argument("--float-submerge-fraction", type=float, default=0.62)
    p.add_argument("--rigid-body-mu", type=float, default=0.30)
    p.add_argument("--rigid-body-ke", type=float, default=800.0)
    p.add_argument("--rigid-body-kd", type=float, default=110.0)
    p.add_argument("--rigid-body-kf", type=float, default=70.0)
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")

    p.add_argument("--water-voxel-size", type=float, default=0.10)
    p.add_argument("--water-particles-per-cell", type=int, default=2)
    p.add_argument("--water-points-per-particle", type=int, default=4)
    p.add_argument("--water-density", type=float, default=950.0)
    p.add_argument("--water-young-modulus", type=float, default=3.0e4)
    p.add_argument("--water-poisson-ratio", type=float, default=0.49)
    p.add_argument("--water-damping", type=float, default=8.0)
    p.add_argument("--water-hardening", type=float, default=0.0)
    p.add_argument("--water-friction", type=float, default=0.0)
    p.add_argument("--water-yield-pressure", type=float, default=1.0e10)
    p.add_argument("--water-tensile-yield-ratio", type=float, default=1.0)
    p.add_argument("--water-yield-stress", type=float, default=0.0)
    p.add_argument("--water-air-drag", type=float, default=1.0)
    p.add_argument("--water-grid-type", choices=["fixed", "sparse"], default="fixed")
    p.add_argument("--water-max-iterations", type=int, default=50)
    p.add_argument("--render-mpm-particles", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)
    p.add_argument("--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render-soft-springs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--render-edge-stride", type=int, default=14)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.2)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.012)

    p.add_argument("--soft-mpm-force-scale", type=float, default=0.12)
    p.add_argument("--soft-mpm-max-dv-per-frame", type=float, default=0.015)

    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--camera-fov", type=float, default=44.0)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--views", default="wide")
    return p.parse_args()


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    return load_ir_checked(
        path,
        newton_import_ir,
        [
            "x0",
            "v0",
            "mass",
            "num_object_points",
            "spring_edges",
            "spring_rest_length",
            "spring_ke",
            "spring_kd",
            "sim_dt",
            "sim_substeps",
            "collision_radius",
        ],
    )


def _filter_object_only_ir(ir: dict[str, np.ndarray], *, particle_mass: float) -> dict[str, np.ndarray]:
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    reverse_z = bool(newton_import_ir.ir_bool(ir, "reverse_z", default=False))
    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)

    x0 = np.asarray(ir["x0"][:n_obj], dtype=np.float32).copy()
    v0 = np.zeros((n_obj, 3), dtype=np.float32)
    if reverse_z:
        x0[:, 2] *= -1.0

    out: dict[str, np.ndarray] = {
        "ir_version": np.asarray(ir.get("ir_version", np.array([2], dtype=np.int32))).copy(),
        "x0": x0,
        "v0": v0,
        "mass": np.full(n_obj, float(particle_mass), dtype=np.float32),
        "num_object_points": np.asarray(n_obj, dtype=np.int32),
        "collision_radius": np.asarray(ir["collision_radius"][:n_obj], dtype=np.float32).copy(),
        "spring_edges": np.asarray(edges[keep], dtype=np.int32).copy(),
        "spring_rest_length": np.asarray(ir["spring_rest_length"], dtype=np.float32).ravel()[keep].copy(),
        "spring_ke": np.asarray(ir["spring_ke"], dtype=np.float32).ravel()[keep].copy(),
        "spring_kd": np.asarray(ir["spring_kd"], dtype=np.float32).ravel()[keep].copy(),
        "reverse_z": np.asarray(False),
        "sim_dt": np.asarray(ir["sim_dt"]).copy(),
        "sim_substeps": np.asarray(ir["sim_substeps"]).copy(),
    }
    for key in (
        "case_name",
        "strict_phystwin_export",
        "spring_ke_mode",
        "contact_collide_fric",
        "contact_collide_elas",
        "contact_collide_object_fric",
        "contact_collide_object_elas",
        "contact_collision_dist",
        "drag_damping",
        "spring_y",
    ):
        if key in ir:
            value = np.asarray(ir[key])
            if key == "spring_y":
                out[key] = value.ravel()[keep].astype(np.float32, copy=True)
            else:
                out[key] = value.copy()
    return out


def _load_surface_points(spec: SoftSpec) -> np.ndarray:
    return load_surface_points(
        spec.case_name,
        reverse_z=spec.reverse_z,
        placement_shift=spec.placement_shift,
    )


def _bind_proxy_vertices_to_particles(
    proxy_vertices: np.ndarray,
    object_particles: np.ndarray,
    particle_start: int,
    *,
    bind_k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(object_particles, dtype=np.float64))
    k = min(bind_k, object_particles.shape[0])
    dists, idx = tree.query(np.asarray(proxy_vertices, dtype=np.float64), k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-5)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return particle_start + idx.astype(np.int32), weights.astype(np.float32, copy=False)


def _prepare_soft_object(
    builder: newton.ModelBuilder,
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> SoftSpec:
    reverse_z_orig = bool(newton_import_ir.ir_bool(ir, "reverse_z", default=False))
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    orig_mass_obj = np.asarray(ir["mass"][:n_obj], dtype=np.float32)
    positive_mass = orig_mass_obj[orig_mass_obj > 1.0e-8]
    mass_ref = float(np.median(positive_mass)) if positive_mass.size else 1.0
    mass_scale = float(args.particle_mass) / max(mass_ref, 1.0e-8)
    ir_obj = _filter_object_only_ir(ir, particle_mass=float(args.particle_mass))
    ir_obj["x0"] = _relax_object_only_positions(ir_obj, mass_scale, args, device=device)
    x0 = np.asarray(ir_obj["x0"], dtype=np.float32)
    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    shift = np.array(
        [
            float(args.sloth_target_xy[0]) - float(center_xy[0]),
            float(args.sloth_target_xy[1]) - float(center_xy[1]),
            float(args.sloth_bottom_z) - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    ir_obj["x0"] = x0 + shift

    particle_start = len(builder.particle_q)
    cfg = newton_import_ir.SimConfig(
        add_ground_plane=False,
        spring_ke_scale=mass_scale,
        spring_kd_scale=mass_scale,
        particle_contacts=False,
        strict_physics_checks=False,
        object_contact_radius=None,
        particle_contact_radius=1.0e-5,
    )
    checks: dict[str, float] = {}
    radius, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=False)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)

    edges_obj = np.asarray(ir_obj["spring_edges"], dtype=np.int32).copy()
    stride = max(1, int(args.render_edge_stride))
    render_edges = edges_obj[::stride].copy() + particle_start
    point_radius = np.maximum(radius * float(args.particle_radius_vis_scale), float(args.particle_radius_vis_min))
    return SoftSpec(
        name="sloth",
        case_name=str(np.asarray(ir_obj.get("case_name", np.array(["sloth"]))).ravel()[0]),
        ir_path=args.sloth_ir.resolve(),
        particle_start=particle_start,
        particle_count=int(np.asarray(ir_obj["num_object_points"]).ravel()[0]),
        drag_damping=float(newton_import_ir.ir_scalar(ir_obj, "drag_damping", default=0.0)) * float(args.drag_damping_scale),
        render_edge_global=render_edges,
        render_point_radius=point_radius.astype(np.float32),
        mass_sum=float(np.asarray(ir_obj["mass"], dtype=np.float32).sum()),
        placement_shift=shift.astype(np.float32),
        reverse_z=reverse_z_orig,
        collision_radius_mean=float(np.mean(radius)),
        collider_mu=float(newton_import_ir.ir_scalar(ir_obj, "contact_collide_object_fric", default=0.4)),
        color_rgb=(0.98, 0.66, 0.18),
        strict_export=bool(np.asarray(ir_obj.get("strict_phystwin_export", np.array([False]))).ravel()[0]),
        spring_ke_mode=str(np.asarray(ir_obj.get("spring_ke_mode", np.array(["unknown"]))).ravel()[0]),
        spring_ke_rel_error_max=float(newton_import_ir.validate_ir_physics(ir_obj, cfg).get("ke_rel_error_max", float("nan"))),
    )


def _relax_object_only_positions(
    ir_obj: dict[str, np.ndarray],
    mass_scale: float,
    args: argparse.Namespace,
    *,
    device: str,
) -> np.ndarray:
    if int(args.relax_steps) <= 0:
        return np.asarray(ir_obj["x0"], dtype=np.float32).copy()

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    cfg = newton_import_ir.SimConfig(
        add_ground_plane=False,
        spring_ke_scale=float(mass_scale),
        spring_kd_scale=float(mass_scale),
        particle_contacts=False,
        strict_physics_checks=False,
        object_contact_radius=None,
        particle_contact_radius=1.0e-5,
    )
    checks: dict[str, float] = {}
    newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=False)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=False,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    relax_dt = min(float(args.sim_dt), 5.0e-05)
    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    for _ in range(int(args.relax_steps)):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, relax_dt)
        state_in, state_out = state_out, state_in
        wp.launch(
            apply_drag_range,
            dim=n_obj,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                0,
                n_obj,
                relax_dt,
                float(args.relax_damping),
            ],
            device=device,
        )
    return state_in.particle_q.numpy().astype(np.float32)


def _build_soft_proxy(model: newton.Model, spec: SoftSpec, args: argparse.Namespace, device: str) -> SoftProxy:
    state = model.state()
    q0 = state.particle_q.numpy().astype(np.float32)
    qd0 = state.particle_qd.numpy().astype(np.float32) if state.particle_qd is not None else np.zeros_like(q0, dtype=np.float32)
    obj_pts = q0[spec.particle_start : spec.particle_start + spec.particle_count]
    try:
        proxy_vertices, tri_local = alpha_shape_surface_mesh(obj_pts)
    except Exception:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(np.asarray(obj_pts, dtype=np.float64))
        tri_local = np.asarray(hull.simplices, dtype=np.int32)
        used = np.unique(tri_local.reshape(-1))
        remap = {old: i for i, old in enumerate(used.tolist())}
        proxy_vertices = obj_pts[used].astype(np.float32, copy=True)
        tri_local = np.vectorize(remap.get, otypes=[np.int32])(tri_local).astype(np.int32, copy=False)

    interp_ids_global, interp_weights = _bind_proxy_vertices_to_particles(
        proxy_vertices,
        obj_pts,
        spec.particle_start,
    )
    local_ids = interp_ids_global - int(spec.particle_start)
    proxy_points = np.sum(obj_pts[local_ids] * interp_weights[:, :, None], axis=1).astype(np.float32, copy=False)
    proxy_vel = np.sum(qd0[interp_ids_global] * interp_weights[:, :, None], axis=1).astype(np.float32, copy=False)
    mesh_points = wp.array(proxy_points, dtype=wp.vec3, device=device)
    mesh_velocities = wp.array(proxy_vel, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(tri_local.reshape(-1).astype(np.int32), dtype=int, device=device)
    mesh = wp.Mesh(mesh_points, mesh_indices, mesh_velocities)
    return SoftProxy(
        collider_id=-1,
        interp_particle_ids_global=interp_ids_global.astype(np.int32, copy=False),
        interp_particle_weights=interp_weights.astype(np.float32, copy=False),
        mesh_points=mesh_points,
        mesh_velocities=mesh_velocities,
        mesh=mesh,
        query_points_world=proxy_points.astype(np.float32, copy=True),
        collider_thickness=max(float(spec.collision_radius_mean), 0.025 * float(args.water_voxel_size)),
        collider_mu=float(np.clip(spec.collider_mu, 0.0, 2.0)),
    )

def build_model(ir: dict[str, np.ndarray], args: argparse.Namespace, device: str) -> tuple[newton.Model, DemoMeta]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)

    # Import sloth first; all topology comes from the filtered PhysTwin IR.
    soft_spec = _prepare_soft_object(builder, ir, args, device)

    # Infinite ground plane at z=0. The pool sits directly on top of this ground.
    static_cfg = builder.default_shape_cfg.copy()
    static_cfg.mu = 0.32
    static_cfg.ke = 1.5e4
    static_cfg.kd = 6.0e2
    static_cfg.kf = 3.0e2
    static_cfg.is_visible = False
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        width=0.0,
        length=0.0,
        cfg=static_cfg,
        label="ground_plane",
    )

    half = float(args.pool_half_extent)
    wall_t = float(args.pool_wall_thickness)
    wall_h = float(args.pool_height)
    wall_z = 0.5 * wall_h
    for label, pos, hx, hy, hz in (
        ("pool_wall_pos_x", (half + 0.5 * wall_t, 0.0, wall_z), 0.5 * wall_t, half + wall_t, 0.5 * wall_h),
        ("pool_wall_neg_x", (-(half + 0.5 * wall_t), 0.0, wall_z), 0.5 * wall_t, half + wall_t, 0.5 * wall_h),
        ("pool_wall_pos_y", (0.0, half + 0.5 * wall_t, wall_z), half + wall_t, 0.5 * wall_t, 0.5 * wall_h),
        ("pool_wall_neg_y", (0.0, -(half + 0.5 * wall_t), wall_z), half + wall_t, 0.5 * wall_t, 0.5 * wall_h),
    ):
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
            hx=float(hx),
            hy=float(hy),
            hz=float(hz),
            cfg=static_cfg,
            label=label,
        )

    bunny_mesh, bunny_vertices_local, bunny_tri_indices, bunny_render_edges, bunny_asset_path = load_bunny_mesh(
        args.bunny_asset,
        args.bunny_prim,
    )
    mesh_scale = float(args.rigid_scale)
    rigid_vertices = bunny_vertices_local * mesh_scale
    mesh_extent = np.ptp(rigid_vertices, axis=0)
    target_sub = float(max(float(args.float_submerge_fraction) * mesh_extent[2], 0.03))
    target_bottom = float(args.water_fill_height - target_sub)

    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.rigid_body_mu)
    rigid_cfg.ke = float(args.rigid_body_ke)
    rigid_cfg.kd = float(args.rigid_body_kd)
    rigid_cfg.kf = float(args.rigid_body_kf)
    rigid_cfg.is_visible = False

    rigid_labels: list[str] = []
    offsets = np.linspace(-0.5 * float(args.bunny_spacing), 0.5 * float(args.bunny_spacing), int(args.bunny_count), dtype=np.float32)
    yaw_values = [16.0, -20.0]
    for idx, px in enumerate(offsets.tolist()):
        qx, qy, qz, qw = _bunny_quat_xyzw(yaw_values[idx % len(yaw_values)])
        q = wp.quat(qx, qy, qz, qw)
        rotated = (rigid_vertices @ quat_to_rotmat((qx, qy, qz, qw)).T).astype(np.float32)
        z_min = float(rotated[:, 2].min())
        pos_z = float(target_bottom - z_min)
        label = f"water_bunny_{idx}"
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(px), 0.0, pos_z), q),
            mass=float(args.rigid_mass),
            inertia=wp.mat33(
                float(args.rigid_mass * (mesh_extent[1] ** 2 + mesh_extent[2] ** 2) / 12.0),
                0.0,
                0.0,
                0.0,
                float(args.rigid_mass * (mesh_extent[0] ** 2 + mesh_extent[2] ** 2) / 12.0),
                0.0,
                0.0,
                0.0,
                float(args.rigid_mass * (mesh_extent[0] ** 2 + mesh_extent[1] ** 2) / 12.0),
            ),
            lock_inertia=True,
            label=label,
        )
        builder.add_shape_mesh(body=body, mesh=bunny_mesh, scale=(mesh_scale, mesh_scale, mesh_scale), cfg=rigid_cfg)
        rigid_labels.append(label)

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    model.particle_mu = float(newton_import_ir.ir_scalar(ir, "contact_collide_object_fric", default=0.4))

    meta = DemoMeta(
        soft_spec=soft_spec,
        bunny_mesh_vertices_local=bunny_vertices_local.astype(np.float32),
        bunny_mesh_tri_indices=bunny_tri_indices.astype(np.int32),
        bunny_mesh_render_edges=bunny_render_edges.astype(np.int32),
        bunny_mesh_scale=mesh_scale,
        bunny_mesh_asset_path=bunny_asset_path,
        rigid_labels=rigid_labels,
        rigid_cluster_center=np.array([0.0, 0.0, float(args.water_fill_height)], dtype=np.float32),
        pool_half_extent=float(args.pool_half_extent),
        pool_height=float(args.pool_height),
        pool_wall_thickness=float(args.pool_wall_thickness),
        ground_z=0.0,
        water_surface_z=float(args.water_fill_height),
        scene_label="Scene: sloth + water + 2 bunny rigids",
    )
    return model, meta


def build_water_system(model: newton.Model, meta: DemoMeta, args: argparse.Namespace, device: str) -> WaterSystem:
    water_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(water_builder)

    dx = float(args.water_voxel_size)
    margin_xy = max(0.75 * dx, 0.06)
    margin_z = max(0.50 * dx, 0.03)
    headroom = max(1.00 * dx, 0.08)
    lo = np.array(
        [
            -float(args.pool_half_extent) + margin_xy,
            -float(args.pool_half_extent) + margin_xy,
            float(meta.ground_z + margin_z),
        ],
        dtype=np.float32,
    )
    hi = np.array(
        [
            float(args.pool_half_extent) - margin_xy,
            float(args.pool_half_extent) - margin_xy,
            float(args.water_fill_height - headroom),
        ],
        dtype=np.float32,
    )
    water_ids = spawn_mpm_particle_block(
        water_builder,
        bounds_lo=lo,
        bounds_hi=hi,
        voxel_size=dx,
        particles_per_cell=int(args.water_particles_per_cell),
        density=float(args.water_density),
        flags=int(newton.ParticleFlags.ACTIVE),
        jitter_scale=1.0,
    )

    water_model = water_builder.finalize(device=device)
    water_model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    sel = wp.array(water_ids, dtype=int, device=device)
    water_model.mpm.young_modulus[sel].fill_(float(args.water_young_modulus))
    water_model.mpm.poisson_ratio[sel].fill_(float(args.water_poisson_ratio))
    water_model.mpm.damping[sel].fill_(float(args.water_damping))
    water_model.mpm.hardening[sel].fill_(float(args.water_hardening))
    water_model.mpm.friction[sel].fill_(float(args.water_friction))
    water_model.mpm.yield_pressure[sel].fill_(float(args.water_yield_pressure))
    water_model.mpm.tensile_yield_ratio[sel].fill_(float(args.water_tensile_yield_ratio))
    water_model.mpm.yield_stress[sel].fill_(float(args.water_yield_stress))

    options = newton.solvers.SolverImplicitMPM.Options()
    options.voxel_size = dx
    options.tolerance = 1.0e-5
    options.transfer_scheme = "pic"
    options.grid_type = str(args.water_grid_type)
    options.grid_padding = 50 if args.water_grid_type == "fixed" else 0
    options.max_active_cell_count = 1 << 17 if args.water_grid_type == "fixed" else -1
    options.strain_basis = "P0"
    options.max_iterations = int(args.water_max_iterations)
    options.critical_fraction = 0.0
    options.air_drag = float(args.water_air_drag)
    options.collider_velocity_mode = "finite_difference"
    water_solver = newton.solvers.SolverImplicitMPM(water_model, options)

    soft_proxy = _build_soft_proxy(model, meta.soft_spec, args, device)
    body_collider_ids = model_particle_collider_body_ids(model)
    collider_meshes = [None] * len(body_collider_ids) + [soft_proxy.mesh]
    collider_body_ids = body_collider_ids + [None]
    static_thickness = max(float(args.pool_wall_thickness), 0.75 * dx)
    static_projection = 1.10 * dx
    collider_thicknesses = [static_thickness if body_id == -1 else None for body_id in body_collider_ids] + [soft_proxy.collider_thickness]
    collider_projection_thresholds = [static_projection if body_id == -1 else None for body_id in body_collider_ids] + [None]
    collider_friction = [None] * len(body_collider_ids) + [soft_proxy.collider_mu]
    water_solver.setup_collider(
        collider_meshes=collider_meshes,
        collider_body_ids=collider_body_ids,
        collider_thicknesses=collider_thicknesses,
        collider_friction=collider_friction,
        collider_projection_threshold=collider_projection_thresholds,
        model=model,
    )
    soft_proxy.collider_id = len(body_collider_ids)

    combined_state = model.state()
    water_state_0 = water_model.state()
    water_state_1 = water_model.state()
    water_state_0.body_q = wp.empty_like(combined_state.body_q)
    water_state_0.body_qd = wp.empty_like(combined_state.body_qd)
    water_state_0.body_f = wp.empty_like(combined_state.body_f)
    water_state_1.body_q = wp.empty_like(combined_state.body_q)
    water_state_1.body_qd = wp.empty_like(combined_state.body_qd)
    water_state_1.body_f = wp.empty_like(combined_state.body_f)
    water_state_0.body_q.assign(combined_state.body_q)
    water_state_0.body_qd.assign(combined_state.body_qd)
    water_state_0.body_f.assign(combined_state.body_f)
    water_state_1.body_q.assign(combined_state.body_q)
    water_state_1.body_qd.assign(combined_state.body_qd)
    water_state_1.body_f.assign(combined_state.body_f)

    use_particle_render = bool(args.render_mpm_particles)
    if use_particle_render:
        render_points = water_state_0.particle_q
        render_radii = wp.clone(water_model.particle_radius)
        render_colors = wp.full(water_model.particle_count, value=wp.vec3(0.36, 0.84, 0.98), dtype=wp.vec3, device=device)
    else:
        render_points = water_solver.sample_render_grains(water_state_0, int(args.water_points_per_particle))
        grain_radius = dx / max(3.0 * float(args.water_points_per_particle), 1.0)
        render_radii = wp.full(render_points.size, value=grain_radius, dtype=float, device=device)
        render_colors = wp.full(render_points.size, value=wp.vec3(0.36, 0.84, 0.98), dtype=wp.vec3, device=device)

    max_nodes = 1 << 20
    return WaterSystem(
        model=water_model,
        solver=water_solver,
        state_0=water_state_0,
        state_1=water_state_1,
        render_points=render_points,
        render_radii=render_radii,
        render_colors=render_colors,
        collider_impulses=wp.zeros(max_nodes, dtype=wp.vec3, device=device),
        collider_impulse_pos=wp.zeros(max_nodes, dtype=wp.vec3, device=device),
        collider_impulse_ids=wp.full(max_nodes, value=-1, dtype=int, device=device),
        collider_body_id=water_solver.collider_body_index,
        body_water_forces=wp.zeros_like(combined_state.body_f),
        soft_particle_forces=wp.zeros(model.particle_count, dtype=wp.vec3, device=device),
        soft_proxy=soft_proxy,
        frame_dt=float(args.sim_dt) * float(args.substeps),
        active_impulse_count=0,
        use_particle_render=use_particle_render,
    )


def _collect_water_impulses(water: WaterSystem) -> None:
    impulses, pos, ids = water.solver.collect_collider_impulses(water.state_0)
    n = min(int(impulses.shape[0]), int(water.collider_impulses.shape[0]))
    water.collider_impulse_ids.fill_(-1)
    if n > 0:
        water.collider_impulses[:n].assign(impulses[:n])
        water.collider_impulse_pos[:n].assign(pos[:n])
        water.collider_impulse_ids[:n].assign(ids[:n])
    water.active_impulse_count = n


def _update_soft_proxy_from_state(state: newton.State, proxy: SoftProxy) -> None:
    q = state.particle_q.numpy().astype(np.float32)
    qd = state.particle_qd.numpy().astype(np.float32)
    pts = np.sum(q[proxy.interp_particle_ids_global] * proxy.interp_particle_weights[:, :, None], axis=1)
    vel = np.sum(qd[proxy.interp_particle_ids_global] * proxy.interp_particle_weights[:, :, None], axis=1)
    valid = np.all(np.isfinite(pts), axis=1)
    pts = np.asarray(pts, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    if not np.all(valid):
        pts[~valid] = proxy.query_points_world[~valid]
        vel[~valid] = 0.0
    proxy.query_points_world = pts.copy()
    proxy.mesh_points.assign(pts)
    proxy.mesh_velocities.assign(vel)
    proxy.mesh.refit()


def _update_soft_particle_forces_from_water(
    model: newton.Model,
    water: WaterSystem,
    args: argparse.Namespace,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    particle_count = model.particle_count
    forces = np.zeros((particle_count, 3), dtype=np.float32)
    n = int(water.active_impulse_count)
    if n <= 0:
        water.soft_particle_forces.zero_()
        return {"sample_count": 0.0, "impulse_norm_sum": 0.0, "force_norm_max": 0.0}

    ids = water.collider_impulse_ids.numpy()[:n].astype(np.int32, copy=False)
    impulses = water.collider_impulses.numpy()[:n].astype(np.float32, copy=False)
    positions = water.collider_impulse_pos.numpy()[:n].astype(np.float32, copy=False)
    mask = ids == int(water.soft_proxy.collider_id)
    if not np.any(mask):
        water.soft_particle_forces.zero_()
        return {"sample_count": 0.0, "impulse_norm_sum": 0.0, "force_norm_max": 0.0}

    pts = positions[mask]
    imp = impulses[mask] * float(args.soft_mpm_force_scale) / max(float(water.frame_dt), 1.0e-8)
    proxy_points = np.asarray(water.soft_proxy.query_points_world, dtype=np.float64)
    tree = cKDTree(proxy_points)
    k = min(4, proxy_points.shape[0])
    dist, nn = tree.query(pts.astype(np.float64, copy=False), k=k)
    if k == 1:
        dist = dist[:, None]
        nn = nn[:, None]
    proxy_weights = 1.0 / np.maximum(dist.astype(np.float32), 1.0e-5)
    proxy_weights /= np.sum(proxy_weights, axis=1, keepdims=True)
    particle_ids = water.soft_proxy.interp_particle_ids_global[nn]
    particle_weights = water.soft_proxy.interp_particle_weights[nn]
    combo = proxy_weights[:, :, None] * particle_weights
    for proxy_col in range(particle_ids.shape[1]):
        for bind_col in range(particle_ids.shape[2]):
            np.add.at(
                forces,
                particle_ids[:, proxy_col, bind_col],
                imp * combo[:, proxy_col, bind_col : bind_col + 1],
            )

    masses = model.particle_mass.numpy().astype(np.float32)
    max_force = masses * (float(args.soft_mpm_max_dv_per_frame) / max(float(water.frame_dt), 1.0e-8))
    norms = np.linalg.norm(forces, axis=1)
    scale = np.ones_like(norms, dtype=np.float32)
    over = norms > np.maximum(max_force, 1.0e-8)
    scale[over] = max_force[over] / np.maximum(norms[over], 1.0e-8)
    forces *= scale[:, None]

    water.soft_particle_forces.assign(forces)
    return {
        "sample_count": float(pts.shape[0]),
        "impulse_norm_sum": float(np.linalg.norm(imp, axis=1).sum()),
        "force_norm_max": float(np.linalg.norm(forces, axis=1).max(initial=0.0)),
    }


def _pool_outline(meta: DemoMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext = float(meta.pool_half_extent) + float(meta.pool_wall_thickness)
    z0 = float(meta.ground_z)
    z1 = float(meta.pool_height)
    corners = np.asarray(
        [
            (-ext, -ext, z0), (ext, -ext, z0), (ext, ext, z0), (-ext, ext, z0),
            (-ext, -ext, z1), (ext, -ext, z1), (ext, ext, z1), (-ext, ext, z1),
        ],
        dtype=np.float32,
    )
    edges = np.asarray(
        [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
        dtype=np.int32,
    )
    starts = corners[edges[:, 0]]
    ends = corners[edges[:, 1]]
    colors = np.tile(np.asarray([[0.90, 0.94, 1.00]], dtype=np.float32), (starts.shape[0], 1))
    return starts, ends, colors


def simulate(model: newton.Model, meta: DemoMeta, args: argparse.Namespace, water: WaterSystem, device: str) -> dict[str, Any]:
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=bool(args.enable_tri_contact),
    )
    contacts = model.contacts()
    control = model.control()
    state_in = model.state()
    state_out = model.state()
    particle_mass = model.particle_mass.numpy().astype(np.float32)

    q_hist: list[np.ndarray] = []
    body_q_hist: list[np.ndarray] = []
    water_hist: list[np.ndarray] = []

    frame_dt = float(args.sim_dt) * float(args.substeps)
    t0 = time.perf_counter()
    for frame in range(int(args.frames)):
        q_hist.append(state_in.particle_q.numpy().astype(np.float32))
        body_q_hist.append(state_in.body_q.numpy().astype(np.float32))
        if water.use_particle_render:
            water_hist.append(water.state_0.particle_q.numpy().astype(np.float32))
        else:
            water_hist.append(water.render_points.numpy().reshape(-1, 3).astype(np.float32))

        for _ in range(int(args.substeps)):
            state_in.clear_forces()
            if water.active_impulse_count > 0:
                wp.launch(
                    compute_body_forces,
                    dim=water.active_impulse_count,
                    inputs=[
                        frame_dt,
                        water.collider_impulse_ids,
                        water.collider_impulses,
                        water.collider_impulse_pos,
                        water.collider_body_id,
                        state_in.body_q,
                        model.body_com,
                        state_in.body_f,
                    ],
                    device=device,
                )
                water.body_water_forces.assign(state_in.body_f)
            else:
                water.body_water_forces.assign(state_in.body_f)

            wp.launch(
                add_dense_particle_forces,
                dim=model.particle_count,
                inputs=[state_in.particle_f, water.soft_particle_forces, 1.0],
                device=device,
            )
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, float(args.sim_dt))
            state_in, state_out = state_out, state_in
            if meta.soft_spec.drag_damping > 0.0:
                wp.launch(
                    apply_drag_range,
                    dim=meta.soft_spec.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        meta.soft_spec.particle_start,
                        meta.soft_spec.particle_count,
                        float(args.sim_dt),
                        meta.soft_spec.drag_damping,
                    ],
                    device=device,
                )

        _update_soft_proxy_from_state(state_in, water.soft_proxy)
        wp.launch(
            subtract_body_force,
            dim=model.body_count,
            inputs=[
                frame_dt,
                state_in.body_q,
                state_in.body_qd,
                water.body_water_forces,
                model.body_inv_inertia,
                model.body_inv_mass,
                water.state_0.body_q,
                water.state_0.body_qd,
            ],
            device=device,
        )
        water.solver.step(water.state_0, water.state_1, None, None, frame_dt)
        water.solver.project_outside(water.state_1, water.state_1, frame_dt)
        water.solver.update_particle_frames(water.state_0, water.state_1, frame_dt)
        if not water.use_particle_render:
            water.solver.update_render_grains(water.state_0, water.state_1, water.render_points, frame_dt)
        water.state_0, water.state_1 = water.state_1, water.state_0
        _collect_water_impulses(water)
        coupling = _update_soft_particle_forces_from_water(model, water, args)

        q_np = state_in.particle_q.numpy().astype(np.float32)
        body_np = state_in.body_q.numpy().astype(np.float32)[:, :3]
        water_np = water.state_0.particle_q.numpy().astype(np.float32)
        if not np.all(np.isfinite(q_np)):
            raise RuntimeError(f"Softbody state contains non-finite values at frame {frame}")
        if not np.all(np.isfinite(body_np)):
            raise RuntimeError(f"Rigid body state contains non-finite values at frame {frame}")
        if not np.all(np.isfinite(water_np)):
            raise RuntimeError(f"Water state contains non-finite values at frame {frame}")
        soft_bbox = np.max(q_np, axis=0) - np.min(q_np, axis=0)
        if float(np.linalg.norm(soft_bbox)) > 6.0:
            raise RuntimeError(f"Softbody exploded at frame {frame}: bbox={soft_bbox}")
        if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == int(args.frames):
            print(
                f"  frame {frame + 1}/{int(args.frames)}"
                f"  soft_z=[{q_np[:,2].min():.3f},{q_np[:,2].max():.3f}]"
                f"  rigid_z=[{body_np[:,2].min():.3f},{body_np[:,2].max():.3f}]"
                f"  water_z=[{water_np[:,2].min():.3f},{water_np[:,2].max():.3f}]"
                f"  coupling_samples={int(coupling['sample_count'])}",
                flush=True,
            )

    return {
        "particle_q_all": np.stack(q_hist),
        "body_q": np.stack(body_q_hist),
        "water_points_all": np.stack(water_hist),
        "water_point_radii": water.render_radii.numpy().astype(np.float32),
        "water_point_colors": water.render_colors.numpy().astype(np.float32),
        "sim_dt": float(args.sim_dt),
        "substeps": int(args.substeps),
        "wall_time_sec": float(time.perf_counter() - t0),
    }


def render_view_mp4(
    *,
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: DemoMeta,
    args: argparse.Namespace,
    device: str,
    view_spec: ViewSpec,
    out_mp4: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps) / max(float(args.slowdown), 1.0e-6)
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
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=bool(args.viewer_headless))
    ffmpeg_proc = None
    try:
        viewer.set_model(model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.show_visual = False
        viewer.show_static = False
        viewer.show_collision = False
        viewer.show_contacts = False
        viewer.show_ui = False
        viewer.picking_enabled = False
        try:
            viewer.camera.fov = float(args.camera_fov)
            viewer.renderer.sky_upper = (0.16, 0.29, 0.44)
            viewer.renderer.sky_lower = (0.02, 0.05, 0.10)
        except Exception:
            pass

        state = model.state()
        target = np.array([0.0, 0.0, max(meta.water_surface_z * 0.68, 1.0)], dtype=np.float32)
        cpos = camera_position(target, view_spec.yaw_deg, view_spec.pitch_deg, view_spec.distance)
        viewer.set_camera(wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])), float(view_spec.pitch_deg), float(view_spec.yaw_deg))

        bunny_mesh_points = wp.array((meta.bunny_mesh_vertices_local * float(meta.bunny_mesh_scale)).astype(np.float32), dtype=wp.vec3, device=device)
        bunny_mesh_indices = wp.array(meta.bunny_mesh_tri_indices.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
        bunny_scales = wp.array(np.ones((len(meta.rigid_labels), 3), dtype=np.float32), dtype=wp.vec3, device=device)
        bunny_colors = wp.array(np.asarray([[0.93, 0.78, 0.22], [0.96, 0.68, 0.28]], dtype=np.float32), dtype=wp.vec3, device=device)
        bunny_materials = wp.array(np.tile(np.asarray([[0.32, 0.0, 0.0, 0.0]], dtype=np.float32), (len(meta.rigid_labels), 1)), dtype=wp.vec4, device=device)

        water_points_wp = wp.empty(sim_data["water_points_all"].shape[1], dtype=wp.vec3, device=device)
        water_radii_wp = wp.array(sim_data["water_point_radii"], dtype=wp.float32, device=device)
        water_colors_wp = wp.array(sim_data["water_point_colors"], dtype=wp.vec3, device=device)

        pool_starts_np, pool_ends_np, pool_colors_np = _pool_outline(meta)
        pool_starts_wp = wp.array(pool_starts_np, dtype=wp.vec3, device=device)
        pool_ends_wp = wp.array(pool_ends_np, dtype=wp.vec3, device=device)
        pool_colors_wp = wp.array(pool_colors_np, dtype=wp.vec3, device=device)
        ground_starts_np, ground_ends_np, ground_colors_np = ground_grid(size=float(args.ground_extent), z=float(meta.ground_z), steps=8)
        ground_starts_wp = wp.array(ground_starts_np, dtype=wp.vec3, device=device)
        ground_ends_wp = wp.array(ground_ends_np, dtype=wp.vec3, device=device)
        ground_colors_wp = wp.array(ground_colors_np, dtype=wp.vec3, device=device)

        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["particle_q_all"].shape[0])
        for frame_idx in range(n_frames):
            q_all = sim_data["particle_q_all"][frame_idx].astype(np.float32, copy=False)
            bq = sim_data["body_q"][frame_idx].astype(np.float32, copy=False)
            state.particle_q.assign(q_all)
            state.body_q.assign(bq)

            sim_t = float(frame_idx) * float(sim_data["sim_dt"]) * float(sim_data["substeps"])
            viewer.begin_frame(sim_t)
            water_points_wp.assign(sim_data["water_points_all"][frame_idx].astype(np.float32, copy=False))
            viewer.log_points("/demo/water", water_points_wp, water_radii_wp, water_colors_wp, hidden=False)

            viewer.log_mesh("/demo/bunny_mesh", bunny_mesh_points, bunny_mesh_indices, hidden=True, backface_culling=True)
            viewer.log_instances("/demo/bunnies", "/demo/bunny_mesh", state.body_q, bunny_scales, bunny_colors, bunny_materials, hidden=False)

            sloth_np = q_all[meta.soft_spec.particle_start : meta.soft_spec.particle_start + meta.soft_spec.particle_count]
            sloth_points_wp = wp.array(sloth_np.astype(np.float32, copy=False), dtype=wp.vec3, device=device)
            sloth_radii_wp = wp.array(meta.soft_spec.render_point_radius.astype(np.float32), dtype=wp.float32, device=device)
            sloth_colors_wp = wp.array(
                np.tile(np.asarray(meta.soft_spec.color_rgb, dtype=np.float32), (meta.soft_spec.particle_count, 1)),
                dtype=wp.vec3,
                device=device,
            )
            viewer.log_points("/demo/sloth", sloth_points_wp, sloth_radii_wp, sloth_colors_wp, hidden=False)

            if bool(args.render_soft_springs) and meta.soft_spec.render_edge_global.size > 0:
                starts_np = q_all[meta.soft_spec.render_edge_global[:, 0]]
                ends_np = q_all[meta.soft_spec.render_edge_global[:, 1]]
                viewer.log_lines(
                    "/demo/sloth_springs",
                    wp.array(starts_np.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(ends_np.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(np.tile(np.asarray(meta.soft_spec.color_rgb, dtype=np.float32), (starts_np.shape[0], 1)), dtype=wp.vec3, device=device),
                    width=1.0,
                    hidden=False,
                )
            else:
                viewer.log_lines("/demo/sloth_springs", None, None, None)

            viewer.log_lines("/demo/pool", pool_starts_wp, pool_ends_wp, pool_colors_wp, width=2.0, hidden=False)
            viewer.log_lines("/demo/ground", ground_starts_wp, ground_ends_wp, ground_colors_wp, width=1.0, hidden=False)
            viewer.end_frame()

            frame = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        view_spec.label,
                        meta.scene_label,
                        f"frame {frame_idx + 1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
                    ],
                    font_size=int(args.label_font_size),
                    x=20,
                    y=18,
                    line_gap=8,
                    bg_alpha=120,
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
        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin is not None:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait(timeout=5)


def save_scene_npz(out_dir: Path, prefix: str, sim_data: dict[str, Any]) -> Path:
    out = out_dir / f"{_output_stem(prefix)}_scene.npz"
    np.savez_compressed(
        out,
        particle_q_all=sim_data["particle_q_all"],
        body_q=sim_data["body_q"],
        water_points_all=sim_data["water_points_all"],
        water_point_radii=sim_data["water_point_radii"],
        water_point_colors=sim_data["water_point_colors"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        wall_time_sec=np.float32(sim_data["wall_time_sec"]),
    )
    return out


def load_scene_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "particle_q_all": np.asarray(data["particle_q_all"], dtype=np.float32),
            "body_q": np.asarray(data["body_q"], dtype=np.float32),
            "water_points_all": np.asarray(data["water_points_all"], dtype=np.float32),
            "water_point_radii": np.asarray(data["water_point_radii"], dtype=np.float32),
            "water_point_colors": np.asarray(data["water_point_colors"], dtype=np.float32),
            "sim_dt": float(np.asarray(data["sim_dt"]).ravel()[0]),
            "substeps": int(np.asarray(data["substeps"]).ravel()[0]),
            "wall_time_sec": float(np.asarray(data["wall_time_sec"]).ravel()[0]) if "wall_time_sec" in data else 0.0,
        }


def build_summary(args: argparse.Namespace, sim_data: dict[str, Any], meta: DemoMeta, panel_paths: list[Path]) -> dict[str, Any]:
    return {
        "experiment": "sloth_bunny_water_killing_demo",
        "sloth_ir": str(args.sloth_ir.resolve()),
        "frames": int(sim_data["particle_q_all"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_data["sim_dt"]) * float(sim_data["substeps"]),
        "wall_time_sec": float(sim_data.get("wall_time_sec", 0.0)),
        "gravity_mag": float(args.gravity_mag),
        "ground_extent": float(args.ground_extent),
        "pool_half_extent": float(args.pool_half_extent),
        "pool_height": float(args.pool_height),
        "pool_wall_thickness": float(args.pool_wall_thickness),
        "water_fill_height": float(args.water_fill_height),
        "sloth_bottom_z": float(args.sloth_bottom_z),
        "particle_mass": float(args.particle_mass),
        "rigid_count": int(args.bunny_count),
        "rigid_mass": float(args.rigid_mass),
        "rigid_scale": float(args.rigid_scale),
        "water_voxel_size": float(args.water_voxel_size),
        "water_particles_per_cell": int(args.water_particles_per_cell),
        "water_points_per_particle": int(args.water_points_per_particle),
        "water_density": float(args.water_density),
        "water_young_modulus": float(args.water_young_modulus),
        "water_poisson_ratio": float(args.water_poisson_ratio),
        "water_damping": float(args.water_damping),
        "water_hardening": float(args.water_hardening),
        "water_friction": float(args.water_friction),
        "water_yield_pressure": float(args.water_yield_pressure),
        "water_tensile_yield_ratio": float(args.water_tensile_yield_ratio),
        "water_yield_stress": float(args.water_yield_stress),
        "water_air_drag": float(args.water_air_drag),
        "water_grid_type": str(args.water_grid_type),
        "panel_videos": [str(path) for path in panel_paths],
        "scene_label": meta.scene_label,
        "soft_object": {
            "name": meta.soft_spec.name,
            "case_name": meta.soft_spec.case_name,
            "ir_path": str(meta.soft_spec.ir_path),
            "particle_count": int(meta.soft_spec.particle_count),
            "mass_sum": float(meta.soft_spec.mass_sum),
            "drag_damping": float(meta.soft_spec.drag_damping),
            "collision_radius_mean": float(meta.soft_spec.collision_radius_mean),
            "collider_mu": float(meta.soft_spec.collider_mu),
            "strict_export": bool(meta.soft_spec.strict_export),
            "spring_ke_mode": meta.soft_spec.spring_ke_mode,
            "spring_ke_rel_error_max": float(meta.soft_spec.spring_ke_rel_error_max),
        },
    }


class Example:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.args.out_dir = self.args.out_dir.resolve()
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        self.view_names = _parse_view_names(self.args.views)

        wp.init()
        self.device = newton_import_ir.resolve_device(self.args.device)
        self.ir = _load_ir(self.args.sloth_ir)
        self.model, self.meta = build_model(self.ir, self.args, self.device)
        self.sim_data: dict[str, Any] | None = None
        self.scene_npz: Path | None = None
        self.panel_paths: list[Path] = []

    def step(self):
        if self.sim_data is not None:
            return
        if self.args.render_only_scene is not None:
            self.scene_npz = self.args.render_only_scene.resolve()
            self.sim_data = load_scene_npz(self.scene_npz)
            print(f"Using existing scene NPZ: {self.scene_npz}", flush=True)
            return

        print("Building native MPM water system...", flush=True)
        water = build_water_system(self.model, self.meta, self.args, self.device)
        print("Simulating dedicated killing scene...", flush=True)
        self.sim_data = simulate(self.model, self.meta, self.args, water, self.device)
        self.scene_npz = save_scene_npz(self.args.out_dir, self.args.prefix, self.sim_data)
        print(f"  Scene NPZ: {self.scene_npz}", flush=True)

    def render(self):
        assert self.sim_data is not None
        if self.args.skip_render:
            return
        selected_views = [view for view in _view_specs() if view.name in self.view_names]
        print(f"Rendering {len(selected_views)} view(s)...", flush=True)
        for view in selected_views:
            out_mp4 = self.args.out_dir / f"{_output_stem(self.args.prefix)}_{view.name}.mp4"
            render_view_mp4(
                model=self.model,
                sim_data=self.sim_data,
                meta=self.meta,
                args=self.args,
                device=self.device,
                view_spec=view,
                out_mp4=out_mp4,
            )
            self.panel_paths.append(out_mp4)
            print(f"  Saved: {out_mp4}", flush=True)

    def finalize(self) -> int:
        assert self.sim_data is not None
        summary = build_summary(self.args, self.sim_data, self.meta, self.panel_paths)
        summary_path = self.args.out_dir / f"{_output_stem(self.args.prefix)}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"  Summary: {summary_path}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    def run(self) -> int:
        self.step()
        self.render()
        return self.finalize()


def main() -> int:
    return Example(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
