#!/usr/bin/env python3
"""Controller-driven sloth moving inside a one-way MPM sand pile.

This demo is intentionally narrower than the two-way sand/water demos:

- load a full PhysTwin IR with controller points intact
- run the sloth with SolverSemiImplicit
- write controller positions every substep
- expose the sloth to MPM only through a moving proxy mesh
- do not feed MPM impulses back into the sloth

The one-way coupling idea follows Newton's `example_mpm_anymal.py`, but the
collider here is a deforming mesh proxy because the sloth is a spring-mass
object, not a rigid-body articulation. Controller particles are kept invisible
in rendering and act as the hidden driver of the motion.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from demo_shared import (
    CORE_DIR,
    alpha_shape_surface_mesh,
    camera_position,
    ground_grid,
    load_core_module,
    load_ir_checked,
    load_surface_points,
    model_particle_collider_body_ids,
    overlay_text_lines_rgb,
)

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
NEWTON_PY_ROOT = CORE_DIR.parents[2] / "newton"
if str(NEWTON_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWTON_PY_ROOT))

path_defaults = load_core_module("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = load_core_module(
    "newton_import_ir", CORE_DIR / "newton_import_ir.py"
)

import newton  # noqa: E402


@dataclass
class SoftSpec:
    name: str
    case_name: str
    case_dir: Path
    ir_path: Path
    particle_count: int
    total_particle_count: int
    render_particle_ids_global: np.ndarray
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    placement_shift: np.ndarray
    surface_reverse_z: bool
    drag_damping: float
    color_rgb: tuple[float, float, float]


@dataclass
class SoftProxy:
    interp_particle_ids_global: np.ndarray
    interp_particle_weights: np.ndarray
    mesh_points: wp.array
    mesh_velocities: wp.array
    mesh_indices: wp.array
    mesh: Any
    query_points_world: np.ndarray
    collider_thickness: float
    collider_friction: float


@dataclass
class ControllerDriver:
    indices_np: np.ndarray
    traj_np: np.ndarray
    indices_wp: wp.array
    target_wp: wp.array
    vel_wp: wp.array
    zero_vel_np: np.ndarray


@dataclass
class DemoMeta:
    soft_spec: SoftSpec
    sand_bounds_lo: np.ndarray
    sand_bounds_hi: np.ndarray
    ground_z: float
    sand_surface_z: float
    scene_label: str


@dataclass
class SandSystem:
    model: newton.Model
    solver: Any
    state_0: newton.State
    state_1: newton.State
    render_radii: wp.array
    render_colors: wp.array
    proxy: SoftProxy


def _default_sloth_ir() -> Path:
    return Path(
        "Newton/phystwin_bridge/ir/double_lift_sloth_normal/"
        "phystwin_ir_v2_bf_strict.npz"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Controller-driven sloth moving inside a one-way MPM sand pile."
    )

    io_group = p.add_argument_group("io")
    io_group.add_argument(
        "--sloth-case-dir",
        type=Path,
        default=Path("Newton/phystwin_bridge/inputs/cases/double_lift_sloth_normal"),
    )
    io_group.add_argument("--sloth-ir", type=Path, default=_default_sloth_ir())
    io_group.add_argument("--out-dir", type=Path, required=True)
    io_group.add_argument("--prefix", default="sloth_mpm_sand_one_way")
    io_group.add_argument("--device", default=path_defaults.default_device())

    sim_group = p.add_argument_group("simulation")
    sim_group.add_argument("--frames", type=int, default=120)
    sim_group.add_argument("--sim-dt", type=float, default=None)
    sim_group.add_argument("--substeps", type=int, default=None)
    sim_group.add_argument("--gravity-mag", type=float, default=9.8)
    sim_group.add_argument("--soft-gravity-mag", type=float, default=0.0)
    sim_group.add_argument(
        "--interpolate-controls", action=argparse.BooleanOptionalAction, default=True
    )

    soft_group = p.add_argument_group("softbody")
    soft_group.add_argument(
        "--sloth-target-xy", type=float, nargs=2, default=(0.0, 0.0)
    )
    soft_group.add_argument(
        "--sloth-bottom-z",
        type=float,
        default=0.06,
        help="Initial sloth bottom height. Defaults to a shallow embed inside the sand pile.",
    )
    soft_group.add_argument("--drag-damping-scale", type=float, default=1.0)
    soft_group.add_argument(
        "--particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    soft_group.add_argument("--spring-edge-stride", type=int, default=16)
    soft_group.add_argument("--point-radius-vis-scale", type=float, default=2.0)
    soft_group.add_argument("--point-radius-vis-min", type=float, default=0.005)
    soft_group.add_argument(
        "--render-soft-springs", action=argparse.BooleanOptionalAction, default=False
    )

    sand_group = p.add_argument_group("sand")
    sand_group.add_argument("--voxel-size", type=float, default=0.03)
    sand_group.add_argument("--particles-per-cell", type=int, default=3)
    sand_group.add_argument(
        "--grid-type", choices=["sparse", "dense", "fixed"], default="sparse"
    )
    sand_group.add_argument("--tolerance", type=float, default=1.0e-5)
    sand_group.add_argument("--max-iterations", type=int, default=50)
    sand_group.add_argument("--air-drag", type=float, default=1.0)
    sand_group.add_argument("--sand-frame-substeps", type=int, default=4)
    sand_group.add_argument(
        "--sand-center-xy", type=float, nargs=2, default=(0.0, 0.0)
    )
    sand_group.add_argument(
        "--sand-base-radius-x", type=float, default=0.60
    )
    sand_group.add_argument(
        "--sand-base-radius-y", type=float, default=0.80
    )
    sand_group.add_argument(
        "--sand-height", type=float, default=0.36
    )
    sand_group.add_argument("--sand-density", type=float, default=2500.0)
    sand_group.add_argument("--proxy-thickness", type=float, default=None)
    sand_group.add_argument("--proxy-friction", type=float, default=0.7)

    render_group = p.add_argument_group("render")
    render_group.add_argument(
        "--skip-render", action=argparse.BooleanOptionalAction, default=False
    )
    render_group.add_argument("--render-fps", type=float, default=30.0)
    render_group.add_argument("--slowdown", type=float, default=1.0)
    render_group.add_argument("--screen-width", type=int, default=1280)
    render_group.add_argument("--screen-height", type=int, default=720)
    render_group.add_argument(
        "--viewer-headless", action=argparse.BooleanOptionalAction, default=True
    )
    render_group.add_argument("--camera-fov", type=float, default=44.0)
    render_group.add_argument(
        "--overlay-label", action=argparse.BooleanOptionalAction, default=True
    )
    render_group.add_argument("--label-font-size", type=int, default=28)
    render_group.add_argument("--camera-yaw", type=float, default=-28.0)
    render_group.add_argument("--camera-pitch", type=float, default=-12.0)
    render_group.add_argument("--camera-distance", type=float, default=4.8)

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
            "collision_radius",
            "controller_idx",
            "controller_traj",
            "sim_dt",
            "sim_substeps",
        ],
    )


def _resolve_timing(args: argparse.Namespace, ir: dict[str, np.ndarray]) -> None:
    if args.sim_dt is None:
        args.sim_dt = float(np.asarray(ir["sim_dt"]).ravel()[0])
    if args.substeps is None:
        args.substeps = int(np.asarray(ir["sim_substeps"]).ravel()[0])
    args.substeps = max(1, int(args.substeps))


def _largest_component_indices(edges: np.ndarray, n_obj: int) -> np.ndarray:
    parent = np.arange(n_obj, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for a, b in np.asarray(edges, dtype=np.int32):
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    components: dict[int, list[int]] = {}
    for i in range(n_obj):
        root = find(i)
        components.setdefault(root, []).append(i)
    largest = max(components.values(), key=len)
    return np.asarray(sorted(largest), dtype=np.int32)


def _spawn_mpm_particle_cone(
    builder: newton.ModelBuilder,
    *,
    center_xy: tuple[float, float],
    base_radius_x: float,
    base_radius_y: float,
    height: float,
    ground_z: float,
    voxel_size: float,
    particles_per_cell: int,
    density: float,
    flags: int,
    seed: int = 0,
) -> np.ndarray:
    """Approximate a sand pile by sampling a tapered cone / pyramid-like heap."""
    bbox_lo = np.array(
        [
            float(center_xy[0]) - float(base_radius_x),
            float(center_xy[1]) - float(base_radius_y),
            float(ground_z),
        ],
        dtype=np.float32,
    )
    bbox_hi = np.array(
        [
            float(center_xy[0]) + float(base_radius_x),
            float(center_xy[1]) + float(base_radius_y),
            float(ground_z + height),
        ],
        dtype=np.float32,
    )
    res = np.array(
        np.ceil(
            max(1, int(particles_per_cell))
            * (bbox_hi - bbox_lo)
            / max(float(voxel_size), 1.0e-4)
        ),
        dtype=int,
    )
    cell = (bbox_hi - bbox_lo) / res
    cell_volume = float(np.prod(cell))
    radius = float(np.max(cell) * 0.5)
    mass = float(cell_volume * density)
    rng = np.random.default_rng(seed)

    pos: list[tuple[float, float, float]] = []
    vel: list[tuple[float, float, float]] = []
    masses: list[float] = []
    radii: list[float] = []
    flags_list: list[int] = []

    for ix in range(int(res[0]) + 1):
        for iy in range(int(res[1]) + 1):
            for iz in range(int(res[2]) + 1):
                base = bbox_lo + np.array(
                    [ix * cell[0], iy * cell[1], iz * cell[2]], dtype=np.float32
                )
                jitter = (rng.random(3, dtype=np.float32) - 0.5) * 0.7 * cell.astype(
                    np.float32
                )
                p = base + jitter
                p[2] = max(float(ground_z + 0.25 * cell[2]), float(p[2]))
                rel_h = (p[2] - float(ground_z)) / max(float(height), 1.0e-6)
                if rel_h < 0.0 or rel_h > 1.0:
                    continue
                taper = max(0.0, 1.0 - rel_h)
                rx = float(base_radius_x) * taper
                ry = float(base_radius_y) * taper
                dx = (p[0] - float(center_xy[0])) / max(rx, 1.0e-6)
                dy = (p[1] - float(center_xy[1])) / max(ry, 1.0e-6)
                if dx * dx + dy * dy > 1.0:
                    continue
                pos.append((float(p[0]), float(p[1]), float(p[2])))
                vel.append((0.0, 0.0, 0.0))
                masses.append(mass)
                radii.append(radius)
                flags_list.append(int(flags))

    begin_id = len(builder.particle_q)
    builder.add_particles(
        pos=pos, vel=vel, mass=masses, radius=radii, flags=flags_list
    )
    end_id = len(builder.particle_q)
    return np.arange(begin_id, end_id, dtype=np.int32)


def _prepare_ir_for_demo(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, bool]:
    ir_demo: dict[str, Any] = {}
    for key, value in ir.items():
        ir_demo[key] = (
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )

    reverse_z = bool(newton_import_ir.ir_bool(ir_demo, "reverse_z", default=False))

    x0 = np.asarray(ir_demo["x0"], dtype=np.float32).copy()
    v0 = np.asarray(ir_demo["v0"], dtype=np.float32).copy()
    ctrl_traj = np.asarray(ir_demo["controller_traj"], dtype=np.float32).copy()
    n_obj = int(np.asarray(ir_demo["num_object_points"]).ravel()[0])

    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0
        ctrl_traj[:, :, 2] *= -1.0

    obj_pts = x0[:n_obj]
    bbox_min = obj_pts.min(axis=0)
    bbox_max = obj_pts.max(axis=0)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    target_bottom_z = float(args.sloth_bottom_z)
    shift = np.array(
        [
            float(args.sloth_target_xy[0]) - float(center_xy[0]),
            float(args.sloth_target_xy[1]) - float(center_xy[1]),
            target_bottom_z - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )

    x0 += shift
    ctrl_traj += shift.reshape(1, 1, 3)

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["controller_traj"] = ctrl_traj
    ir_demo["reverse_z"] = np.asarray(False)
    return ir_demo, shift, reverse_z


def _build_controller_driver(
    ir: dict[str, np.ndarray], device: str
) -> ControllerDriver:
    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int32).ravel()
    ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
    if ctrl_idx.size == 0:
        raise ValueError("This demo requires controller_idx/controller_traj in the IR.")
    if (
        ctrl_traj.ndim != 3
        or ctrl_traj.shape[1] != ctrl_idx.size
        or ctrl_traj.shape[2] != 3
    ):
        raise ValueError(
            f"Unexpected controller_traj shape {ctrl_traj.shape} for controller count {ctrl_idx.size}"
        )
    return ControllerDriver(
        indices_np=ctrl_idx,
        traj_np=ctrl_traj,
        indices_wp=wp.array(ctrl_idx, dtype=wp.int32, device=device),
        target_wp=wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device),
        vel_wp=wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device),
        zero_vel_np=np.zeros((ctrl_idx.size, 3), dtype=np.float32),
    )


def _build_soft_model(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> tuple[newton.Model, SoftSpec, DemoMeta]:
    cfg = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        gravity=-float(args.soft_gravity_mag),
        gravity_from_reverse_z=False,
        spring_ke_scale=1.0,
        spring_kd_scale=1.0,
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=True,
        disable_particle_contact_kernel=not bool(args.particle_contact_kernel),
        shape_contacts=False,
        add_ground_plane=False,
        strict_physics_checks=False,
        apply_drag=True,
        drag_damping_scale=float(args.drag_damping_scale),
        up_axis="Z",
        device=device,
    )
    model_result = newton_import_ir.build_model(ir, cfg, device)
    model = model_result.model

    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int32).ravel()
    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    obj_edge_mask = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    obj_edges = edges[obj_edge_mask]
    largest_ids = _largest_component_indices(obj_edges, n_obj)
    render_ids = np.setdiff1d(largest_ids, ctrl_idx, assume_unique=False)
    render_edge_mask = np.isin(obj_edges[:, 0], render_ids) & np.isin(
        obj_edges[:, 1], render_ids
    )
    render_edges = obj_edges[render_edge_mask][
        :: max(1, int(args.spring_edge_stride))
    ].astype(np.int32, copy=True)
    point_radius = np.minimum(
        model_result.radius[render_ids].astype(np.float32)
        * float(args.point_radius_vis_scale),
        float(args.point_radius_vis_min),
    )
    case_dir = args.sloth_case_dir.resolve()
    case_name = case_dir.name
    spec = SoftSpec(
        name="sloth",
        case_name=case_name,
        case_dir=case_dir,
        ir_path=args.sloth_ir.resolve(),
        particle_count=n_obj,
        total_particle_count=model.particle_count,
        render_particle_ids_global=render_ids,
        render_edge_global=render_edges,
        render_point_radius=point_radius,
        placement_shift=np.zeros(3, dtype=np.float32),
        surface_reverse_z=False,
        drag_damping=float(newton_import_ir.ir_scalar(ir, "drag_damping", default=0.0))
        * float(args.drag_damping_scale),
        color_rgb=(0.97, 0.68, 0.24),
    )
    bounds_lo = np.array(
        [
            float(args.sand_center_xy[0]) - float(args.sand_base_radius_x),
            float(args.sand_center_xy[1]) - float(args.sand_base_radius_y),
            0.0,
        ],
        dtype=np.float32,
    )
    bounds_hi = np.array(
        [
            float(args.sand_center_xy[0]) + float(args.sand_base_radius_x),
            float(args.sand_center_xy[1]) + float(args.sand_base_radius_y),
            float(args.sand_height),
        ],
        dtype=np.float32,
    )
    meta = DemoMeta(
        soft_spec=spec,
        sand_bounds_lo=bounds_lo,
        sand_bounds_hi=bounds_hi,
        ground_z=float(bounds_lo[2]),
        sand_surface_z=float(bounds_hi[2]),
        scene_label="Scene: controller-driven sloth inside one-way MPM sand pile",
    )
    return model, spec, meta


def _build_proxy(
    model: newton.Model,
    spec: SoftSpec,
    args: argparse.Namespace,
    device: str,
) -> SoftProxy:
    state = model.state()
    q0 = state.particle_q.numpy().astype(np.float32)
    qd0 = (
        state.particle_qd.numpy().astype(np.float32)
        if state.particle_qd is not None
        else np.zeros_like(q0, dtype=np.float32)
    )
    obj_pts = q0[spec.render_particle_ids_global]

    try:
        surface_points = load_surface_points(
            spec.case_name,
            reverse_z=spec.surface_reverse_z,
            placement_shift=spec.placement_shift,
        )
        proxy_vertices, tri_local = alpha_shape_surface_mesh(surface_points)
    except Exception:
        proxy_vertices, tri_local = alpha_shape_surface_mesh(obj_pts)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(obj_pts, dtype=np.float64))
    k = min(4, obj_pts.shape[0])
    dists, idx = tree.query(np.asarray(proxy_vertices, dtype=np.float64), k=k)
    if k == 1:
        idx = idx[:, None]
        dists = dists[:, None]
    weights = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-5)
    weights /= np.sum(weights, axis=1, keepdims=True)

    global_ids = spec.render_particle_ids_global[idx.astype(np.int32)]
    proxy_points = np.sum(
        obj_pts[idx.astype(np.int32)] * weights[:, :, None], axis=1
    ).astype(np.float32, copy=False)
    proxy_vel = np.sum(qd0[global_ids] * weights[:, :, None], axis=1).astype(
        np.float32, copy=False
    )
    mesh_points = wp.array(proxy_points, dtype=wp.vec3, device=device)
    mesh_velocities = wp.array(proxy_vel, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(
        tri_local.reshape(-1).astype(np.int32), dtype=int, device=device
    )
    mesh = wp.Mesh(mesh_points, mesh_indices, mesh_velocities)

    auto_thickness = max(
        float(np.mean(spec.render_point_radius)), 0.75 * float(args.voxel_size)
    )
    thickness = (
        float(args.proxy_thickness)
        if args.proxy_thickness is not None
        else auto_thickness
    )
    return SoftProxy(
        interp_particle_ids_global=global_ids.astype(np.int32, copy=False),
        interp_particle_weights=weights.astype(np.float32, copy=False),
        mesh_points=mesh_points,
        mesh_velocities=mesh_velocities,
        mesh_indices=mesh_indices,
        mesh=mesh,
        query_points_world=proxy_points.astype(np.float32, copy=True),
        collider_thickness=thickness,
        collider_friction=float(args.proxy_friction),
    )


def _build_sand_system(
    proxy: SoftProxy, args: argparse.Namespace, device: str
) -> SandSystem:
    sand_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(sand_builder)
    sand_builder.add_ground_plane()

    _spawn_mpm_particle_cone(
        sand_builder,
        center_xy=(
            float(args.sand_center_xy[0]),
            float(args.sand_center_xy[1]),
        ),
        base_radius_x=float(args.sand_base_radius_x),
        base_radius_y=float(args.sand_base_radius_y),
        height=float(args.sand_height),
        ground_z=0.0,
        voxel_size=float(args.voxel_size),
        particles_per_cell=int(args.particles_per_cell),
        density=float(args.sand_density),
        flags=int(newton.ParticleFlags.ACTIVE),
        seed=0,
    )

    sand_model = sand_builder.finalize(device=device)
    sand_model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    if (
        getattr(sand_model, "mpm", None) is not None
        and getattr(sand_model.mpm, "hardening", None) is not None
    ):
        sand_model.mpm.hardening.fill_(0.0)

    options = newton.solvers.SolverImplicitMPM.Options()
    options.voxel_size = float(args.voxel_size)
    options.tolerance = float(args.tolerance)
    options.transfer_scheme = "pic"
    options.grid_type = str(args.grid_type)
    options.grid_padding = 50 if args.grid_type == "fixed" else 0
    options.max_active_cell_count = 1 << 15 if args.grid_type == "fixed" else -1
    options.strain_basis = "P0"
    options.max_iterations = int(args.max_iterations)
    options.critical_fraction = 0.0
    options.air_drag = float(args.air_drag)
    options.collider_velocity_mode = "finite_difference"
    sand_solver = newton.solvers.SolverImplicitMPM(sand_model, options)

    body_collider_ids = model_particle_collider_body_ids(sand_model)
    collider_meshes = [None] * len(body_collider_ids) + [proxy.mesh]
    collider_body_ids = body_collider_ids + [None]
    collider_thicknesses = [None] * len(body_collider_ids) + [proxy.collider_thickness]
    collider_friction = [None] * len(body_collider_ids) + [proxy.collider_friction]
    sand_solver.setup_collider(
        collider_meshes=collider_meshes,
        collider_body_ids=collider_body_ids,
        collider_thicknesses=collider_thicknesses,
        collider_friction=collider_friction,
        model=sand_model,
    )

    return SandSystem(
        model=sand_model,
        solver=sand_solver,
        state_0=sand_model.state(),
        state_1=sand_model.state(),
        render_radii=wp.clone(sand_model.particle_radius),
        render_colors=wp.full(
            sand_model.particle_count,
            value=wp.vec3(0.77, 0.69, 0.48),
            dtype=wp.vec3,
            device=device,
        ),
        proxy=proxy,
    )


def _apply_controller_targets(
    state: newton.State,
    controller: ControllerDriver,
    frame: int,
    substep: int,
    substeps: int,
    device: str,
    *,
    interpolate: bool,
) -> None:
    last_frame = int(controller.traj_np.shape[0]) - 1
    if frame >= last_frame:
        target = controller.traj_np[last_frame]
        controller.target_wp.assign(target.astype(np.float32, copy=False))
        controller.vel_wp.assign(controller.zero_vel_np)
    else:
        target = newton_import_ir.interpolate_controller(
            controller.traj_np,
            frame,
            substep,
            substeps,
            interpolate,
        )
        controller.target_wp.assign(target.astype(np.float32, copy=False))
        controller.vel_wp.assign(controller.zero_vel_np)
    wp.launch(
        newton_import_ir._write_kinematic_state,
        dim=controller.indices_np.size,
        inputs=[
            state.particle_q,
            state.particle_qd,
            controller.indices_wp,
            controller.target_wp,
            controller.vel_wp,
        ],
        device=device,
    )


def _update_proxy_from_soft_state(
    state: newton.State, proxy: SoftProxy, frame_dt: float
) -> None:
    q = state.particle_q.numpy().astype(np.float32)
    pts = np.sum(
        q[proxy.interp_particle_ids_global] * proxy.interp_particle_weights[:, :, None],
        axis=1,
    )
    pts = np.asarray(pts, dtype=np.float32)
    valid = np.all(np.isfinite(pts), axis=1)
    if not np.all(valid):
        pts[~valid] = proxy.query_points_world[~valid]
    vel = (pts - proxy.query_points_world) / max(float(frame_dt), 1.0e-8)
    proxy.query_points_world = pts.copy()
    proxy.mesh_points.assign(pts)
    proxy.mesh_velocities.assign(vel.astype(np.float32, copy=False))
    proxy.mesh.refit()


def simulate(
    model: newton.Model,
    ir: dict[str, np.ndarray],
    spec: SoftSpec,
    controller: ControllerDriver,
    sand: SandSystem,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=True,
    )
    cfg = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        gravity=-float(args.soft_gravity_mag),
        gravity_from_reverse_z=False,
        spring_ke_scale=1.0,
        spring_kd_scale=1.0,
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=True,
        disable_particle_contact_kernel=not bool(args.particle_contact_kernel),
        shape_contacts=False,
        add_ground_plane=False,
        strict_physics_checks=False,
        apply_drag=True,
        drag_damping_scale=float(args.drag_damping_scale),
        up_axis="Z",
        device=device,
    )
    collision_pipeline_enabled = bool(newton_import_ir._use_collision_pipeline(cfg, ir))
    contacts = model.contacts() if collision_pipeline_enabled else None
    control = model.control()
    state_in = model.state()
    state_out = model.state()

    particle_grid = model.particle_grid if model.particle_grid is not None else None
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(
            model.particle_cohesion
        )
        search_radius = max(
            search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6))
        )

    frame_dt = float(args.sim_dt) * float(args.substeps)
    max_frames = int(args.frames)
    if max_frames < 2:
        raise ValueError(f"Need at least 2 frames, got {max_frames}")

    soft_hist: list[np.ndarray] = [state_in.particle_q.numpy().astype(np.float32)]
    sand_hist: list[np.ndarray] = [sand.state_0.particle_q.numpy().astype(np.float32)]
    t0 = time.perf_counter()

    for frame in range(1, max_frames):
        for sub in range(int(args.substeps)):
            state_in.clear_forces()
            _apply_controller_targets(
                state_in,
                controller,
                frame,
                sub,
                int(args.substeps),
                device,
                interpolate=bool(args.interpolate_controls),
            )

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if collision_pipeline_enabled:
                assert contacts is not None
                model.collide(state_in, contacts)

            solver.step(state_in, state_out, control, contacts, float(args.sim_dt))
            state_in, state_out = state_out, state_in

            if spec.drag_damping > 0.0:
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=spec.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        spec.particle_count,
                        float(args.sim_dt),
                        float(spec.drag_damping),
                    ],
                    device=device,
                )

        _update_proxy_from_soft_state(state_in, sand.proxy, frame_dt)
        sand_substeps = max(1, int(args.sand_frame_substeps))
        sand_dt = frame_dt / float(sand_substeps)
        for _ in range(sand_substeps):
            sand.solver.step(sand.state_0, sand.state_1, None, None, sand_dt)
            sand.solver.project_outside(sand.state_1, sand.state_1, sand_dt)
            sand.state_0, sand.state_1 = sand.state_1, sand.state_0

        soft_hist.append(state_in.particle_q.numpy().astype(np.float32))
        sand_hist.append(sand.state_0.particle_q.numpy().astype(np.float32))

        if frame == 1 or frame % 10 == 0 or frame + 1 == max_frames:
            sloth_np = soft_hist[-1][: spec.particle_count]
            sand_np = sand_hist[-1]
            print(
                f"  frame {frame + 1}/{max_frames}"
                f"  sloth_z=[{sloth_np[:,2].min():.3f},{sloth_np[:,2].max():.3f}]"
                f"  sand_z=[{sand_np[:,2].min():.3f},{sand_np[:,2].max():.3f}]",
                flush=True,
            )

    return {
        "soft_particle_q_all": np.stack(soft_hist),
        "sand_points_all": np.stack(sand_hist),
        "sand_point_radii": sand.render_radii.numpy().astype(np.float32),
        "sand_point_colors": sand.render_colors.numpy().astype(np.float32),
        "sim_dt": float(args.sim_dt),
        "substeps": int(args.substeps),
        "frame_dt": frame_dt,
        "wall_time_sec": float(time.perf_counter() - t0),
    }


def render_mp4(
    *,
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: DemoMeta,
    args: argparse.Namespace,
    device: str,
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

    viewer = newton.viewer.ViewerGL(
        width=width, height=height, vsync=False, headless=bool(args.viewer_headless)
    )
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
            viewer.renderer.sky_upper = (0.17, 0.31, 0.46)
            viewer.renderer.sky_lower = (0.02, 0.05, 0.10)
        except Exception:
            pass

        target = 0.5 * (meta.sand_bounds_lo + meta.sand_bounds_hi)
        target[2] = max(float(meta.sand_surface_z) * 0.8, 0.25)
        cpos = camera_position(
            target,
            float(args.camera_yaw),
            float(args.camera_pitch),
            float(args.camera_distance),
        )
        viewer.set_camera(
            wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
            float(args.camera_pitch),
            float(args.camera_yaw),
        )

        sloth_render_ids = np.asarray(
            meta.soft_spec.render_particle_ids_global, dtype=np.int32
        )
        sloth_count = int(sloth_render_ids.shape[0])
        sand_points_wp = wp.empty(
            sim_data["sand_points_all"].shape[1], dtype=wp.vec3, device=device
        )
        sand_radii_wp = wp.array(
            sim_data["sand_point_radii"], dtype=wp.float32, device=device
        )
        sand_colors_wp = wp.array(
            sim_data["sand_point_colors"], dtype=wp.vec3, device=device
        )

        sloth_points_wp = wp.empty(sloth_count, dtype=wp.vec3, device=device)
        sloth_radii_wp = wp.array(
            meta.soft_spec.render_point_radius.astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        sloth_colors_wp = wp.array(
            np.tile(
                np.asarray(meta.soft_spec.color_rgb, dtype=np.float32), (sloth_count, 1)
            ),
            dtype=wp.vec3,
            device=device,
        )

        ground_starts_np, ground_ends_np, ground_colors_np = ground_grid(
            size=4.0,
            steps=18,
            z=float(meta.ground_z),
        )
        ground_starts_wp = wp.array(ground_starts_np, dtype=wp.vec3, device=device)
        ground_ends_wp = wp.array(ground_ends_np, dtype=wp.vec3, device=device)
        ground_colors_wp = wp.array(ground_colors_np, dtype=wp.vec3, device=device)

        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["soft_particle_q_all"].shape[0])
        for frame_idx in range(n_frames):
            q_all = sim_data["soft_particle_q_all"][frame_idx].astype(
                np.float32, copy=False
            )
            sloth_points_wp.assign(q_all[sloth_render_ids])
            sand_points_wp.assign(
                sim_data["sand_points_all"][frame_idx].astype(np.float32, copy=False)
            )

            sim_t = float(frame_idx) * float(sim_data["frame_dt"])
            viewer.begin_frame(sim_t)
            viewer.log_points(
                "/demo/sand",
                sand_points_wp,
                sand_radii_wp,
                sand_colors_wp,
                hidden=False,
            )
            viewer.log_lines(
                "/demo/ground",
                ground_starts_wp,
                ground_ends_wp,
                ground_colors_wp,
                width=1.0,
                hidden=False,
            )
            viewer.log_points(
                "/demo/sloth",
                sloth_points_wp,
                sloth_radii_wp,
                sloth_colors_wp,
                hidden=False,
            )

            if (
                bool(args.render_soft_springs)
                and meta.soft_spec.render_edge_global.size > 0
            ):
                starts_np = q_all[meta.soft_spec.render_edge_global[:, 0]]
                ends_np = q_all[meta.soft_spec.render_edge_global[:, 1]]
                viewer.log_lines(
                    "/demo/sloth_springs",
                    wp.array(
                        starts_np.astype(np.float32), dtype=wp.vec3, device=device
                    ),
                    wp.array(ends_np.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(
                        np.tile(
                            np.asarray([[0.18, 0.18, 0.20]], dtype=np.float32),
                            (starts_np.shape[0], 1),
                        ),
                        dtype=wp.vec3,
                        device=device,
                    ),
                    width=1.0,
                    hidden=False,
                )
            else:
                viewer.log_lines("/demo/sloth_springs", None, None, None)

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if bool(args.overlay_label):
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        "Invisible Controller Driver",
                        meta.scene_label,
                        f"frame {frame_idx + 1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
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
        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin is not None:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait(timeout=5)


def _write_outputs(
    args: argparse.Namespace,
    ir: dict[str, np.ndarray],
    sim_data: dict[str, Any],
    out_mp4: Path | None,
) -> None:
    scene_npz = args.out_dir / f"{args.prefix}_scene.npz"
    np.savez_compressed(
        scene_npz,
        soft_particle_q_all=sim_data["soft_particle_q_all"],
        sand_points_all=sim_data["sand_points_all"],
        sand_point_radii=sim_data["sand_point_radii"],
        sand_point_colors=sim_data["sand_point_colors"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        frame_dt=np.float32(sim_data["frame_dt"]),
        wall_time_sec=np.float32(sim_data["wall_time_sec"]),
    )
    summary = {
        "experiment": "sloth_mpm_sand_one_way",
        "sloth_ir": str(args.sloth_ir.resolve()),
        "case_name": str(
            np.asarray(ir.get("case_name", np.array(["unknown"]))).ravel()[0]
        ),
        "frames": int(sim_data["soft_particle_q_all"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_data["frame_dt"]),
        "wall_time_sec": float(sim_data["wall_time_sec"]),
        "soft_gravity_mag": float(args.soft_gravity_mag),
        "sloth_bottom_z_effective": float(args.sloth_bottom_z),
        "voxel_size": float(args.voxel_size),
        "particles_per_cell": int(args.particles_per_cell),
        "grid_type": str(args.grid_type),
        "sand_frame_substeps": int(args.sand_frame_substeps),
        "sand_center_xy": [float(v) for v in np.asarray(args.sand_center_xy, dtype=np.float32)],
        "sand_base_radius_x": float(args.sand_base_radius_x),
        "sand_base_radius_y": float(args.sand_base_radius_y),
        "sand_height": float(args.sand_height),
        "controller_count": int(np.asarray(ir["controller_idx"]).size),
        "render_video": str(out_mp4) if out_mp4 is not None else None,
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sloth_case_dir = args.sloth_case_dir.resolve()
    if not args.sloth_case_dir.exists():
        raise FileNotFoundError(f"Missing sloth case dir: {args.sloth_case_dir}")

    wp.init()
    device = newton_import_ir.resolve_device(args.device)

    raw_ir = _load_ir(args.sloth_ir)
    _resolve_timing(args, raw_ir)
    ir_demo, shift, reverse_z = _prepare_ir_for_demo(raw_ir, args)

    print(f"Using sloth case dir: {args.sloth_case_dir}", flush=True)
    print(f"Using sloth IR: {args.sloth_ir.resolve()}", flush=True)

    model, spec, meta = _build_soft_model(ir_demo, args, device)
    spec.placement_shift = shift.astype(np.float32)
    spec.surface_reverse_z = bool(reverse_z)
    controller = _build_controller_driver(ir_demo, device)
    sand = _build_sand_system(_build_proxy(model, spec, args, device), args, device)

    print("Simulating controller-driven sloth...", flush=True)
    sim_data = simulate(model, ir_demo, spec, controller, sand, args, device)

    out_mp4: Path | None = None
    if not bool(args.skip_render):
        out_mp4 = args.out_dir / f"{args.prefix}_wide.mp4"
        print("Rendering MP4...", flush=True)
        render_mp4(
            model=model,
            sim_data=sim_data,
            meta=meta,
            args=args,
            device=device,
            out_mp4=out_mp4,
        )

    _write_outputs(args, ir_demo, sim_data, out_mp4)
    print(f"Wrote outputs to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
