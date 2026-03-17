#!/usr/bin/env python3
"""Two-way MPM sand <-> spring-mass sloth via shadow dynamic patch bodies.

This demo follows the official Newton two-way coupling pattern:

1. spring-mass sloth is simulated with SolverSemiImplicit
2. a separate shadow rigid-body model approximates the sloth surface as K patches
3. MPM sand reads those patch bodies through ``setup_collider(model=shadow_model)``
4. MPM impulses are collected with ``collect_collider_impulses()``
5. patch body wrench is mapped back to sloth particles as external forces

The Newton core is not modified. All coupling glue lives in this script.
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
    camera_position,
    compute_body_forces,
    ground_grid,
    load_core_module,
    load_ir_checked,
    load_surface_points,
    overlay_text_lines_rgb,
)

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
NEWTON_PY_ROOT = CORE_DIR.parents[2] / "newton"
if str(NEWTON_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWTON_PY_ROOT))

path_defaults = load_core_module("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = load_core_module("newton_import_ir", CORE_DIR / "newton_import_ir.py")

import newton  # noqa: E402


@dataclass
class SoftSpec:
    name: str
    case_name: str
    case_dir: Path
    ir_path: Path
    particle_count: int
    render_particle_ids_global: np.ndarray
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    placement_shift: np.ndarray
    surface_reverse_z: bool
    drag_damping: float
    color_rgb: tuple[float, float, float]
    collider_mu: float
    collision_radius_mean: float


@dataclass
class PatchSegment:
    body_id: int
    particle_ids_global: np.ndarray
    surface_particle_ids_global: np.ndarray
    particle_masses: np.ndarray
    total_mass: float
    rest_points_world: np.ndarray
    rest_points_local: np.ndarray
    rest_surface_points_world: np.ndarray
    rest_com_world: np.ndarray
    prev_com_world: np.ndarray
    prev_quat_xyzw: np.ndarray
    inv_inertia_local: np.ndarray


@dataclass
class ShadowSystem:
    model: newton.Model
    state: newton.State
    body_forces: wp.array
    patches: list[PatchSegment]
    patch_thickness: float
    patch_friction: float
    ground_thickness: float
    ground_friction: float


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
    collider_body_id: wp.array


def _default_sloth_ir() -> Path:
    return Path(
        "Newton/phystwin_bridge/ir/double_lift_sloth_normal/"
        "phystwin_ir_v2_bf_strict.npz"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-way MPM sand / spring-mass sloth coupling via dynamic patch bodies."
    )

    io_group = p.add_argument_group("io")
    io_group.add_argument(
        "--sloth-case-dir",
        type=Path,
        default=Path("Newton/phystwin_bridge/inputs/cases/double_lift_sloth_normal"),
    )
    io_group.add_argument("--sloth-ir", type=Path, default=_default_sloth_ir())
    io_group.add_argument("--out-dir", type=Path, required=True)
    io_group.add_argument("--prefix", default="exp1_sand_two_way_mpm")
    io_group.add_argument("--device", default=path_defaults.default_device())

    sim_group = p.add_argument_group("simulation")
    sim_group.add_argument("--frames", type=int, default=180)
    sim_group.add_argument("--sim-dt", type=float, default=None)
    sim_group.add_argument("--substeps", type=int, default=None)
    sim_group.add_argument("--gravity-mag", type=float, default=9.8)
    sim_group.add_argument("--sand-frame-substeps", type=int, default=1)

    scene_group = p.add_argument_group("scene")
    scene_group.add_argument("--sloth-target-xy", type=float, nargs=2, default=(0.0, 0.0))
    scene_group.add_argument("--drop-height", type=float, default=1.0, help="Sloth bottom height above sand top.")
    scene_group.add_argument("--sand-center-xy", type=float, nargs=2, default=(0.0, 0.0))
    scene_group.add_argument("--sand-base-radius-x", type=float, default=0.60)
    scene_group.add_argument("--sand-base-radius-y", type=float, default=0.80)
    scene_group.add_argument("--sand-height", type=float, default=0.36)

    soft_group = p.add_argument_group("softbody")
    soft_group.add_argument("--particle-mass", type=float, default=0.001)
    soft_group.add_argument("--drag-damping-scale", type=float, default=1.0)
    soft_group.add_argument("--spring-ke-scale", type=float, default=1.0)
    soft_group.add_argument("--spring-kd-scale", type=float, default=1.0)
    soft_group.add_argument(
        "--particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    soft_group.add_argument("--spring-edge-stride", type=int, default=16)
    soft_group.add_argument("--point-radius-vis-scale", type=float, default=2.0)
    soft_group.add_argument("--point-radius-vis-min", type=float, default=0.008)
    soft_group.add_argument(
        "--render-soft-springs", action=argparse.BooleanOptionalAction, default=False
    )

    patch_group = p.add_argument_group("patches")
    patch_group.add_argument("--patch-count", type=int, default=48)
    patch_group.add_argument("--min-patch-particles", type=int, default=4)
    patch_group.add_argument("--patch-force-scale", type=float, default=1.0)
    patch_group.add_argument("--max-particle-dv-per-frame", type=float, default=1.0)
    patch_group.add_argument("--patch-inertia-floor", type=float, default=1.0e-5)
    patch_group.add_argument("--patch-friction", type=float, default=None)
    patch_group.add_argument("--patch-thickness", type=float, default=None)
    patch_group.add_argument("--ground-friction", type=float, default=0.5)

    sand_group = p.add_argument_group("sand")
    sand_group.add_argument("--voxel-size", type=float, default=0.03)
    sand_group.add_argument("--particles-per-cell", type=int, default=3)
    sand_group.add_argument("--grid-type", choices=["sparse", "dense", "fixed"], default="sparse")
    sand_group.add_argument("--tolerance", type=float, default=1.0e-5)
    sand_group.add_argument("--max-iterations", type=int, default=50)
    sand_group.add_argument("--air-drag", type=float, default=1.0)
    sand_group.add_argument("--sand-density", type=float, default=2500.0)

    render_group = p.add_argument_group("render")
    render_group.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    render_group.add_argument("--render-fps", type=float, default=30.0)
    render_group.add_argument("--slowdown", type=float, default=1.0)
    render_group.add_argument("--screen-width", type=int, default=1280)
    render_group.add_argument("--screen-height", type=int, default=720)
    render_group.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    render_group.add_argument("--camera-fov", type=float, default=44.0)
    render_group.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    render_group.add_argument("--label-font-size", type=int, default=28)
    render_group.add_argument("--camera-yaw", type=float, default=-28.0)
    render_group.add_argument("--camera-pitch", type=float, default=-12.0)
    render_group.add_argument("--camera-distance", type=float, default=5.2)
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
        ra, rb = find(int(a)), find(int(b))
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
                base = bbox_lo + np.array([ix * cell[0], iy * cell[1], iz * cell[2]], dtype=np.float32)
                jitter = (rng.random(3, dtype=np.float32) - 0.5) * 0.7 * cell.astype(np.float32)
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
    builder.add_particles(pos=pos, vel=vel, mass=masses, radius=radii, flags=flags_list)
    end_id = len(builder.particle_q)
    return np.arange(begin_id, end_id, dtype=np.int32)


def _prepare_object_only_ir(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, bool]:
    ir_demo: dict[str, Any] = {}
    for key, value in ir.items():
        ir_demo[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    reverse_z = bool(newton_import_ir.ir_bool(ir_demo, "reverse_z", default=False))
    n_obj = int(np.asarray(ir_demo["num_object_points"]).ravel()[0])
    x0 = np.asarray(ir_demo["x0"], dtype=np.float32).copy()[:n_obj]
    v0 = np.zeros_like(np.asarray(ir_demo["v0"], dtype=np.float32), dtype=np.float32)[:n_obj]
    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0

    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    target_bottom_z = float(args.sand_height) + float(args.drop_height)
    shift = np.array(
        [
            float(args.sloth_target_xy[0]) - float(center_xy[0]),
            float(args.sloth_target_xy[1]) - float(center_xy[1]),
            target_bottom_z - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    x0 += shift

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = np.full(n_obj, float(args.particle_mass), dtype=np.float32)
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
    return ir_demo, shift, reverse_z


def _build_soft_model(
    ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> tuple[newton.Model, SoftSpec, DemoMeta]:
    cfg = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
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
    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    obj_edges = edges[(edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)]
    largest_ids = _largest_component_indices(obj_edges, n_obj)
    render_ids = largest_ids
    render_edge_mask = np.isin(obj_edges[:, 0], render_ids) & np.isin(obj_edges[:, 1], render_ids)
    render_edges = obj_edges[render_edge_mask][:: max(1, int(args.spring_edge_stride))].astype(np.int32, copy=True)
    point_radius = np.maximum(
        model_result.radius[render_ids].astype(np.float32) * float(args.point_radius_vis_scale),
        float(args.point_radius_vis_min),
    )

    case_dir = args.sloth_case_dir.resolve()
    spec = SoftSpec(
        name="sloth",
        case_name=case_dir.name,
        case_dir=case_dir,
        ir_path=args.sloth_ir.resolve(),
        particle_count=n_obj,
        render_particle_ids_global=render_ids.astype(np.int32, copy=False),
        render_edge_global=render_edges,
        render_point_radius=point_radius.astype(np.float32),
        placement_shift=np.zeros(3, dtype=np.float32),
        surface_reverse_z=False,
        drag_damping=float(newton_import_ir.ir_scalar(ir, "drag_damping", default=0.0)) * float(args.drag_damping_scale),
        color_rgb=(0.97, 0.68, 0.24),
        collider_mu=float(newton_import_ir.ir_scalar(ir, "contact_collide_object_fric", default=0.4)),
        collision_radius_mean=float(np.mean(model_result.radius[:n_obj])),
    )
    sand_bounds_lo = np.array(
        [
            float(args.sand_center_xy[0]) - float(args.sand_base_radius_x),
            float(args.sand_center_xy[1]) - float(args.sand_base_radius_y),
            0.0,
        ],
        dtype=np.float32,
    )
    sand_bounds_hi = np.array(
        [
            float(args.sand_center_xy[0]) + float(args.sand_base_radius_x),
            float(args.sand_center_xy[1]) + float(args.sand_base_radius_y),
            float(args.sand_height),
        ],
        dtype=np.float32,
    )
    meta = DemoMeta(
        soft_spec=spec,
        sand_bounds_lo=sand_bounds_lo,
        sand_bounds_hi=sand_bounds_hi,
        ground_z=0.0,
        sand_surface_z=float(args.sand_height),
        scene_label="Scene: sloth freefall + native MPM sand + patch-body two-way coupling",
    )
    return model, spec, meta


def _resolve_surface_particle_ids(spec: SoftSpec, object_q0: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    surface_points = load_surface_points(
        spec.case_name,
        reverse_z=spec.surface_reverse_z,
        placement_shift=spec.placement_shift,
    )
    tree = cKDTree(np.asarray(object_q0, dtype=np.float64))
    _dist, idx = tree.query(np.asarray(surface_points, dtype=np.float64), k=1)
    surface_ids = np.unique(idx.astype(np.int32))
    surface_ids = np.intersect1d(surface_ids, spec.render_particle_ids_global, assume_unique=False)
    if surface_ids.size < 4:
        return spec.render_particle_ids_global.copy()
    return surface_ids.astype(np.int32, copy=False)


def _farthest_point_seeds(points: np.ndarray, k: int) -> np.ndarray:
    if k >= points.shape[0]:
        return points.copy()
    seeds = np.empty((k, 3), dtype=np.float32)
    seeds[0] = points[np.argmin(points[:, 0])]
    min_dist = np.sum((points - seeds[0]) ** 2, axis=1)
    for i in range(1, k):
        idx = int(np.argmax(min_dist))
        seeds[i] = points[idx]
        dist = np.sum((points - seeds[i]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return seeds


def _kmeans(points: np.ndarray, k: int, max_iter: int = 32) -> tuple[np.ndarray, np.ndarray]:
    centers = _farthest_point_seeds(points, k)
    labels = np.zeros(points.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        dist2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if members.shape[0] > 0:
                centers[j] = np.mean(members, axis=0)
    return labels, centers


def _cluster_patch_particles(
    surface_ids: np.ndarray,
    all_ids: np.ndarray,
    rest_points_world: np.ndarray,
    patch_count: int,
    min_patch_particles: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    surface_pts = rest_points_world[surface_ids]
    k = max(1, min(int(patch_count), max(1, surface_ids.shape[0] // max(min_patch_particles, 1))))
    labels_surface, centers = _kmeans(surface_pts, k)

    counts = np.bincount(labels_surface, minlength=k)
    large_clusters = np.where(counts >= int(min_patch_particles))[0]
    if large_clusters.size == 0:
        large_clusters = np.array([int(np.argmax(counts))], dtype=np.int32)

    remapped = np.empty(surface_ids.shape[0], dtype=np.int32)
    for idx, label in enumerate(labels_surface):
        if label in large_clusters:
            remapped[idx] = int(np.where(large_clusters == label)[0][0])
        else:
            pt = surface_pts[idx]
            target = np.argmin(np.sum((centers[large_clusters] - pt[None, :]) ** 2, axis=1))
            remapped[idx] = int(target)

    active_centers = np.zeros((large_clusters.shape[0], 3), dtype=np.float32)
    for j in range(active_centers.shape[0]):
        active_centers[j] = np.mean(surface_pts[remapped == j], axis=0)

    all_pts = rest_points_world[all_ids]
    dist2 = np.sum((all_pts[:, None, :] - active_centers[None, :, :]) ** 2, axis=2)
    labels_all = np.argmin(dist2, axis=1).astype(np.int32)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(active_centers.shape[0]):
        particle_ids = all_ids[labels_all == j]
        surface_patch_ids = surface_ids[remapped == j]
        if particle_ids.size == 0 or surface_patch_ids.size < 4:
            continue
        patches.append((particle_ids.astype(np.int32, copy=False), surface_patch_ids.astype(np.int32, copy=False)))
    return patches


def _box_mesh_from_points(points_world: np.ndarray, origin_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float32)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    half = np.maximum(0.5 * (maxs - mins), 2.0e-3)
    local_center = center - origin_world
    verts = np.asarray(
        [
            local_center + [-half[0], -half[1], -half[2]],
            local_center + [half[0], -half[1], -half[2]],
            local_center + [half[0], half[1], -half[2]],
            local_center + [-half[0], half[1], -half[2]],
            local_center + [-half[0], -half[1], half[2]],
            local_center + [half[0], -half[1], half[2]],
            local_center + [half[0], half[1], half[2]],
            local_center + [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    # Face winding: outward-pointing normals (CCW when viewed from outside).
    # cross(V1-V0, V2-V0) must point away from box center for each face.
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],  # bottom (-Z normal)
            [4, 5, 6], [4, 6, 7],  # top    (+Z normal)
            [0, 5, 4], [0, 1, 5],  # front  (-Y normal)
            [1, 6, 5], [1, 2, 6],  # right  (+X normal)
            [2, 7, 6], [2, 3, 7],  # back   (+Y normal)
            [3, 4, 7], [3, 0, 4],  # left   (-X normal)
        ],
        dtype=np.int32,
    )
    return verts, faces


def _build_patch_mesh(points_world: np.ndarray, origin_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float32)
    return _box_mesh_from_points(pts, origin_world)


def _mat33_from_numpy(mat: np.ndarray) -> wp.mat33:
    m = np.asarray(mat, dtype=np.float32)
    return wp.mat33(
        float(m[0, 0]), float(m[0, 1]), float(m[0, 2]),
        float(m[1, 0]), float(m[1, 1]), float(m[1, 2]),
        float(m[2, 0]), float(m[2, 1]), float(m[2, 2]),
    )


def _compute_inertia(points_local: np.ndarray, masses: np.ndarray, floor_scale: float) -> tuple[np.ndarray, np.ndarray]:
    I = np.zeros((3, 3), dtype=np.float64)
    for r, m in zip(np.asarray(points_local, dtype=np.float64), np.asarray(masses, dtype=np.float64), strict=True):
        rr = float(np.dot(r, r))
        I += float(m) * (rr * np.eye(3) - np.outer(r, r))
    floor = max(float(floor_scale) * float(np.sum(masses)), 1.0e-6)
    I += np.eye(3) * floor
    I = I.astype(np.float32)
    return I, np.linalg.inv(I).astype(np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1.0e-8)
    return q


def _quat_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = max(math.sqrt(x * x + y * y + z * z + w * w), 1.0e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_angular_velocity(prev_quat: np.ndarray, curr_quat: np.ndarray, dt: float) -> np.ndarray:
    q_rel = _quat_mul(curr_quat, _quat_conj(prev_quat))
    if q_rel[3] < 0.0:
        q_rel = -q_rel
    sin_half = float(np.linalg.norm(q_rel[:3]))
    if sin_half < 1.0e-8 or dt <= 0.0:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[:3] / sin_half
    angle = 2.0 * math.atan2(sin_half, float(q_rel[3]))
    return (axis * (angle / dt)).astype(np.float32)


def _kabsch_rotation(rest_local: np.ndarray, current_local: np.ndarray, masses: np.ndarray) -> np.ndarray:
    if rest_local.shape[0] < 3:
        return np.eye(3, dtype=np.float32)
    w = np.asarray(masses, dtype=np.float64)
    w_sum = max(float(np.sum(w)), 1.0e-12)
    w /= w_sum
    H = (w[:, None] * rest_local.astype(np.float64)).T @ current_local.astype(np.float64)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R.astype(np.float32)


def _build_shadow_system(
    model: newton.Model,
    spec: SoftSpec,
    args: argparse.Namespace,
    device: str,
) -> ShadowSystem:
    from scipy.spatial import cKDTree

    q0 = model.state().particle_q.numpy().astype(np.float32)[: spec.particle_count]
    masses_all = model.particle_mass.numpy().astype(np.float32)[: spec.particle_count]

    surface_ids = _resolve_surface_particle_ids(spec, q0)
    patch_defs = _cluster_patch_particles(
        surface_ids=surface_ids,
        all_ids=spec.render_particle_ids_global,
        rest_points_world=q0,
        patch_count=int(args.patch_count),
        min_patch_particles=int(args.min_patch_particles),
    )
    if not patch_defs:
        raise RuntimeError("Failed to build any valid surface patches for two-way coupling.")

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.mu = float(args.ground_friction)
    builder.add_ground_plane(cfg=ground_cfg)

    patches: list[PatchSegment] = []
    patch_mu = float(args.patch_friction) if args.patch_friction is not None else float(spec.collider_mu)
    patch_thickness = (
        float(args.patch_thickness)
        if args.patch_thickness is not None
        else max(2.0 * spec.collision_radius_mean, 1.5 * float(args.voxel_size))
    )
    for particle_ids, surface_patch_ids in patch_defs:
        pts = q0[particle_ids]
        masses = masses_all[particle_ids]
        total_mass = float(np.sum(masses))
        com = np.average(pts, axis=0, weights=masses).astype(np.float32)
        rest_local = (pts - com[None, :]).astype(np.float32)
        inertia, inv_inertia = _compute_inertia(rest_local, masses, float(args.patch_inertia_floor))

        body = builder.add_body(
            xform=wp.transform(wp.vec3(*com.tolist()), wp.quat_identity()),
            mass=total_mass,
            inertia=_mat33_from_numpy(inertia),
            label=f"patch_{len(patches):03d}",
        )
        mesh_vertices_local, mesh_faces = _build_patch_mesh(q0[surface_patch_ids], com)
        patch_mesh = newton.Mesh(mesh_vertices_local.astype(np.float32), mesh_faces.reshape(-1).astype(np.int32), compute_inertia=False)
        patch_cfg = builder.default_shape_cfg.copy()
        patch_cfg.mu = patch_mu
        builder.add_shape_mesh(body=body, mesh=patch_mesh, cfg=patch_cfg)

        patches.append(
            PatchSegment(
                body_id=int(body),
                particle_ids_global=particle_ids.astype(np.int32, copy=False),
                surface_particle_ids_global=surface_patch_ids.astype(np.int32, copy=False),
                particle_masses=masses.astype(np.float32, copy=False),
                total_mass=float(total_mass),
                rest_points_world=pts.astype(np.float32, copy=True),
                rest_points_local=rest_local.astype(np.float32, copy=True),
                rest_surface_points_world=q0[surface_patch_ids].astype(np.float32, copy=True),
                rest_com_world=com.astype(np.float32, copy=True),
                prev_com_world=com.astype(np.float32, copy=True),
                prev_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                inv_inertia_local=inv_inertia.astype(np.float32, copy=True),
            )
        )

    shadow_model = builder.finalize(device=device)
    shadow_state = shadow_model.state()
    body_forces = wp.zeros_like(shadow_state.body_f)
    return ShadowSystem(
        model=shadow_model,
        state=shadow_state,
        body_forces=body_forces,
        patches=patches,
        patch_thickness=float(patch_thickness),
        patch_friction=float(patch_mu),
        ground_thickness=max(2.0 * float(args.voxel_size), 0.02),
        ground_friction=float(args.ground_friction),
    )


def _update_shadow_state_from_soft(
    soft_state: newton.State,
    shadow: ShadowSystem,
    frame_dt: float,
) -> None:
    q = soft_state.particle_q.numpy().astype(np.float32)
    qd = soft_state.particle_qd.numpy().astype(np.float32)
    body_q = shadow.state.body_q.numpy().astype(np.float32)
    body_qd = shadow.state.body_qd.numpy().astype(np.float32)

    for patch in shadow.patches:
        pts = q[patch.particle_ids_global]
        vels = qd[patch.particle_ids_global]
        masses = patch.particle_masses
        com = np.average(pts, axis=0, weights=masses).astype(np.float32)
        centered = (pts - com[None, :]).astype(np.float32)
        R = _kabsch_rotation(patch.rest_points_local, centered, masses)
        quat = _rotmat_to_quat_xyzw(R)
        if float(np.dot(quat, patch.prev_quat_xyzw)) < 0.0:
            quat = -quat
        omega = _quat_angular_velocity(patch.prev_quat_xyzw, quat, frame_dt)
        com_vel = np.average(vels, axis=0, weights=masses).astype(np.float32)

        body_q[patch.body_id, :3] = com
        body_q[patch.body_id, 3:7] = quat
        body_qd[patch.body_id, :3] = com_vel
        body_qd[patch.body_id, 3:6] = omega

        patch.prev_com_world = com
        patch.prev_quat_xyzw = quat

    shadow.state.body_q.assign(body_q)
    shadow.state.body_qd.assign(body_qd)


def _build_sand_system(
    shadow: ShadowSystem,
    args: argparse.Namespace,
    device: str,
) -> SandSystem:
    sand_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(sand_builder)
    _spawn_mpm_particle_cone(
        sand_builder,
        center_xy=(float(args.sand_center_xy[0]), float(args.sand_center_xy[1])),
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
    if getattr(sand_model, "mpm", None) is not None and getattr(sand_model.mpm, "hardening", None) is not None:
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
    options.collider_velocity_mode = "instantaneous"
    sand_solver = newton.solvers.SolverImplicitMPM(sand_model, options)

    # Use the SAME thickness for ground and patches so ground doesn't dominate
    # the closest-collider SDF query (collision_sdf picks min_sdf over all colliders).
    collider_thicknesses = [shadow.patch_thickness] + [shadow.patch_thickness] * shadow.model.body_count
    collider_friction = [shadow.ground_friction] + [shadow.patch_friction] * shadow.model.body_count
    sand_solver.setup_collider(
        model=shadow.model,
        collider_thicknesses=collider_thicknesses,
        collider_friction=collider_friction,
        body_q=shadow.state.body_q,
    )

    sand_state_0 = sand_model.state()
    sand_state_1 = sand_model.state()
    sand_state_0.body_q = wp.empty_like(shadow.state.body_q)
    sand_state_0.body_qd = wp.empty_like(shadow.state.body_qd)
    sand_state_0.body_f = wp.empty_like(shadow.state.body_f)
    sand_state_1.body_q = wp.empty_like(shadow.state.body_q)
    sand_state_1.body_qd = wp.empty_like(shadow.state.body_qd)
    sand_state_1.body_f = wp.empty_like(shadow.state.body_f)
    sand_state_0.body_q.assign(shadow.state.body_q)
    sand_state_0.body_qd.assign(shadow.state.body_qd)
    sand_state_0.body_f.zero_()
    sand_state_1.body_q.assign(shadow.state.body_q)
    sand_state_1.body_qd.assign(shadow.state.body_qd)
    sand_state_1.body_f.zero_()

    return SandSystem(
        model=sand_model,
        solver=sand_solver,
        state_0=sand_state_0,
        state_1=sand_state_1,
        render_radii=wp.clone(sand_model.particle_radius),
        render_colors=wp.full(
            sand_model.particle_count,
            value=wp.vec3(0.77, 0.69, 0.48),
            dtype=wp.vec3,
            device=device,
        ),
        collider_body_id=sand_solver.collider_body_index,
    )


def _body_forces_to_particle_forces(
    model: newton.Model,
    soft_state: newton.State,
    shadow: ShadowSystem,
    args: argparse.Namespace,
    frame_dt: float,
) -> tuple[np.ndarray, float, float, int, int]:
    q = soft_state.particle_q.numpy().astype(np.float32)
    body_q = shadow.state.body_q.numpy().astype(np.float32)
    body_f = shadow.body_forces.numpy().astype(np.float32)
    external = np.zeros((model.particle_count, 3), dtype=np.float32)

    for patch in shadow.patches:
        f_lin = body_f[patch.body_id, :3]
        tau = body_f[patch.body_id, 3:6]
        if not np.any(np.isfinite(f_lin)) or not np.any(np.isfinite(tau)):
            continue
        if float(np.linalg.norm(f_lin)) < 1.0e-12 and float(np.linalg.norm(tau)) < 1.0e-12:
            continue
        quat = body_q[patch.body_id, 3:7]
        R = _quat_to_rotmat(quat)
        inv_inertia_world = R @ patch.inv_inertia_local @ R.T
        alpha = inv_inertia_world @ tau
        body_com = body_q[patch.body_id, :3]
        r = q[patch.particle_ids_global] - body_com[None, :]
        accel = (f_lin[None, :] / max(patch.total_mass, 1.0e-8)) + np.cross(alpha[None, :], r)
        forces = patch.particle_masses[:, None] * accel * float(args.patch_force_scale)
        external[patch.particle_ids_global] += forces.astype(np.float32, copy=False)

    pre_clamp_z = float(external[:, 2].sum())
    pre_clamp_norm = float(np.linalg.norm(external, axis=1).sum())

    masses_all = model.particle_mass.numpy().astype(np.float32)
    max_force = masses_all * (float(args.max_particle_dv_per_frame) / max(frame_dt, 1.0e-8))
    norms = np.linalg.norm(external, axis=1)
    n_with_force = int(np.count_nonzero(norms > 1.0e-12))
    scale = np.ones_like(norms, dtype=np.float32)
    over = norms > np.maximum(max_force, 1.0e-8)
    n_clamped = int(np.count_nonzero(over))
    scale[over] = max_force[over] / np.maximum(norms[over], 1.0e-8)
    external *= scale[:, None]
    return external, pre_clamp_z, pre_clamp_norm, n_clamped, n_with_force


def _sync_shadow_to_sand(shadow: ShadowSystem, sand: SandSystem) -> None:
    sand.state_0.body_q.assign(shadow.state.body_q)
    sand.state_0.body_qd.assign(shadow.state.body_qd)
    sand.state_0.body_f.zero_()
    sand.state_1.body_q.assign(shadow.state.body_q)
    sand.state_1.body_qd.assign(shadow.state.body_qd)
    sand.state_1.body_f.zero_()


@dataclass
class CouplingDiag:
    sample_count: int
    impulse_norm_sum: float
    impulse_z_sum: float
    impulse_z_abs_sum: float
    patch_impulse_z_sum: float
    patch_impulse_z_abs_sum: float
    ground_impulse_z_sum: float
    ground_impulse_z_abs_sum: float
    body_f_z_sum: float
    body_f_z_abs_sum: float
    body_f_z_per_patch: np.ndarray
    ext_f_z_sum: float
    ext_f_norm_sum: float
    support_impulse_ratio: float
    support_force_ratio: float
    n_clamped: int
    n_with_force: int
    clamp_ratio: float


def _collect_patch_body_forces(
    sand: SandSystem,
    shadow: ShadowSystem,
    frame_dt: float,
    device: str,
) -> tuple[int, float, float, float, float, float, float, float, np.ndarray]:
    impulses, pos, ids = sand.solver.collect_collider_impulses(sand.state_0)
    shadow.body_forces.zero_()
    n = int(ids.shape[0])
    if n <= 0:
        zero_pf = np.zeros(len(shadow.patches), dtype=np.float32)
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero_pf
    wp.launch(
        compute_body_forces,
        dim=n,
        inputs=[
            frame_dt,
            ids,
            impulses,
            pos,
            sand.collider_body_id,
            shadow.state.body_q,
            shadow.model.body_com,
            shadow.body_forces,
        ],
        device=device,
    )
    impulses_np = impulses.numpy().astype(np.float32, copy=False)
    ids_np = ids.numpy().astype(np.int32, copy=False)
    collider_body_np = sand.collider_body_id.numpy().astype(np.int32, copy=False)
    valid = (ids_np >= 0) & (ids_np < collider_body_np.shape[0])
    body_ids = np.full(ids_np.shape, fill_value=-2, dtype=np.int32)
    body_ids[valid] = collider_body_np[ids_np[valid]]
    patch_mask = body_ids >= 0
    ground_mask = body_ids == -1
    body_f_np = shadow.body_forces.numpy().astype(np.float32, copy=False)
    body_f_z_per_patch = np.array(
        [float(body_f_np[p.body_id, 2]) for p in shadow.patches], dtype=np.float32
    )
    return (
        n,
        float(np.linalg.norm(impulses_np, axis=1).sum()),
        float(impulses_np[:, 2].sum()),
        float(np.abs(impulses_np[:, 2]).sum()),
        float(impulses_np[patch_mask, 2].sum()) if np.any(patch_mask) else 0.0,
        float(np.abs(impulses_np[patch_mask, 2]).sum()) if np.any(patch_mask) else 0.0,
        float(impulses_np[ground_mask, 2].sum()) if np.any(ground_mask) else 0.0,
        float(np.abs(impulses_np[ground_mask, 2]).sum()) if np.any(ground_mask) else 0.0,
        body_f_z_per_patch,
    )


def simulate(
    model: newton.Model,
    spec: SoftSpec,
    shadow: ShadowSystem,
    sand: SandSystem,
    ir: dict[str, np.ndarray],
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
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
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
    contacts = model.contacts() if newton_import_ir._use_collision_pipeline(cfg, ir) else None
    control = model.control()
    state_in = model.state()
    state_out = model.state()

    particle_grid = model.particle_grid if model.particle_grid is not None else None
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))

    frame_dt = float(args.sim_dt) * float(args.substeps)
    max_frames = int(args.frames)
    external_forces_wp = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    coupling_sample_hist: list[int] = [0]
    coupling_impulse_hist: list[float] = [0.0]
    soft_hist: list[np.ndarray] = [state_in.particle_q.numpy().astype(np.float32)]
    sand_hist: list[np.ndarray] = [sand.state_0.particle_q.numpy().astype(np.float32)]
    total_soft_mass = float(model.particle_mass.numpy().astype(np.float32)[: spec.particle_count].sum())
    total_gravity_force = float(args.gravity_mag) * total_soft_mass

    _update_shadow_state_from_soft(state_in, shadow, frame_dt)
    _sync_shadow_to_sand(shadow, sand)

    t0 = time.perf_counter()
    for frame in range(1, max_frames):
        for _sub in range(int(args.substeps)):
            state_in.clear_forces()
            wp.launch(
                add_dense_particle_forces,
                dim=model.particle_count,
                inputs=[state_in.particle_f, external_forces_wp, 1.0],
                device=device,
            )

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if contacts is not None:
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

        _update_shadow_state_from_soft(state_in, shadow, frame_dt)
        _sync_shadow_to_sand(shadow, sand)

        sand_dt = frame_dt / max(int(args.sand_frame_substeps), 1)
        for _ in range(max(int(args.sand_frame_substeps), 1)):
            sand.solver.step(sand.state_0, sand.state_1, None, None, sand_dt)
            sand.solver.project_outside(sand.state_1, sand.state_1, sand_dt)
            sand.state_0, sand.state_1 = sand.state_1, sand.state_0

        (
            sample_count,
            impulse_norm_sum,
            impulse_z_sum,
            impulse_z_abs_sum,
            patch_impulse_z_sum,
            patch_impulse_z_abs_sum,
            ground_impulse_z_sum,
            ground_impulse_z_abs_sum,
            body_f_z_per_patch,
        ) = (
            _collect_patch_body_forces(sand, shadow, frame_dt, device)
        )
        external_force_np, ext_pre_z, ext_pre_norm, n_clamped, n_with_force = (
            _body_forces_to_particle_forces(model, state_in, shadow, args, frame_dt)
        )
        external_forces_wp.assign(external_force_np)
        ext_post_z = float(external_force_np[:, 2].sum())

        sloth_gravity_z = -total_gravity_force
        body_f_z_total = float(body_f_z_per_patch.sum())
        support_impulse_ratio = (
            float(patch_impulse_z_sum) / (total_gravity_force * frame_dt)
            if total_gravity_force > 1.0e-12 and frame_dt > 1.0e-12
            else 0.0
        )
        support_force_ratio = (
            body_f_z_total / total_gravity_force
            if total_gravity_force > 1.0e-12
            else 0.0
        )

        diag = CouplingDiag(
            sample_count=int(sample_count),
            impulse_norm_sum=float(impulse_norm_sum),
            impulse_z_sum=float(impulse_z_sum),
            impulse_z_abs_sum=float(impulse_z_abs_sum),
            patch_impulse_z_sum=float(patch_impulse_z_sum),
            patch_impulse_z_abs_sum=float(patch_impulse_z_abs_sum),
            ground_impulse_z_sum=float(ground_impulse_z_sum),
            ground_impulse_z_abs_sum=float(ground_impulse_z_abs_sum),
            body_f_z_sum=body_f_z_total,
            body_f_z_abs_sum=float(np.abs(body_f_z_per_patch).sum()),
            body_f_z_per_patch=body_f_z_per_patch,
            ext_f_z_sum=ext_post_z,
            ext_f_norm_sum=float(np.linalg.norm(external_force_np, axis=1).sum()),
            support_impulse_ratio=support_impulse_ratio,
            support_force_ratio=support_force_ratio,
            n_clamped=n_clamped,
            n_with_force=n_with_force,
            clamp_ratio=float(n_clamped) / max(n_with_force, 1),
        )

        soft_hist.append(state_in.particle_q.numpy().astype(np.float32))
        sand_hist.append(sand.state_0.particle_q.numpy().astype(np.float32))
        coupling_sample_hist.append(int(sample_count))
        coupling_impulse_hist.append(float(impulse_norm_sum))

        if frame == 1 or frame % 10 == 0 or frame + 1 == max_frames:
            sloth_np = soft_hist[-1][: spec.particle_count]
            sand_np = sand_hist[-1]
            active_patches = int(np.count_nonzero(np.abs(body_f_z_per_patch) > 1.0e-6))
            print(
                f"  frame {frame + 1}/{max_frames}"
                f"  sloth_z=[{sloth_np[:,2].min():.3f},{sloth_np[:,2].max():.3f}]"
                f"  sand_z=[{sand_np[:,2].min():.3f},{sand_np[:,2].max():.3f}]",
                flush=True,
            )
            print(
                f"    DIAG: gravity_z={sloth_gravity_z:.1f}"
                f"  impulse_z={diag.impulse_z_sum:.3f}"
                f"  impulse_abs_z={diag.impulse_z_abs_sum:.3f}"
                f"  patch_impulse_z={diag.patch_impulse_z_sum:.3f}"
                f"  patch_impulse_abs_z={diag.patch_impulse_z_abs_sum:.3f}"
                f"  ground_impulse_z={diag.ground_impulse_z_sum:.3f}"
                f"  ground_impulse_abs_z={diag.ground_impulse_z_abs_sum:.3f}"
                f"  body_f_z={diag.body_f_z_sum:.3f}"
                f"  body_f_abs_z={diag.body_f_z_abs_sum:.3f}"
                f"  ext_pre_z={ext_pre_z:.3f}"
                f"  ext_post_z={diag.ext_f_z_sum:.3f}"
                f"  support_impulse_ratio={diag.support_impulse_ratio:.4f}"
                f"  support_force_ratio={diag.support_force_ratio:.4f}"
                f"  patches_active={active_patches}/{len(shadow.patches)}"
                f"  clamped={diag.n_clamped}/{diag.n_with_force}"
                f"  samples={diag.sample_count}",
                flush=True,
            )

    return {
        "soft_particle_q_all": np.stack(soft_hist),
        "sand_points_all": np.stack(sand_hist),
        "sand_point_radii": sand.render_radii.numpy().astype(np.float32),
        "sand_point_colors": sand.render_colors.numpy().astype(np.float32),
        "coupling_sample_count": np.asarray(coupling_sample_hist, dtype=np.int32),
        "coupling_impulse_norm_sum": np.asarray(coupling_impulse_hist, dtype=np.float32),
        "sim_dt": float(args.sim_dt),
        "substeps": int(args.substeps),
        "frame_dt": frame_dt,
        "wall_time_sec": float(time.perf_counter() - t0),
        "patch_count": int(len(shadow.patches)),
    }


def render_video(
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
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps_out:.6f}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
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
            viewer.renderer.sky_upper = (0.17, 0.31, 0.46)
            viewer.renderer.sky_lower = (0.02, 0.05, 0.10)
        except Exception:
            pass

        target = 0.5 * (meta.sand_bounds_lo + meta.sand_bounds_hi)
        target[2] = max(float(meta.sand_surface_z) * 0.85, 0.25)
        cpos = camera_position(target, float(args.camera_yaw), float(args.camera_pitch), float(args.camera_distance))
        viewer.set_camera(wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])), float(args.camera_pitch), float(args.camera_yaw))

        sloth_render_ids = np.asarray(meta.soft_spec.render_particle_ids_global, dtype=np.int32)
        sloth_count = int(sloth_render_ids.shape[0])
        sand_points_wp = wp.empty(sim_data["sand_points_all"].shape[1], dtype=wp.vec3, device=device)
        sand_radii_wp = wp.array(sim_data["sand_point_radii"], dtype=wp.float32, device=device)
        sand_colors_wp = wp.array(sim_data["sand_point_colors"], dtype=wp.vec3, device=device)
        sloth_points_wp = wp.empty(sloth_count, dtype=wp.vec3, device=device)
        sloth_radii_wp = wp.array(meta.soft_spec.render_point_radius.astype(np.float32), dtype=wp.float32, device=device)
        sloth_colors_wp = wp.array(
            np.tile(np.asarray(meta.soft_spec.color_rgb, dtype=np.float32), (sloth_count, 1)),
            dtype=wp.vec3,
            device=device,
        )

        ground_starts_np, ground_ends_np, ground_colors_np = ground_grid(size=4.0, steps=18, z=float(meta.ground_z))
        ground_starts_wp = wp.array(ground_starts_np, dtype=wp.vec3, device=device)
        ground_ends_wp = wp.array(ground_ends_np, dtype=wp.vec3, device=device)
        ground_colors_wp = wp.array(ground_colors_np, dtype=wp.vec3, device=device)

        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["soft_particle_q_all"].shape[0])
        for frame_idx in range(n_frames):
            q_all = sim_data["soft_particle_q_all"][frame_idx].astype(np.float32, copy=False)
            sloth_points_wp.assign(q_all[sloth_render_ids])
            sand_points_wp.assign(sim_data["sand_points_all"][frame_idx].astype(np.float32, copy=False))
            sim_t = float(frame_idx) * float(sim_data["frame_dt"])

            viewer.begin_frame(sim_t)
            viewer.log_points("/demo/sand", sand_points_wp, sand_radii_wp, sand_colors_wp, hidden=False)
            viewer.log_lines("/demo/ground", ground_starts_wp, ground_ends_wp, ground_colors_wp, width=1.0, hidden=False)
            viewer.log_points("/demo/sloth", sloth_points_wp, sloth_radii_wp, sloth_colors_wp, hidden=False)
            if bool(args.render_soft_springs) and meta.soft_spec.render_edge_global.size > 0:
                starts_np = q_all[meta.soft_spec.render_edge_global[:, 0]]
                ends_np = q_all[meta.soft_spec.render_edge_global[:, 1]]
                viewer.log_lines(
                    "/demo/sloth_springs",
                    wp.array(starts_np.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(ends_np.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(
                        np.tile(np.asarray([[0.18, 0.18, 0.20]], dtype=np.float32), (starts_np.shape[0], 1)),
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
                        "Two-Way Patch Bodies",
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


def _write_outputs(args: argparse.Namespace, ir: dict[str, Any], sim_data: dict[str, Any], out_mp4: Path | None) -> None:
    scene_npz = args.out_dir / f"{args.prefix}_scene.npz"
    np.savez_compressed(
        scene_npz,
        soft_particle_q_all=sim_data["soft_particle_q_all"],
        sand_points_all=sim_data["sand_points_all"],
        sand_point_radii=sim_data["sand_point_radii"],
        sand_point_colors=sim_data["sand_point_colors"],
        coupling_sample_count=sim_data["coupling_sample_count"],
        coupling_impulse_norm_sum=sim_data["coupling_impulse_norm_sum"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        frame_dt=np.float32(sim_data["frame_dt"]),
        wall_time_sec=np.float32(sim_data["wall_time_sec"]),
    )
    summary = {
        "experiment": "exp1_sand_two_way_mpm",
        "sloth_ir": str(args.sloth_ir.resolve()),
        "case_name": str(np.asarray(ir.get("case_name", np.array(["unknown"]))).ravel()[0]),
        "frames": int(sim_data["soft_particle_q_all"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_data["frame_dt"]),
        "wall_time_sec": float(sim_data["wall_time_sec"]),
        "gravity_mag": float(args.gravity_mag),
        "drop_height": float(args.drop_height),
        "patch_count_actual": int(sim_data["patch_count"]),
        "patch_force_scale": float(args.patch_force_scale),
        "max_particle_dv_per_frame": float(args.max_particle_dv_per_frame),
        "voxel_size": float(args.voxel_size),
        "particles_per_cell": int(args.particles_per_cell),
        "grid_type": str(args.grid_type),
        "coupling_samples_total": int(np.sum(sim_data["coupling_sample_count"])),
        "coupling_impulse_norm_total": float(np.sum(sim_data["coupling_impulse_norm_sum"])),
        "render_video": str(out_mp4) if out_mp4 is not None else None,
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sloth_case_dir = args.sloth_case_dir.resolve()

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    raw_ir = _load_ir(args.sloth_ir)
    _resolve_timing(args, raw_ir)
    ir_obj, shift, reverse_z = _prepare_object_only_ir(raw_ir, args)

    print(f"Using sloth case dir: {args.sloth_case_dir}", flush=True)
    print(f"Using sloth IR: {args.sloth_ir.resolve()}", flush=True)
    print("Building object-only sloth model...", flush=True)
    model, spec, meta = _build_soft_model(ir_obj, args, device)
    spec.placement_shift = shift.astype(np.float32)
    spec.surface_reverse_z = bool(reverse_z)

    print("Building dynamic patch shadow bodies...", flush=True)
    shadow = _build_shadow_system(model, spec, args, device)
    print(f"  built {len(shadow.patches)} patch bodies", flush=True)

    print("Building sand system...", flush=True)
    sand = _build_sand_system(shadow, args, device)

    print("Running two-way simulation...", flush=True)
    sim_data = simulate(model, spec, shadow, sand, ir_obj, args, device)

    out_mp4: Path | None = None
    if not bool(args.skip_render):
        out_mp4 = args.out_dir / f"{args.prefix}.mp4"
        print("Rendering MP4...", flush=True)
        render_video(model, sim_data, meta, args, device, out_mp4)

    _write_outputs(args, ir_obj, sim_data, out_mp4)
    print(f"Wrote outputs to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
