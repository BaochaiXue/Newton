#!/usr/bin/env python3
"""Standalone soft sloth + native Newton MPM sand-bed demo.

This script is intentionally independent from the killing demo. It follows the
Newton example style:

- parse args
- build scene/model
- build MPM system
- simulate
- render

The imported sloth keeps its PhysTwin particle-spring representation. A
deforming proxy mesh is used only as an MPM collider adapter so we can exercise
native Newton MPM two-way coupling without modifying Newton core. The MPM sand
parameters and visual style are intentionally pushed toward Newton's official
`example_mpm_anymal.py`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_core(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


path_defaults = _load_core("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = _load_core("newton_import_ir", CORE_DIR / "newton_import_ir.py")

import newton  # noqa: E402


@dataclass
class SoftSpec:
    name: str
    case_name: str
    color_rgb: tuple[float, float, float]
    ir_path: Path
    particle_start: int
    particle_count: int
    render_particle_ids_global: np.ndarray
    drag_damping: float
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    mass_sum: float
    placement_shift: np.ndarray
    reverse_z: bool
    collision_radius_mean: float
    collider_mu: float


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
    collider_projection_threshold: float
    collider_mu: float


@dataclass
class DemoMeta:
    spec: SoftSpec
    bed_half_extent_x: float
    bed_half_extent_y: float
    sand_height: float
    ground_z: float
    scene_label: str


@dataclass
class SandSystem:
    model: newton.Model
    solver: Any
    state_0: newton.State
    state_1: newton.State
    collider_impulses: wp.array
    collider_impulse_pos: wp.array
    collider_impulse_ids: wp.array
    collider_body_id: wp.array
    body_sand_forces: wp.array | None
    frame_dt: float
    sim_dt: float
    active_impulse_count: int
    render_radii: wp.array
    render_colors: wp.array
    proxy: SoftProxy


@dataclass
class ViewSpec:
    name: str
    label: str
    yaw_deg: float
    pitch_deg: float
    distance: float


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    cid = collider_ids[tid]
    if cid < 0 or cid >= body_ids.shape[0]:
        return
    body_index = body_ids[cid]
    if body_index == -1:
        return
    f_world = collider_impulses[tid] / dt
    X_wb = body_q[body_index]
    r = collider_impulse_pos[tid] - wp.transform_point(X_wb, body_com[body_index])
    wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def add_dense_particle_forces(
    particle_f: wp.array(dtype=wp.vec3),
    external_f: wp.array(dtype=wp.vec3),
    scale: float,
):
    tid = wp.tid()
    particle_f[tid] = particle_f[tid] + external_f[tid] * scale


@wp.kernel
def apply_drag_range(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_start: int,
    particle_count: int,
    dt: float,
    damping: float,
):
    tid = wp.tid()
    if tid >= particle_count:
        return
    i = particle_start + tid
    v = particle_qd[i]
    scale = wp.exp(-dt * damping)
    particle_q[i] = particle_q[i] - v * dt * (1.0 - scale)
    particle_qd[i] = v * scale


def _overlay_text_lines_rgb(frame: np.ndarray, lines: list[str], font_size: int = 24) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    img = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    font = None
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, size=max(10, int(font_size)))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    x = 18
    y = 16
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.rectangle((x - 8, y - 4, x + bbox[2] + 8, y + bbox[3] + 4), fill=(0, 0, 0, 110))
        draw.text((x, y), line, fill=(255, 255, 255, 255), font=font)
        y += (bbox[3] - bbox[1]) + 6
    return np.asarray(img, dtype=np.uint8)


def _camera_position(target: np.ndarray, yaw_deg: float, pitch_deg: float, distance: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    front = np.array(
        [
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float32,
    )
    front /= max(float(np.linalg.norm(front)), 1.0e-6)
    return target.astype(np.float32) - front * float(distance)


def _substep_feedback_scale(substep: int, substeps: int) -> float:
    if substeps <= 1:
        return 1.0
    # Apply stale MPM feedback strongly at the beginning of the frame and fade it
    # out across soft-body substeps to avoid injecting outdated impulses late.
    return max(0.0, 1.0 - (float(substep) / float(substeps - 1)))


def _alpha_shape_surface_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import Delaunay, cKDTree

    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(pts)
    k = min(8, pts.shape[0])
    dists, _ = tree.query(pts, k=k)
    nn_col = min(4, dists.shape[1] - 1)
    spacing = float(np.median(dists[:, nn_col]))
    alpha = max(2.5 * spacing, 1.0e-5)

    tetra = Delaunay(pts)
    face_dict: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    local_faces = (((0, 2, 1), 3), ((0, 1, 3), 2), ((0, 3, 2), 1), ((1, 2, 3), 0))
    simplices = np.asarray(tetra.simplices, dtype=np.int32)
    for tet in simplices:
        tet_pts = pts[tet]
        A = 2.0 * (tet_pts[1:] - tet_pts[[0]])
        b = np.sum(tet_pts[1:] ** 2 - tet_pts[[0]] ** 2, axis=1)
        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        radius = float(np.linalg.norm(center - tet_pts[0]))
        if not np.isfinite(radius) or radius > alpha:
            continue
        for face_local, opp_local in local_faces:
            face = np.asarray([tet[idx] for idx in face_local], dtype=np.int32)
            opp = pts[tet[opp_local]]
            p0, p1, p2 = pts[face]
            normal = np.cross(p1 - p0, p2 - p0)
            if np.dot(normal, opp - p0) > 0.0:
                face[[1, 2]] = face[[2, 1]]
            key = tuple(sorted(int(v) for v in face))
            if key in face_dict:
                face_dict[key] = (face_dict[key][0], face_dict[key][1] + 1)
            else:
                face_dict[key] = (face, 1)

    boundary = [face for face, count in face_dict.values() if count == 1]
    faces = np.asarray(boundary, dtype=np.int32)
    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return pts[unique_vertices].astype(np.float32), inverse.reshape(-1, 3).astype(np.int32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soft sloth particle-spring system drops into a Newton MPM sand bed.")
    p.add_argument(
        "--sloth-ir",
        type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_sloth_normal/"
            "20260305_163731_full_parity/double_lift_sloth_normal_ir.npz"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="sloth_soft_mpm_sand")
    p.add_argument("--device", default=path_defaults.default_device())
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--sim-dt", type=float, default=1.0e-4)
    p.add_argument("--substeps", type=int, default=300)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument("--sand-half-extent-x", type=float, default=0.9)
    p.add_argument("--sand-half-extent-y", type=float, default=1.4)
    p.add_argument("--sand-height", type=float, default=0.18)
    p.add_argument("--hidden-floor-z", type=float, default=-1.25)

    p.add_argument("--sand-voxel-size", type=float, default=0.05)
    p.add_argument("--sand-particles-per-cell", type=int, default=3)
    p.add_argument("--sand-points-per-particle", type=int, default=1)
    p.add_argument("--sand-density", type=float, default=2500.0)
    p.add_argument("--sand-young-modulus", type=float, default=1.0e15)
    p.add_argument("--sand-poisson-ratio", type=float, default=0.3)
    p.add_argument("--sand-friction", type=float, default=0.68)
    p.add_argument("--sand-yield-pressure", type=float, default=1.0e12)
    p.add_argument("--sand-air-drag", type=float, default=1.0)
    p.add_argument("--sand-grid-type", choices=["fixed", "sparse"], default="sparse")
    p.add_argument("--sand-max-iterations", type=int, default=50)
    p.add_argument("--sand-transfer-scheme", choices=["apic", "pic"], default="pic")
    p.add_argument("--sand-critical-fraction", type=float, default=0.0)

    p.add_argument("--sloth-center-xy", type=float, nargs=2, default=(-0.15, 0.20))
    p.add_argument("--sloth-center-z", type=float, default=3.0)
    p.add_argument("--particle-mass", type=float, default=0.001)
    p.add_argument("--drag-damping-scale", type=float, default=0.0)
    p.add_argument("--spring-k-scale", type=float, default=0.0003)
    p.add_argument("--soft-mpm-force-scale", type=float, default=0.01)
    p.add_argument("--soft-mpm-max-dv-per-frame", type=float, default=0.0001)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--screen-width", type=int, default=960)
    p.add_argument("--screen-height", type=int, default=540)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--views", default="close_a")
    p.add_argument("--camera-fov", type=float, default=45.0)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=24)
    return p.parse_args()


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    ir = newton_import_ir.load_ir(path.resolve())
    required = ["x0", "v0", "mass", "num_object_points", "spring_edges", "spring_rest_length", "spring_ke", "spring_kd", "sim_dt", "sim_substeps"]
    missing = [key for key in required if key not in ir]
    if missing:
        raise KeyError(f"{path} missing IR fields: {missing}")
    return ir


def _load_surface_points(spec: SoftSpec) -> np.ndarray:
    case_dir = BRIDGE_ROOT / "inputs" / "cases" / spec.case_name
    final_data_path = case_dir / "final_data.pkl"
    if not final_data_path.exists():
        raise FileNotFoundError(f"Missing final_data.pkl: {final_data_path}")
    with final_data_path.open("rb") as handle:
        final_data = pickle.load(handle)
    surface = np.asarray(final_data["surface_points"], dtype=np.float32).reshape(-1, 3)
    if spec.reverse_z:
        surface[:, 2] *= -1.0
    surface += spec.placement_shift.astype(np.float32, copy=False)
    return surface


def _largest_component_indices(edges: np.ndarray, n_obj: int) -> np.ndarray:
    parent = np.arange(n_obj, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in np.asarray(edges, dtype=np.int32):
        union(int(a), int(b))

    buckets: dict[int, list[int]] = {}
    for i in range(n_obj):
        root = find(i)
        buckets.setdefault(root, []).append(i)
    largest = max(buckets.values(), key=len)
    return np.asarray(sorted(largest), dtype=np.int32)


def _copy_ir_for_demo(ir: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, bool]:
    ir_demo: dict[str, Any] = {}
    for key, value in ir.items():
        if isinstance(value, np.ndarray):
            ir_demo[key] = np.array(value, copy=True)
        else:
            ir_demo[key] = value

    reverse_z = newton_import_ir.ir_bool(ir_demo, "reverse_z", default=False)
    n_obj = int(np.asarray(ir_demo["num_object_points"]).ravel()[0])

    x0 = np.asarray(ir_demo["x0"], dtype=np.float32).copy()[:n_obj]
    v0 = np.zeros_like(np.asarray(ir_demo["v0"], dtype=np.float32), dtype=np.float32)[:n_obj]
    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0

    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    source_center = 0.5 * (bbox_min + bbox_max)
    target_center = np.array(
        [float(args.sloth_center_xy[0]), float(args.sloth_center_xy[1]), float(args.sloth_center_z)],
        dtype=np.float32,
    )
    shift = target_center - source_center
    x0 += shift
    ir_demo["x0"] = x0
    ir_demo["v0"] = v0

    mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()
    ir_demo["mass"] = np.full(n_obj, float(args.particle_mass), dtype=np.float32)
    ir_demo["collision_radius"] = np.asarray(ir_demo["collision_radius"], dtype=np.float32).copy()[:n_obj]
    ir_demo["num_object_points"] = np.asarray(n_obj, dtype=np.int32)

    edges = np.asarray(ir_demo["spring_edges"], dtype=np.int32)
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    ir_demo["spring_edges"] = edges[keep].astype(np.int32, copy=True)
    for key in (
        "spring_ke",
        "spring_kd",
        "spring_rest_length",
        "spring_y",
        "spring_y_raw",
    ):
        if key in ir_demo:
            ir_demo[key] = np.asarray(ir_demo[key], dtype=np.float32).copy().ravel()[keep]

    ir_demo.pop("controller_idx", None)
    ir_demo.pop("controller_traj", None)
    return ir_demo, shift.astype(np.float32), bool(reverse_z)


def _build_soft_model(ir: dict[str, np.ndarray], args: argparse.Namespace, device: str) -> tuple[newton.Model, DemoMeta, dict[str, Any]]:
    cfg_validate = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_k_scale),
        spring_kd_scale=float(args.spring_k_scale),
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
        add_ground_plane=False,
        strict_physics_checks=True,
        apply_drag=True,
        drag_damping_scale=float(args.drag_damping_scale),
        up_axis="Z",
        device=device,
    )
    newton_import_ir.validate_ir_physics(ir, cfg_validate)
    ir_demo, shift, reverse_z = _copy_ir_for_demo(ir, args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.sloth_ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_k_scale),
        spring_kd_scale=float(args.spring_k_scale),
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
        add_ground_plane=False,
        strict_physics_checks=False,
        apply_drag=True,
        drag_damping_scale=float(args.drag_damping_scale),
        up_axis="Z",
        device=device,
    )
    checks: dict[str, Any] = {}
    particle_contacts = newton_import_ir._resolve_particle_contacts(cfg, ir_demo)
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    radius_all, _, _ = newton_import_ir._add_particles(builder, ir_demo, cfg, particle_contacts)
    newton_import_ir._add_springs(builder, ir_demo, cfg, checks)

    hidden_cfg = builder.default_shape_cfg.copy()
    newton_import_ir._configure_ground_contact_material(hidden_cfg, ir_demo, cfg, checks, context="tray_container")
    hidden_cfg.is_visible = False

    builder.add_ground_plane(cfg=hidden_cfg)

    model = builder.finalize(device=device)
    if not (particle_contacts and (not cfg.disable_particle_contact_kernel)):
        model.particle_grid = None
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    newton_import_ir._apply_ps_object_collision_mapping(model, ir_demo, cfg, checks)

    n_obj = int(np.asarray(ir_demo["num_object_points"]).ravel()[0])
    edges = np.asarray(ir_demo["spring_edges"], dtype=np.int32)
    render_local_ids = _largest_component_indices(edges, n_obj)
    render_mask = np.zeros(n_obj, dtype=bool)
    render_mask[render_local_ids] = True
    render_edge_mask = render_mask[edges[:, 0]] & render_mask[edges[:, 1]]
    render_edges = edges[render_edge_mask][::24].astype(np.int32, copy=True)
    point_radius = np.maximum(radius_all[:n_obj].astype(np.float32)[render_local_ids] * 2.5, 0.008)
    spec = SoftSpec(
        name="sloth",
        case_name=str(np.asarray(ir_demo.get("case_name", np.array(["sloth"]))).ravel()[0]),
        color_rgb=(0.98, 0.66, 0.18),
        ir_path=args.sloth_ir.resolve(),
        particle_start=0,
        particle_count=n_obj,
        render_particle_ids_global=render_local_ids.astype(np.int32, copy=False),
        drag_damping=float(newton_import_ir.ir_scalar(ir_demo, "drag_damping", default=0.0)) * float(args.drag_damping_scale),
        render_edge_global=render_edges,
        render_point_radius=point_radius.astype(np.float32),
        mass_sum=float(np.asarray(ir_demo["mass"], dtype=np.float32)[:n_obj].sum()),
        placement_shift=shift.astype(np.float32),
        reverse_z=bool(reverse_z),
        collision_radius_mean=float(np.mean(radius_all[:n_obj])),
        collider_mu=float(newton_import_ir.ir_scalar(ir_demo, "contact_collide_object_fric", default=0.4)),
    )
    meta = DemoMeta(
        spec=spec,
        bed_half_extent_x=float(args.sand_half_extent_x),
        bed_half_extent_y=float(args.sand_half_extent_y),
        sand_height=float(args.sand_height),
        ground_z=0.0,
        scene_label="Scene: sloth softbody + native MPM sand",
    )
    return model, meta, ir_demo


def _model_particle_collider_body_ids(model: newton.Model) -> list[int]:
    shape_flags = model.shape_flags.numpy()
    body_ids: list[int] = []
    for body_id in range(-1, model.body_count):
        shape_ids = np.asarray(model.body_shapes.get(body_id, []), dtype=np.int32)
        if shape_ids.size == 0:
            continue
        if np.any((shape_flags[shape_ids] & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0):
            body_ids.append(int(body_id))
    return body_ids


def _build_proxy(model: newton.Model, meta: DemoMeta, args: argparse.Namespace, device: str) -> SoftProxy:
    state = model.state()
    q0 = state.particle_q.numpy().astype(np.float32)
    qd0 = state.particle_qd.numpy().astype(np.float32) if state.particle_qd is not None else np.zeros_like(q0, dtype=np.float32)
    spec = meta.spec
    obj_pts = q0[spec.render_particle_ids_global]
    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.asarray(obj_pts, dtype=np.float64))
    tri_local = np.asarray(hull.simplices, dtype=np.int32)
    used = np.unique(tri_local.reshape(-1))
    remap = {old: i for i, old in enumerate(used.tolist())}
    proxy_vertices = obj_pts[used].astype(np.float32, copy=True)
    tri_local = np.vectorize(remap.get, otypes=[np.int32])(tri_local).astype(np.int32, copy=False)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(obj_pts, dtype=np.float64))
    dists, idx = tree.query(np.asarray(proxy_vertices, dtype=np.float64), k=min(4, obj_pts.shape[0]))
    if idx.ndim == 1:
        idx = idx[:, None]
        dists = dists[:, None]
    weights = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-5)
    weights /= np.sum(weights, axis=1, keepdims=True)
    interp_ids = idx.astype(np.int32)
    proxy_points = np.sum(obj_pts[interp_ids] * weights[:, :, None], axis=1).astype(np.float32, copy=False)
    proxy_vel = np.sum(qd0[interp_ids] * weights[:, :, None], axis=1).astype(np.float32, copy=False)
    mesh_points = wp.array(proxy_points, dtype=wp.vec3, device=device)
    mesh_velocities = wp.array(proxy_vel, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(tri_local.reshape(-1).astype(np.int32), dtype=int, device=device)
    mesh = wp.Mesh(mesh_points, mesh_indices, mesh_velocities)
    return SoftProxy(
        collider_id=-1,
        interp_particle_ids_global=spec.render_particle_ids_global[interp_ids],
        interp_particle_weights=weights.astype(np.float32, copy=False),
        mesh_points=mesh_points,
        mesh_velocities=mesh_velocities,
        mesh=mesh,
        query_points_world=proxy_points.astype(np.float32, copy=True),
        collider_thickness=max(4.0 * spec.collision_radius_mean, 1.5 * float(args.sand_voxel_size)),
        collider_projection_threshold=max(2.0 * float(args.sand_voxel_size), 6.0 * spec.collision_radius_mean),
        collider_mu=spec.collider_mu,
    )


def _build_sand_system(model: newton.Model, meta: DemoMeta, args: argparse.Namespace, device: str) -> SandSystem:
    sand_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(sand_builder)

    lo = np.array(
        [
            -float(args.sand_half_extent_x),
            -float(args.sand_half_extent_y),
            0.0,
        ],
        dtype=np.float32,
    )
    hi = np.array(
        [
            float(args.sand_half_extent_x),
            float(args.sand_half_extent_y),
            float(args.sand_height),
        ],
        dtype=np.float32,
    )
    res = np.array(np.ceil(max(1, int(args.sand_particles_per_cell)) * (hi - lo) / max(float(args.sand_voxel_size), 1.0e-4)), dtype=int)
    cell = (hi - lo) / res
    cell_volume = float(np.prod(cell))
    radius = float(np.max(cell) * 0.5)
    mass = float(cell_volume * float(args.sand_density))
    sand_builder.add_particle_grid(
        pos=wp.vec3(lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1,
        dim_y=int(res[1]) + 1,
        dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]),
        cell_y=float(cell[1]),
        cell_z=float(cell[2]),
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
        flags=int(newton.ParticleFlags.ACTIVE),
    )

    sand_model = sand_builder.finalize(device=device)
    sand_model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    sel = wp.array(np.arange(sand_model.particle_count, dtype=np.int32), dtype=int, device=device)
    sand_model.mpm.young_modulus[sel].fill_(float(args.sand_young_modulus))
    sand_model.mpm.poisson_ratio[sel].fill_(float(args.sand_poisson_ratio))
    sand_model.mpm.friction[sel].fill_(float(args.sand_friction))
    sand_model.mpm.yield_pressure[sel].fill_(float(args.sand_yield_pressure))
    sand_model.mpm.tensile_yield_ratio[sel].fill_(0.0)
    sand_model.mpm.yield_stress[sel].fill_(0.0)

    options = newton.solvers.SolverImplicitMPM.Options()
    options.voxel_size = float(args.sand_voxel_size)
    options.tolerance = 1.0e-5
    options.transfer_scheme = str(args.sand_transfer_scheme)
    options.grid_type = str(args.sand_grid_type)
    options.grid_padding = 50 if args.sand_grid_type == "fixed" else 0
    options.max_active_cell_count = (1 << 16) if args.sand_grid_type == "fixed" else -1
    options.strain_basis = "P0"
    options.max_iterations = int(args.sand_max_iterations)
    options.critical_fraction = float(args.sand_critical_fraction)
    options.air_drag = float(args.sand_air_drag)
    options.collider_velocity_mode = "finite_difference"
    sand_solver = newton.solvers.SolverImplicitMPM(sand_model, options)

    proxy = _build_proxy(model, meta, args, device)
    body_collider_ids = _model_particle_collider_body_ids(model)
    collider_meshes = [None] * len(body_collider_ids) + [proxy.mesh]
    collider_body_ids = body_collider_ids + [None]
    ground_thickness = 0.75 * float(args.sand_voxel_size)
    ground_proj = 1.00 * float(args.sand_voxel_size)
    collider_thicknesses = [ground_thickness if body_id == -1 else None for body_id in body_collider_ids] + [proxy.collider_thickness]
    collider_projection_thresholds = [ground_proj if body_id == -1 else None for body_id in body_collider_ids] + [proxy.collider_projection_threshold]
    collider_friction = [None] * len(body_collider_ids) + [proxy.collider_mu]
    sand_solver.setup_collider(
        collider_meshes=collider_meshes,
        collider_body_ids=collider_body_ids,
        collider_thicknesses=collider_thicknesses,
        collider_friction=collider_friction,
        collider_projection_threshold=collider_projection_thresholds,
        model=model,
    )
    proxy.collider_id = len(body_collider_ids)

    combined_state = model.state()
    sand_state_0 = sand_model.state()
    sand_state_1 = sand_model.state()
    sand_state_0.body_q = wp.empty(0, dtype=wp.transform, device=device) if combined_state.body_q is None else wp.empty_like(combined_state.body_q)
    sand_state_0.body_qd = wp.empty(0, dtype=wp.spatial_vector, device=device) if combined_state.body_qd is None else wp.empty_like(combined_state.body_qd)
    sand_state_0.body_f = wp.empty(0, dtype=wp.spatial_vector, device=device) if combined_state.body_f is None else wp.empty_like(combined_state.body_f)
    sand_state_1.body_q = wp.empty_like(sand_state_0.body_q)
    sand_state_1.body_qd = wp.empty_like(sand_state_0.body_qd)
    sand_state_1.body_f = wp.empty_like(sand_state_0.body_f)

    render_radii = wp.clone(sand_model.particle_radius)
    render_colors = wp.full(sand_model.particle_count, value=wp.vec3(0.87, 0.84, 0.63), dtype=wp.vec3, device=device)
    max_nodes = 1 << 20
    return SandSystem(
        model=sand_model,
        solver=sand_solver,
        state_0=sand_state_0,
        state_1=sand_state_1,
        collider_impulses=wp.zeros(max_nodes, dtype=wp.vec3, device=device),
        collider_impulse_pos=wp.zeros(max_nodes, dtype=wp.vec3, device=device),
        collider_impulse_ids=wp.full(max_nodes, value=-1, dtype=int, device=device),
        collider_body_id=sand_solver.collider_body_index,
        body_sand_forces=None,
        frame_dt=float(args.sim_dt) * float(args.substeps),
        sim_dt=float(args.sim_dt),
        active_impulse_count=0,
        render_radii=render_radii,
        render_colors=render_colors,
        proxy=proxy,
    )


def _collect_sand_impulses(sand: SandSystem) -> None:
    impulses, pos, ids = sand.solver.collect_collider_impulses(sand.state_0)
    n = min(int(impulses.shape[0]), int(sand.collider_impulses.shape[0]))
    sand.collider_impulse_ids.fill_(-1)
    if n > 0:
        sand.collider_impulses[:n].assign(impulses[:n])
        sand.collider_impulse_pos[:n].assign(pos[:n])
        sand.collider_impulse_ids[:n].assign(ids[:n])
    sand.active_impulse_count = n


def _update_proxy_from_soft_state(state: newton.State, proxy: SoftProxy) -> None:
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


def _update_soft_particle_forces_from_sand(model: newton.Model, meta: DemoMeta, sand: SandSystem, args: argparse.Namespace) -> dict[str, float]:
    from scipy.spatial import cKDTree

    particle_count = model.particle_count
    forces = np.zeros((particle_count, 3), dtype=np.float32)
    n = int(sand.active_impulse_count)
    if n <= 0:
        return {"sample_count": 0.0, "impulse_norm_sum": 0.0, "force_norm_max": 0.0, "forces": forces}

    ids = sand.collider_impulse_ids.numpy()[:n].astype(np.int32, copy=False)
    impulses = sand.collider_impulses.numpy()[:n].astype(np.float32, copy=False)
    positions = sand.collider_impulse_pos.numpy()[:n].astype(np.float32, copy=False)
    mask = ids == int(sand.proxy.collider_id)
    if not np.any(mask):
        return {"sample_count": 0.0, "impulse_norm_sum": 0.0, "force_norm_max": 0.0, "forces": forces}

    pts = positions[mask]
    imp = impulses[mask] * (float(args.soft_mpm_force_scale) / max(float(sand.frame_dt), 1.0e-8))
    proxy_points = np.asarray(sand.proxy.query_points_world, dtype=np.float64)
    tree = cKDTree(proxy_points)
    k = min(4, proxy_points.shape[0])
    dist, nn = tree.query(pts.astype(np.float64, copy=False), k=k)
    if k == 1:
        dist = dist[:, None]
        nn = nn[:, None]
    proxy_w = 1.0 / np.maximum(dist.astype(np.float32), 1.0e-5)
    proxy_w /= np.sum(proxy_w, axis=1, keepdims=True)
    particle_ids = sand.proxy.interp_particle_ids_global[nn]
    particle_weights = sand.proxy.interp_particle_weights[nn]
    combo = proxy_w[:, :, None] * particle_weights
    for proxy_col in range(particle_ids.shape[1]):
        for bind_col in range(particle_ids.shape[2]):
            np.add.at(forces, particle_ids[:, proxy_col, bind_col], imp * combo[:, proxy_col, bind_col : bind_col + 1])

    masses = model.particle_mass.numpy().astype(np.float32)
    max_force = masses * (float(args.soft_mpm_max_dv_per_frame) / max(float(args.sim_dt) * float(args.substeps), 1.0e-8))
    norms = np.linalg.norm(forces, axis=1)
    scale = np.ones_like(norms, dtype=np.float32)
    over = norms > np.maximum(max_force, 1.0e-8)
    scale[over] = max_force[over] / np.maximum(norms[over], 1.0e-8)
    forces *= scale[:, None]

    return {
        "sample_count": float(pts.shape[0]),
        "impulse_norm_sum": float(np.linalg.norm(imp, axis=1).sum()),
        "force_norm_max": float(np.linalg.norm(forces, axis=1).max(initial=0.0)),
        "forces": forces,
    }


def _tray_outline(meta: DemoMeta, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext_x = float(meta.bed_half_extent_x)
    ext_y = float(meta.bed_half_extent_y)
    z0 = float(meta.ground_z)
    z1 = float(meta.sand_height)
    corners = np.asarray(
        [
            (-ext_x, -ext_y, z0), (ext_x, -ext_y, z0), (ext_x, ext_y, z0), (-ext_x, ext_y, z0),
            (-ext_x, -ext_y, z1), (ext_x, -ext_y, z1), (ext_x, ext_y, z1), (-ext_x, ext_y, z1),
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
    colors = np.tile(np.asarray([[0.88, 0.90, 0.94]], dtype=np.float32), (starts.shape[0], 1))
    return starts, ends, colors


def _ground_grid(size: float = 3.0, steps: int = 12, z: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.linspace(-size, size, steps + 1, dtype=np.float32)
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    for v in vals:
        starts.append([-size, float(v), z])
        ends.append([size, float(v), z])
        starts.append([float(v), -size, z])
        ends.append([float(v), size, z])
    starts_np = np.asarray(starts, dtype=np.float32)
    ends_np = np.asarray(ends, dtype=np.float32)
    colors_np = np.tile(np.asarray([[0.17, 0.24, 0.33]], dtype=np.float32), (starts_np.shape[0], 1))
    return starts_np, ends_np, colors_np


def render_view_mp4(
    *,
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: DemoMeta,
    args: argparse.Namespace,
    device: str,
    out_mp4: Path,
):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps) / max(float(args.slowdown), 1.0e-6)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps_out:.6f}", "-i", "-", "-an",
        "-vcodec", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", str(out_mp4),
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
        target = np.array([0.0, 0.0, max(0.45 * meta.sand_height, 0.22)], dtype=np.float32)
        cpos = _camera_position(target, -34.0, -14.0, 4.3)
        viewer.set_camera(wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])), -14.0, -34.0)
        sand_points_wp = wp.empty(sim_data["sand_points_all"].shape[1], dtype=wp.vec3, device=device)
        sand_radii_wp = wp.array(sim_data["sand_point_radii"], dtype=wp.float32, device=device)
        sand_colors_wp = wp.array(sim_data["sand_point_colors"], dtype=wp.vec3, device=device)
        ground_starts_np, ground_ends_np, ground_colors_np = _ground_grid(size=4.0, steps=18, z=float(meta.ground_z))
        ground_starts_wp = wp.array(ground_starts_np, dtype=wp.vec3, device=device)
        ground_ends_wp = wp.array(ground_ends_np, dtype=wp.vec3, device=device)
        ground_colors_wp = wp.array(ground_colors_np, dtype=wp.vec3, device=device)
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["particle_q_all"].shape[0])
        for frame_idx in range(n_frames):
            q_all = sim_data["particle_q_all"][frame_idx].astype(np.float32, copy=False)
            sand_points_wp.assign(sim_data["sand_points_all"][frame_idx].astype(np.float32, copy=False))
            state.particle_q.assign(q_all)
            if state.particle_qd is not None:
                state.particle_qd.zero_()
            sim_t = float(frame_idx) * float(sim_data["sim_dt"]) * float(sim_data["substeps"])
            viewer.begin_frame(sim_t)
            sloth_pts = wp.array(q_all[meta.spec.render_particle_ids_global], dtype=wp.vec3, device=device)
            sloth_radii = wp.array(meta.spec.render_point_radius.astype(np.float32), dtype=wp.float32, device=device)
            sloth_colors = wp.array(
                np.tile(np.asarray(meta.spec.color_rgb, dtype=np.float32), (meta.spec.render_particle_ids_global.shape[0], 1)),
                dtype=wp.vec3,
                device=device,
            )
            viewer.log_points("/demo/sloth", sloth_pts, sloth_radii, sloth_colors, hidden=False)
            if meta.spec.render_edge_global.size > 0:
                starts = q_all[meta.spec.render_edge_global[:, 0]]
                ends = q_all[meta.spec.render_edge_global[:, 1]]
                viewer.log_lines(
                    "/demo/sloth_springs",
                    wp.array(starts, dtype=wp.vec3, device=device),
                    wp.array(ends, dtype=wp.vec3, device=device),
                    wp.array(np.tile(np.asarray(meta.spec.color_rgb, dtype=np.float32), (starts.shape[0], 1)), dtype=wp.vec3, device=device),
                    width=1.0,
                    hidden=False,
                )
            viewer.log_points("/demo/sand", sand_points_wp, sand_radii_wp, sand_colors_wp, hidden=False)
            viewer.log_lines("/demo/ground", ground_starts_wp, ground_ends_wp, ground_colors_wp, width=1.0, hidden=False)
            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                frame = _overlay_text_lines_rgb(frame, ["Close View", meta.scene_label, f"frame {frame_idx+1:03d}/{n_frames:03d}  t={sim_t:.3f}s"], font_size=int(args.label_font_size))
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


class Example:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.args.out_dir = self.args.out_dir.resolve()
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        wp.init()
        self.device = newton_import_ir.resolve_device(self.args.device)
        self.ir = _load_ir(self.args.sloth_ir)
        self.model, self.meta, self.demo_ir = _build_soft_model(self.ir, self.args, self.device)
        self.sand = _build_sand_system(self.model, self.meta, self.args, self.device)
        self.sim_data: dict[str, Any] | None = None

    def step(self):
        solver = newton.solvers.SolverSemiImplicit(
            self.model,
            angular_damping=0.05,
            friction_smoothing=1.0,
            enable_tri_contact=True,
        )
        contacts = self.model.contacts()
        control = self.model.control()
        state_in = self.model.state()
        state_out = self.model.state()
        q_hist: list[np.ndarray] = [state_in.particle_q.numpy().astype(np.float32)]
        sand_hist: list[np.ndarray] = [self.sand.state_0.particle_q.numpy().astype(np.float32)]
        coupling_sample_hist: list[int] = [0]
        coupling_impulse_hist: list[float] = [0.0]
        t0 = time.perf_counter()
        _collect_sand_impulses(self.sand)
        frame_dt = float(self.args.sim_dt) * float(self.args.substeps)
        for frame in range(1, int(self.args.frames)):
            coupling = _update_soft_particle_forces_from_sand(self.model, self.meta, self.sand, self.args)
            coupling_forces_wp = wp.array(coupling["forces"], dtype=wp.vec3, device=self.device)
            for _sub in range(int(self.args.substeps)):
                state_in.clear_forces()
                feedback_scale = float(_substep_feedback_scale(_sub, int(self.args.substeps)))
                wp.launch(
                    add_dense_particle_forces,
                    dim=self.model.particle_count,
                    inputs=[state_in.particle_f, coupling_forces_wp, feedback_scale],
                    device=self.device,
                )
                self.model.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, float(self.args.sim_dt))
                state_in, state_out = state_out, state_in
                if self.meta.spec.drag_damping > 0.0:
                    wp.launch(
                        apply_drag_range,
                        dim=self.meta.spec.particle_count,
                        inputs=[state_in.particle_q, state_in.particle_qd, 0, self.meta.spec.particle_count, float(self.args.sim_dt), self.meta.spec.drag_damping],
                        device=self.device,
                    )
            _update_proxy_from_soft_state(state_in, self.sand.proxy)
            self.sand.solver.step(self.sand.state_0, self.sand.state_1, contacts=None, control=None, dt=frame_dt)
            self.sand.state_0, self.sand.state_1 = self.sand.state_1, self.sand.state_0
            _collect_sand_impulses(self.sand)

            q_np = state_in.particle_q.numpy().astype(np.float32)
            sand_np = self.sand.state_0.particle_q.numpy().astype(np.float32)
            if not np.all(np.isfinite(q_np)):
                raise RuntimeError(f"Softbody state contains non-finite values at frame {frame}")
            if not np.all(np.isfinite(sand_np)):
                raise RuntimeError(f"Sand state contains non-finite values at frame {frame}")
            q_render = q_np[self.meta.spec.render_particle_ids_global]
            bbox_size = float(np.linalg.norm(np.max(q_render, axis=0) - np.min(q_render, axis=0)))
            if bbox_size > 200.0:
                raise RuntimeError(f"Softbody exploded at frame {frame}: bbox_size={bbox_size:.3f}")
            q_hist.append(q_np)
            sand_hist.append(sand_np)
            coupling_sample_hist.append(int(coupling["sample_count"]))
            coupling_impulse_hist.append(float(coupling["impulse_norm_sum"]))
            if frame == 1 or (frame + 1) % 5 == 0 or frame + 1 == int(self.args.frames):
                print(
                    f"  frame {frame+1}/{int(self.args.frames)}"
                    f"  z=[{q_np[:, 2].min():.3f}, {q_np[:, 2].max():.3f}]"
                    f"  bbox={bbox_size:.3f}"
                    f"  coupling_samples={int(coupling['sample_count'])}",
                    flush=True,
                )
        self.sim_data = {
            "particle_q_all": np.stack(q_hist),
            "sand_points_all": np.stack(sand_hist),
            "sand_point_radii": self.sand.render_radii.numpy().astype(np.float32),
            "sand_point_colors": self.sand.render_colors.numpy().astype(np.float32),
            "coupling_sample_count": np.asarray(coupling_sample_hist, dtype=np.int32),
            "coupling_impulse_norm_sum": np.asarray(coupling_impulse_hist, dtype=np.float32),
            "sim_dt": float(self.args.sim_dt),
            "substeps": int(self.args.substeps),
            "wall_time_sec": float(time.perf_counter() - t0),
        }

    def render(self):
        assert self.sim_data is not None
        out_mp4 = self.args.out_dir / f"{self.args.prefix}_close_a.mp4"
        render_view_mp4(model=self.model, sim_data=self.sim_data, meta=self.meta, args=self.args, device=self.device, out_mp4=out_mp4)
        scene_npz = self.args.out_dir / f"{self.args.prefix}_scene.npz"
        np.savez_compressed(
            scene_npz,
            particle_q_all=self.sim_data["particle_q_all"],
            sand_points_all=self.sim_data["sand_points_all"],
            sand_point_radii=self.sim_data["sand_point_radii"],
            sand_point_colors=self.sim_data["sand_point_colors"],
            coupling_sample_count=self.sim_data["coupling_sample_count"],
            coupling_impulse_norm_sum=self.sim_data["coupling_impulse_norm_sum"],
            sim_dt=np.float32(self.sim_data["sim_dt"]),
            substeps=np.int32(self.sim_data["substeps"]),
            wall_time_sec=np.float32(self.sim_data["wall_time_sec"]),
        )
        summary = {
            "experiment": "sloth_soft_mpm_sand_demo",
            "sloth_ir": str(self.args.sloth_ir.resolve()),
            "frames": int(self.sim_data["particle_q_all"].shape[0]),
            "sim_dt": float(self.sim_data["sim_dt"]),
            "substeps": int(self.sim_data["substeps"]),
            "wall_time_sec": float(self.sim_data["wall_time_sec"]),
            "sand_half_extent_x": float(self.args.sand_half_extent_x),
            "sand_half_extent_y": float(self.args.sand_half_extent_y),
            "sand_height": float(self.args.sand_height),
            "sand_voxel_size": float(self.args.sand_voxel_size),
            "sand_particles_per_cell": int(self.args.sand_particles_per_cell),
            "sand_density": float(self.args.sand_density),
            "sand_transfer_scheme": str(self.args.sand_transfer_scheme),
            "sand_grid_type": str(self.args.sand_grid_type),
            "particle_mass": float(self.args.particle_mass),
            "frames_with_coupling": int(np.count_nonzero(self.sim_data["coupling_sample_count"])),
            "coupling_samples_total": int(np.sum(self.sim_data["coupling_sample_count"])),
            "coupling_impulse_norm_total": float(np.sum(self.sim_data["coupling_impulse_norm_sum"])),
            "panel_videos": [str(out_mp4)],
        }
        (self.args.out_dir / f"{self.args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run(self) -> int:
        print("Building soft sloth + sand system...", flush=True)
        self.step()
        self.render()
        return 0


def main() -> int:
    return Example(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
