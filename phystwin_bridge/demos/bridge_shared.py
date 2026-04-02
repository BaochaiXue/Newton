#!/usr/bin/env python3
"""Shared utilities for standalone PhysTwin bridge demos."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import warp as wp

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
EARTH_GROUND_COLOR = (0.76, 0.66, 0.46)


def load_core_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ir_checked(path: Path, importer: Any, required: list[str]) -> dict[str, np.ndarray]:
    ir = importer.load_ir(path.resolve())
    missing = [key for key in required if key not in ir]
    if missing:
        raise KeyError(f"{path} missing IR fields: {missing}")
    return ir


def load_surface_points(
    case_name: str,
    *,
    reverse_z: bool = False,
    placement_shift: np.ndarray | None = None,
) -> np.ndarray:
    case_dir = BRIDGE_ROOT / "inputs" / "cases" / case_name
    final_data_path = case_dir / "final_data.pkl"
    if not final_data_path.exists():
        raise FileNotFoundError(f"Missing final_data.pkl: {final_data_path}")
    with final_data_path.open("rb") as handle:
        final_data = pickle.load(handle)
    surface = np.asarray(final_data["surface_points"], dtype=np.float32).reshape(-1, 3)
    if reverse_z:
        surface[:, 2] *= -1.0
    if placement_shift is not None:
        surface += np.asarray(placement_shift, dtype=np.float32).reshape(1, 3)
    return surface


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


@wp.func
def _pair_penalty_contact_force(
    n: wp.vec3,
    v_rel: wp.vec3,
    signed_gap: float,
    k_n: float,
    k_d: float,
    k_f: float,
    k_mu: float,
):
    """Returns a pairwise penalty contact force.

    Physics note:
    - ``signed_gap < 0`` means the two particle spheres overlap.
    - The normal response combines an elastic penalty ``k_n * signed_gap``
      with approach-only damping ``k_d * min(v_n, 0)``.
    - The tangential response is friction-limited so the lateral force does
      not exceed ``mu * |f_n|``.
    """

    v_n = wp.dot(n, v_rel)
    normal_impulse = signed_gap * k_n + wp.min(v_n, 0.0) * k_d

    v_t = v_rel - n * v_n
    v_t_mag = wp.length(v_t)
    if v_t_mag > 0.0:
        v_t = v_t / v_t_mag

    tangential_impulse = wp.min(v_t_mag * k_f, k_mu * wp.abs(normal_impulse))
    return -n * normal_impulse - v_t * tangential_impulse


def compute_visual_particle_radii(
    particle_radius: np.ndarray,
    *,
    radius_scale: float | None,
    radius_cap: float | None,
) -> np.ndarray:
    """Builds render-only radii from physical radii.

    The physics model keeps using the original ``particle_radius`` values.
    If no render override is requested, rendering uses the physical particle
    radius directly. Optional scale/cap overrides still remain available for
    cases that intentionally want a different visual thickness.
    """

    radii = np.array(particle_radius, dtype=np.float32, copy=False)
    if radius_scale is None and radius_cap is None:
        return radii.astype(np.float32, copy=False)
    scaled = radii * (1.0 if radius_scale is None else float(radius_scale))
    if radius_cap is None:
        return scaled.astype(np.float32, copy=False)
    return np.minimum(scaled, float(radius_cap)).astype(np.float32, copy=False)


@contextmanager
def temporary_particle_radius_override(
    model: Any, render_radii: np.ndarray | None
) -> Iterator[None]:
    """Temporarily swaps in render-only radii and restores the physical ones."""

    if render_radii is None or getattr(model, "particle_radius", None) is None:
        yield
        return

    physical_radii = model.particle_radius.numpy().astype(np.float32, copy=False).copy()
    model.particle_radius.assign(np.array(render_radii, dtype=np.float32, copy=False))
    try:
        yield
    finally:
        model.particle_radius.assign(physical_radii)


def overlay_text_lines_rgb(
    frame: np.ndarray,
    lines: list[str],
    *,
    font_size: int = 24,
    x: int = 18,
    y: int = 16,
    line_gap: int = 6,
    bg_alpha: int = 110,
) -> np.ndarray:
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

    y_cursor = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.rectangle((x - 8, y_cursor - 4, x + bbox[2] + 8, y_cursor + bbox[3] + 4), fill=(0, 0, 0, bg_alpha))
        draw.text((x, y_cursor), line, fill=(255, 255, 255, 255), font=font)
        y_cursor += (bbox[3] - bbox[1]) + line_gap
    return np.asarray(img, dtype=np.uint8)


def apply_viewer_shape_colors(
    viewer: Any,
    model: Any,
    *,
    extra_rules: list[tuple[Callable[[str], bool], tuple[float, float, float]]] | None = None,
    ground_color: tuple[float, float, float] = EARTH_GROUND_COLOR,
) -> dict[int, tuple[float, float, float]]:
    """Applies a consistent presentation palette to ViewerGL shapes.

    We keep the logic at the bridge layer so every offline MP4 / GIF can share
    the same readable earth-tone ground without touching Newton core rendering.
    """

    labels = list(getattr(model, "shape_label", []))
    shape_colors: dict[int, tuple[float, float, float]] = {}
    for idx, label in enumerate(labels):
        name = str(label).lower()
        if "ground" in name or "plane" in name:
            shape_colors[idx] = tuple(float(c) for c in ground_color)
        if extra_rules:
            for predicate, color in extra_rules:
                if predicate(name):
                    shape_colors[idx] = tuple(float(c) for c in color)
                    break
    if shape_colors:
        viewer.update_shape_colors(shape_colors)
    return shape_colors


def camera_position(target: np.ndarray, yaw_deg: float, pitch_deg: float, distance: float) -> np.ndarray:
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


def alpha_shape_surface_mesh(points: np.ndarray, *, min_boundary_faces: int = 4) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import Delaunay, cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        raise ValueError(f"Need at least 4 points for alpha shape, got {pts.shape[0]}")
    tree = cKDTree(pts)
    k = min(8, pts.shape[0])
    dists, _ = tree.query(pts, k=k)
    nn_col = min(4, dists.shape[1] - 1)
    spacing = float(np.median(dists[:, nn_col]))
    alpha = max(2.5 * spacing, 1.0e-5)

    tetra = Delaunay(pts)
    face_dict: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    local_faces = (((0, 2, 1), 3), ((0, 1, 3), 2), ((0, 3, 2), 1), ((1, 2, 3), 0))
    for tet in np.asarray(tetra.simplices, dtype=np.int32):
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
    if len(boundary) < min_boundary_faces:
        raise ValueError("Alpha shape surface mesh is degenerate")
    faces = np.asarray(boundary, dtype=np.int32)
    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return pts[unique_vertices].astype(np.float32), inverse.reshape(-1, 3).astype(np.int32)


def spawn_mpm_particle_block(
    builder: Any,
    *,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    voxel_size: float,
    particles_per_cell: int,
    density: float,
    flags: int,
    jitter_scale: float = 1.0,
) -> np.ndarray:
    res = np.array(
        np.ceil(max(1, int(particles_per_cell)) * (bounds_hi - bounds_lo) / max(float(voxel_size), 1.0e-4)),
        dtype=int,
    )
    cell = (bounds_hi - bounds_lo) / res
    cell_volume = float(np.prod(cell))
    radius = float(np.max(cell) * 0.5)
    mass = float(cell_volume * density)
    begin_id = len(builder.particle_q)
    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1,
        dim_y=int(res[1]) + 1,
        dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]),
        cell_y=float(cell[1]),
        cell_z=float(cell[2]),
        mass=mass,
        jitter=float(jitter_scale) * radius,
        radius_mean=radius,
        flags=flags,
    )
    end_id = len(builder.particle_q)
    return np.arange(begin_id, end_id, dtype=np.int32)


def model_particle_collider_body_ids(model: Any) -> list[int]:
    import newton

    shape_flags = model.shape_flags.numpy()
    body_ids: list[int] = []
    for body_id in range(-1, model.body_count):
        shape_ids = np.asarray(model.body_shapes.get(body_id, []), dtype=np.int32)
        if shape_ids.size == 0:
            continue
        if np.any((shape_flags[shape_ids] & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0):
            body_ids.append(int(body_id))
    return body_ids


def ground_grid(
    size: float,
    *,
    steps: int = 12,
    z: float = 0.0,
    color: tuple[float, float, float] = EARTH_GROUND_COLOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.linspace(-size, size, 2 * int(steps) + 1, dtype=np.float32)
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    for v in vals:
        starts.append([-size, float(v), z])
        ends.append([size, float(v), z])
        starts.append([float(v), -size, z])
        ends.append([float(v), size, z])
    starts_np = np.asarray(starts, dtype=np.float32)
    ends_np = np.asarray(ends, dtype=np.float32)
    colors_np = np.tile(np.asarray([color], dtype=np.float32), (starts_np.shape[0], 1))
    return starts_np, ends_np, colors_np
