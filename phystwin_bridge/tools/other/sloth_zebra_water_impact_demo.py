#!/usr/bin/env python3
"""Dramatic Newton demo: PhysTwin sloth/zebra drop onto native-water bunny rigids.

Scene
-----
- Four native Newton bunny rigid bodies interact with a native Newton MPM water bed.
- Two PhysTwin soft bodies (sloth + zebra) are imported from IR, dropped from
  rest roughly 1 m above the water, and collide with the bunny rigids below.
- The script renders four camera views and composes them into one labeled 2x2 MP4.

Design constraints
------------------
- Soft-body spring parameters are loaded from PhysTwin IR without hard-coded rewrites.
- Internal spring semantics remain the exported PhysTwin->Newton mapping:
  ``spring_ke = spring_Y / rest_length`` and ``spring_kd`` from IR.
- Scene-specific additions are limited to the native Newton MPM water setup and
  to rendering/camera layout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
import warp as wp  # noqa: E402


@dataclass
class SoftSpec:
    name: str
    color_rgb: tuple[float, float, float]
    ir_path: Path
    particle_start: int
    particle_count: int
    drag_damping: float
    spring_edge_local: np.ndarray
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    mass_sum: float
    strict_export: bool
    spring_ke_mode: str
    spring_ke_rel_error_max: float
    bbox_extent: tuple[float, float, float]


@dataclass
class ViewSpec:
    name: str
    label: str
    yaw_deg: float
    pitch_deg: float
    distance: float


@dataclass
class DemoMeta:
    mesh_verts_local: np.ndarray
    mesh_tri_indices: np.ndarray
    mesh_render_edges: np.ndarray
    mesh_scale: float
    mesh_asset_path: str
    object_specs: list[SoftSpec]
    rigid_labels: list[str]
    rigid_cluster_center: np.ndarray
    water_level: float
    hidden_floor_z: float


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
    frame_dt: float
    active_impulse_count: int


def _water_soft_direct_coupling_supported() -> bool:
    # Newton's current implicit MPM collider path is shape-based. The imported
    # PhysTwin soft objects are particles + springs without collision shapes, so
    # they are not direct MPM colliders in this demo.
    return False


@wp.kernel
def _apply_drag_range(
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


@wp.kernel
def _compute_body_forces_from_water(
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
def _subtract_body_force_for_water_step(
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a dramatic multi-view Newton demo: sloth + zebra from PhysTwin "
            "drop onto native bunny rigid bodies in MPM water."
        )
    )
    p.add_argument(
        "--sloth-ir",
        type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_sloth_normal/"
            "20260305_163731_full_parity/double_lift_sloth_normal_ir.npz"
        ),
    )
    p.add_argument(
        "--zebra-ir",
        type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_zebra_normal/"
            "20260305_163731_full_parity/double_lift_zebra_normal_ir.npz"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="sloth_zebra_water_impact")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--water-level", type=float, default=0.0)
    p.add_argument("--drop-gap", type=float, default=1.0)
    p.add_argument("--hidden-floor-z", type=float, default=-1.25)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument("--sloth-target-xy", type=float, nargs=2, default=(-0.24, -0.08))
    p.add_argument("--zebra-target-xy", type=float, nargs=2, default=(0.24, 0.08))

    p.add_argument("--rigid-shape", choices=["bunny"], default="bunny")
    p.add_argument("--rigid-count", type=int, default=4)
    p.add_argument("--rigid-mass", type=float, default=90.0)
    p.add_argument("--rigid-scale", type=float, default=0.16)
    p.add_argument("--rigid-body-mu", type=float, default=0.55)
    p.add_argument("--rigid-body-ke", type=float, default=2.0e4)
    p.add_argument("--rigid-body-kd", type=float, default=7.0e2)
    p.add_argument("--rigid-body-kf", type=float, default=4.0e2)
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")

    p.add_argument("--float-submerge-fraction", type=float, default=0.46)

    p.add_argument("--water-extent", type=float, default=1.45)
    p.add_argument("--water-depth", type=float, default=0.34)
    p.add_argument("--water-wall-thickness", type=float, default=0.08)
    p.add_argument("--water-wall-height", type=float, default=0.28)
    p.add_argument("--water-voxel-size", type=float, default=0.14)
    p.add_argument("--water-particles-per-cell", type=int, default=2)
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
    p.add_argument("--water-max-iterations", type=int, default=40)
    p.add_argument("--water-points-per-particle", type=int, default=4)

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)
    p.add_argument("--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--spring-line-width", type=float, default=0.008)
    p.add_argument("--render-edge-stride", type=int, default=14)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.2)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.012)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--screen-width", type=int, default=960)
    p.add_argument("--screen-height", type=int, default=540)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-fov", type=float, default=50.0)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=26)
    p.add_argument(
        "--render-only-scene",
        type=Path,
        default=None,
        help="Skip simulation and render from an existing *_scene.npz produced by this script.",
    )
    p.add_argument(
        "--views",
        default="wide,close_a,close_b,close_c",
        help="Comma-separated render views to emit.",
    )
    return p.parse_args()


def _output_stem(prefix: str) -> str:
    return prefix


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stdout}")


def load_bunny_mesh(asset_name: str, prim_path: str):
    import newton.examples
    import newton.usd
    from pxr import Usd

    asset_path = Path(newton.examples.get_asset(asset_name))
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    stage = Usd.Stage.Open(str(asset_path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path} in {asset_path}")

    mesh = newton.usd.get_mesh(prim)
    points = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if indices.size % 3 != 0:
        raise ValueError(f"Bunny mesh index length must be multiple of 3, got {indices.size}")
    tri_indices = indices.reshape(-1, 3)

    e01 = tri_indices[:, [0, 1]]
    e12 = tri_indices[:, [1, 2]]
    e20 = tri_indices[:, [2, 0]]
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0).astype(np.int32, copy=False)
    return mesh, points, tri_indices, all_edges, str(asset_path)


def quat_to_rotmat(q_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray):
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _overlay_text_lines_rgb(frame: np.ndarray, lines: list[str], font_size: int = 28) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    if frame.dtype != np.uint8:
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
    pad_x = 8
    pad_y = 5
    gap = 7
    for line in lines:
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle(
            (x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y),
            fill=(0, 0, 0, 145),
        )
        draw.text(
            (x, y),
            line,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
        y += th + gap
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
    front /= max(float(np.linalg.norm(front)), 1e-6)
    return target.astype(np.float32) - front * float(distance)


def _compose_2x2(inputs: list[Path], labels: list[str], out_mp4: Path) -> None:
    if len(inputs) != 4 or len(labels) != 4:
        raise ValueError("compose_2x2 expects 4 inputs and 4 labels")
    filters: list[str] = []
    for idx, text in enumerate(labels):
        safe = text.replace("'", "\\'")
        filters.append(
            f"[{idx}:v]drawtext=text='{safe}':"
            "x=20:y=20:fontsize=30:fontcolor=white:"
            "box=1:boxcolor=black@0.62:boxborderw=8[v"
            f"{idx}]"
        )
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[vout]")
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd.extend(["-i", str(inp)])
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_mp4),
        ]
    )
    _run_ffmpeg(cmd)


def _parse_view_names(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("--views must contain at least one view name")
    known = {spec.name for spec in _view_specs()}
    unknown = [item for item in items if item not in known]
    if unknown:
        raise ValueError(f"Unknown view(s): {unknown}. Known: {sorted(known)}")
    return items


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    ir = newton_import_ir.load_ir(path.resolve())
    required = [
        "x0",
        "v0",
        "mass",
        "num_object_points",
        "spring_edges",
        "spring_rest_length",
        "spring_ke",
        "spring_kd",
        "drag_damping",
        "sim_dt",
        "sim_substeps",
        "reverse_z",
    ]
    missing = [key for key in required if key not in ir]
    if missing:
        raise KeyError(f"{path} missing IR fields: {missing}")
    return ir


def _validate_ir_pair(sloth_ir: dict[str, np.ndarray], zebra_ir: dict[str, np.ndarray]) -> None:
    checks = {
        "sim_dt": (newton_import_ir.ir_scalar(sloth_ir, "sim_dt"), newton_import_ir.ir_scalar(zebra_ir, "sim_dt")),
        "sim_substeps": (
            int(newton_import_ir.ir_scalar(sloth_ir, "sim_substeps")),
            int(newton_import_ir.ir_scalar(zebra_ir, "sim_substeps")),
        ),
        "reverse_z": (
            bool(newton_import_ir.ir_bool(sloth_ir, "reverse_z")),
            bool(newton_import_ir.ir_bool(zebra_ir, "reverse_z")),
        ),
    }
    for key, (a, b) in checks.items():
        if a != b:
            raise ValueError(f"IR mismatch for {key}: sloth={a!r}, zebra={b!r}")


def _spring_formula_rel_error(ir: dict[str, np.ndarray]) -> float:
    if "spring_y" not in ir or "spring_rest_length" not in ir or "spring_ke" not in ir:
        return float("nan")
    spring_y = np.asarray(ir["spring_y"], dtype=np.float64).ravel()
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float64).ravel()
    ke = np.asarray(ir["spring_ke"], dtype=np.float64).ravel()
    denom = np.maximum(np.abs(ke), 1.0e-12)
    return float(np.max(np.abs((spring_y / rest) - ke) / denom))


def _prepare_soft_object(
    *,
    builder: newton.ModelBuilder,
    ir: dict[str, np.ndarray],
    ir_path: Path,
    name: str,
    color_rgb: tuple[float, float, float],
    particle_start: int,
    target_xy: tuple[float, float],
    target_bottom_z: float,
    render_edge_stride: int,
    particle_radius_vis_scale: float,
    particle_radius_vis_min: float,
) -> SoftSpec:
    reverse_z = newton_import_ir.ir_bool(ir, "reverse_z", default=False)
    x0 = np.asarray(ir["x0"], dtype=np.float32).copy()
    v0 = np.asarray(ir["v0"], dtype=np.float32).copy()
    mass = np.asarray(ir["mass"], dtype=np.float32).copy()
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0

    pos = x0[:n_obj].copy()
    vel = np.zeros_like(pos, dtype=np.float32)
    mass_obj = mass[:n_obj].copy()
    radius_all, _, _ = newton_import_ir.resolve_collision_radius(ir, x0.shape[0])
    radius_obj = radius_all[:n_obj].astype(np.float32, copy=True)

    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    shift = np.array(
        [
            float(target_xy[0]) - float(center_xy[0]),
            float(target_xy[1]) - float(center_xy[1]),
            float(target_bottom_z) - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    pos += shift

    builder.add_particles(
        pos=[tuple(row.tolist()) for row in pos],
        vel=[tuple(row.tolist()) for row in vel],
        mass=mass_obj.astype(float).tolist(),
        radius=radius_obj.astype(float).tolist(),
        flags=[int(newton.ParticleFlags.ACTIVE)] * n_obj,
    )

    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    edges_obj = edges[keep].astype(np.int32, copy=True)
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).ravel()[keep]
    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel()[keep]
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel()[keep]

    for idx in range(edges_obj.shape[0]):
        i, j = int(edges_obj[idx, 0]), int(edges_obj[idx, 1])
        builder.add_spring(
            i=particle_start + i,
            j=particle_start + j,
            ke=float(ke[idx]),
            kd=float(kd[idx]),
            control=0.0,
        )
        builder.spring_rest_length[-1] = float(rest[idx])

    render_edges = edges_obj[:: max(1, int(render_edge_stride))].copy() + particle_start
    point_radius = np.maximum(radius_obj * float(particle_radius_vis_scale), float(particle_radius_vis_min))
    strict_export = bool(np.asarray(ir.get("strict_phystwin_export", np.array([False]))).ravel()[0])
    spring_ke_mode = str(np.asarray(ir.get("spring_ke_mode", np.array(["unknown"]))).ravel()[0])
    spec = SoftSpec(
        name=name,
        color_rgb=color_rgb,
        ir_path=ir_path.resolve(),
        particle_start=particle_start,
        particle_count=n_obj,
        drag_damping=float(newton_import_ir.ir_scalar(ir, "drag_damping", default=0.0)),
        spring_edge_local=edges_obj,
        render_edge_global=render_edges,
        render_point_radius=point_radius.astype(np.float32),
        mass_sum=float(mass_obj.sum()),
        strict_export=strict_export,
        spring_ke_mode=spring_ke_mode,
        spring_ke_rel_error_max=_spring_formula_rel_error(ir),
        bbox_extent=tuple(np.ptp(pos, axis=0).astype(np.float64).tolist()),
    )
    if spec.strict_export and spec.spring_ke_mode != "y_over_rest":
        raise ValueError(
            f"{name}: expected spring_ke_mode='y_over_rest' for strict export, got {spec.spring_ke_mode!r}"
        )
    return spec


def _float_cluster_xy(count: int) -> list[tuple[float, float]]:
    seeds = [
        (-0.30, 0.00),
        (0.00, 0.22),
        (0.30, 0.00),
        (0.00, -0.22),
        (-0.18, -0.36),
        (0.18, 0.36),
    ]
    if count > len(seeds):
        raise ValueError(f"rigid_count={count} exceeds built-in layout capacity {len(seeds)}")
    return seeds[:count]


def _bunny_quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(float(yaw_deg))
    sz = math.sin(0.5 * yaw)
    cz = math.cos(0.5 * yaw)
    tilt = math.radians(90.0)
    sx = math.sin(0.5 * tilt)
    cx = math.cos(0.5 * tilt)
    qx = sx * cz
    qy = sx * sz
    qz = cx * sz
    qw = cx * cz
    return (qx, qy, qz, qw)


def build_model(
    sloth_ir: dict[str, np.ndarray],
    zebra_ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> tuple[newton.Model, DemoMeta]:
    _validate_ir_pair(sloth_ir, zebra_ir)

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)

    target_bottom_z = float(args.water_level + args.drop_gap)
    object_specs: list[SoftSpec] = []
    particle_start = 0

    object_specs.append(
        _prepare_soft_object(
            builder=builder,
            ir=sloth_ir,
            ir_path=args.sloth_ir,
            name="sloth",
            color_rgb=(0.98, 0.66, 0.18),
            particle_start=particle_start,
            target_xy=tuple(float(v) for v in args.sloth_target_xy),
            target_bottom_z=target_bottom_z,
            render_edge_stride=args.render_edge_stride,
            particle_radius_vis_scale=args.particle_radius_vis_scale,
            particle_radius_vis_min=args.particle_radius_vis_min,
        )
    )
    particle_start += object_specs[-1].particle_count

    object_specs.append(
        _prepare_soft_object(
            builder=builder,
            ir=zebra_ir,
            ir_path=args.zebra_ir,
            name="zebra",
            color_rgb=(0.30, 0.82, 0.96),
            particle_start=particle_start,
            target_xy=tuple(float(v) for v in args.zebra_target_xy),
            target_bottom_z=target_bottom_z,
            render_edge_stride=args.render_edge_stride,
            particle_radius_vis_scale=args.particle_radius_vis_scale,
            particle_radius_vis_min=args.particle_radius_vis_min,
        )
    )
    particle_start += object_specs[-1].particle_count

    basin_cfg = builder.default_shape_cfg.copy()
    basin_cfg.mu = 0.35
    basin_cfg.ke = 1.5e4
    basin_cfg.kd = 6.0e2
    basin_cfg.kf = 3.0e2
    basin_cfg.is_visible = False

    water_extent = float(args.water_extent)
    water_depth = float(args.water_depth)
    wall_thickness = float(args.water_wall_thickness)
    wall_height = float(args.water_wall_height)
    basin_floor_z = float(args.water_level - water_depth - 0.5 * wall_thickness)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, basin_floor_z), wp.quat_identity()),
        hx=water_extent + wall_thickness,
        hy=water_extent + wall_thickness,
        hz=0.5 * wall_thickness,
        cfg=basin_cfg,
        label="water_basin_floor",
    )
    wall_z = float(args.water_level - 0.5 * water_depth + 0.5 * wall_height)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(water_extent + 0.5 * wall_thickness, 0.0, wall_z), wp.quat_identity()),
        hx=0.5 * wall_thickness,
        hy=water_extent + wall_thickness,
        hz=0.5 * wall_height,
        cfg=basin_cfg,
        label="water_wall_pos_x",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(-(water_extent + 0.5 * wall_thickness), 0.0, wall_z), wp.quat_identity()),
        hx=0.5 * wall_thickness,
        hy=water_extent + wall_thickness,
        hz=0.5 * wall_height,
        cfg=basin_cfg,
        label="water_wall_neg_x",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, water_extent + 0.5 * wall_thickness, wall_z), wp.quat_identity()),
        hx=water_extent + wall_thickness,
        hy=0.5 * wall_thickness,
        hz=0.5 * wall_height,
        cfg=basin_cfg,
        label="water_wall_pos_y",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, -(water_extent + 0.5 * wall_thickness), wall_z), wp.quat_identity()),
        hx=water_extent + wall_thickness,
        hy=0.5 * wall_thickness,
        hz=0.5 * wall_height,
        cfg=basin_cfg,
        label="water_wall_neg_y",
    )
    builder.add_shape_plane(
        xform=wp.transform(wp.vec3(0.0, 0.0, float(args.hidden_floor_z)), wp.quat_identity()),
        width=0.0,
        length=0.0,
        body=-1,
        cfg=basin_cfg,
        label="hidden_floor",
    )

    mesh, mesh_verts_local, mesh_tri_indices, mesh_render_edges, mesh_asset_path = load_bunny_mesh(
        args.bunny_asset,
        args.bunny_prim,
    )
    mesh_scale = float(args.rigid_scale)
    mesh_scaled = mesh_verts_local * mesh_scale
    mesh_extent = np.ptp(mesh_scaled, axis=0)
    bunny_half_height = float(0.5 * mesh_extent[2])
    target_sub = float(max(args.float_submerge_fraction * mesh_extent[2], 0.03))
    target_bottom = float(args.water_level - target_sub)

    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.rigid_body_mu)
    rigid_cfg.ke = float(args.rigid_body_ke)
    rigid_cfg.kd = float(args.rigid_body_kd)
    rigid_cfg.kf = float(args.rigid_body_kf)

    rigid_labels: list[str] = []
    cluster_xy = _float_cluster_xy(int(args.rigid_count))
    yaw_values = [18.0, -34.0, 42.0, 0.0, -18.0, 31.0]
    for idx, (px, py) in enumerate(cluster_xy):
        qx, qy, qz, qw = _bunny_quat_xyzw(yaw_values[idx])
        q = wp.quat(qx, qy, qz, qw)
        rotated = (mesh_scaled @ quat_to_rotmat((qx, qy, qz, qw)).T).astype(np.float32)
        z_min = float(rotated[:, 2].min())
        z_max = float(rotated[:, 2].max())
        pos_z = float(target_bottom - z_min)
        label = f"water_bunny_{idx}"
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(px), float(py), pos_z), q),
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
        builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            scale=(mesh_scale, mesh_scale, mesh_scale),
            cfg=rigid_cfg,
        )
        rigid_labels.append(label)

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    model.particle_mu = float(
        np.mean(
            [
                float(newton_import_ir.ir_scalar(sloth_ir, "contact_collide_fric", default=0.5)),
                float(newton_import_ir.ir_scalar(zebra_ir, "contact_collide_fric", default=0.5)),
            ]
        )
    )

    cluster_center = np.array(
        [np.mean([xy[0] for xy in cluster_xy]), np.mean([xy[1] for xy in cluster_xy]), args.water_level],
        dtype=np.float32,
    )

    meta = DemoMeta(
        mesh_verts_local=mesh_verts_local.astype(np.float32),
        mesh_tri_indices=mesh_tri_indices.astype(np.int32),
        mesh_render_edges=mesh_render_edges.astype(np.int32),
        mesh_scale=mesh_scale,
        mesh_asset_path=mesh_asset_path,
        object_specs=object_specs,
        rigid_labels=rigid_labels,
        rigid_cluster_center=cluster_center,
        water_level=float(args.water_level),
        hidden_floor_z=float(args.hidden_floor_z),
    )
    return model, meta


def _spawn_mpm_particle_block(
    builder: newton.ModelBuilder,
    *,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    voxel_size: float,
    particles_per_cell: int,
    density: float,
    flags: int,
) -> np.ndarray:
    res = np.array(
        np.ceil(max(1, int(particles_per_cell)) * (bounds_hi - bounds_lo) / max(voxel_size, 1.0e-4)),
        dtype=int,
    )
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.5)
    mass = float(cell_volume * density)
    begin_id = len(builder.particle_q)
    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1,
        dim_y=int(res[1]) + 1,
        dim_z=int(res[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=1.25 * radius,
        radius_mean=radius,
        flags=flags,
    )
    end_id = len(builder.particle_q)
    return np.arange(begin_id, end_id, dtype=np.int32)


def build_water_system(model: newton.Model, args: argparse.Namespace, device: str) -> WaterSystem:
    water_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(water_builder)

    extent = float(args.water_extent)
    depth = float(args.water_depth)
    lo = np.array([-extent, -extent, float(args.water_level - depth)], dtype=np.float32)
    hi = np.array([extent, extent, float(args.water_level)], dtype=np.float32)
    water_ids = _spawn_mpm_particle_block(
        water_builder,
        bounds_lo=lo,
        bounds_hi=hi,
        voxel_size=float(args.water_voxel_size),
        particles_per_cell=int(args.water_particles_per_cell),
        density=float(args.water_density),
        flags=int(newton.ParticleFlags.ACTIVE),
    )

    water_model = water_builder.finalize(device=device)
    water_model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))

    water_sel = wp.array(water_ids, dtype=int, device=device)
    water_model.mpm.young_modulus[water_sel].fill_(float(args.water_young_modulus))
    water_model.mpm.poisson_ratio[water_sel].fill_(float(args.water_poisson_ratio))
    water_model.mpm.damping[water_sel].fill_(float(args.water_damping))
    water_model.mpm.hardening[water_sel].fill_(float(args.water_hardening))
    water_model.mpm.friction[water_sel].fill_(float(args.water_friction))
    water_model.mpm.yield_pressure[water_sel].fill_(float(args.water_yield_pressure))
    water_model.mpm.tensile_yield_ratio[water_sel].fill_(float(args.water_tensile_yield_ratio))
    water_model.mpm.yield_stress[water_sel].fill_(float(args.water_yield_stress))

    mpm_options = newton.solvers.SolverImplicitMPM.Options()
    mpm_options.voxel_size = float(args.water_voxel_size)
    mpm_options.tolerance = 1.0e-5
    mpm_options.transfer_scheme = "pic"
    mpm_options.grid_type = str(args.water_grid_type)
    mpm_options.grid_padding = 50 if args.water_grid_type == "fixed" else 0
    mpm_options.max_active_cell_count = (1 << 16) if args.water_grid_type == "fixed" else -1
    mpm_options.strain_basis = "P0"
    mpm_options.max_iterations = int(args.water_max_iterations)
    mpm_options.critical_fraction = 0.0
    mpm_options.air_drag = float(args.water_air_drag)
    mpm_options.collider_velocity_mode = "finite_difference"

    water_solver = newton.solvers.SolverImplicitMPM(water_model, mpm_options)
    water_solver.setup_collider(model=model)

    combined_state_template = model.state()
    water_state_0 = water_model.state()
    water_state_1 = water_model.state()
    water_state_0.body_q = wp.empty_like(combined_state_template.body_q)
    water_state_0.body_qd = wp.empty_like(combined_state_template.body_qd)
    water_state_0.body_f = wp.empty_like(combined_state_template.body_f)
    water_state_1.body_q = wp.empty_like(combined_state_template.body_q)
    water_state_1.body_qd = wp.empty_like(combined_state_template.body_qd)
    water_state_1.body_f = wp.empty_like(combined_state_template.body_f)

    grains = water_solver.sample_render_grains(water_state_0, int(args.water_points_per_particle))
    grain_radius = float(args.water_voxel_size) / max(3.0 * float(args.water_points_per_particle), 1.0)
    grain_radii = wp.full(grains.size, value=grain_radius, dtype=float, device=device)
    grain_colors = wp.full(grains.size, value=wp.vec3(0.34, 0.84, 0.98), dtype=wp.vec3, device=device)

    max_nodes = 1 << 20
    collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
    collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
    collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=device)
    collider_body_id = water_solver.collider_body_index
    body_water_forces = wp.zeros_like(combined_state_template.body_f)

    frame_dt = float(args.sim_dt) * float(args.substeps) if args.sim_dt is not None and args.substeps is not None else 0.0
    if frame_dt <= 0.0:
        frame_dt = 0.0

    return WaterSystem(
        model=water_model,
        solver=water_solver,
        state_0=water_state_0,
        state_1=water_state_1,
        render_points=grains,
        render_radii=grain_radii,
        render_colors=grain_colors,
        collider_impulses=collider_impulses,
        collider_impulse_pos=collider_impulse_pos,
        collider_impulse_ids=collider_impulse_ids,
        collider_body_id=collider_body_id,
        body_water_forces=body_water_forces,
        frame_dt=frame_dt,
        active_impulse_count=0,
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


def simulate(
    model: newton.Model,
    meta: DemoMeta,
    args: argparse.Namespace,
    sloth_ir: dict[str, np.ndarray],
    water: WaterSystem,
    device: str,
) -> dict[str, Any]:
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

    sim_dt = float(args.sim_dt) if args.sim_dt is not None else float(newton_import_ir.ir_scalar(sloth_ir, "sim_dt"))
    substeps = int(args.substeps) if args.substeps is not None else int(newton_import_ir.ir_scalar(sloth_ir, "sim_substeps"))
    substeps = max(1, substeps)
    frames = max(2, int(args.frames))

    q_hist: list[np.ndarray] = []
    body_q_hist: list[np.ndarray] = []
    body_qd_hist: list[np.ndarray] = []
    water_points_hist: list[np.ndarray] = []

    frame_dt = float(sim_dt) * float(substeps)
    water.frame_dt = frame_dt

    t0 = time.perf_counter()
    for frame in range(frames):
        q_hist.append(state_in.particle_q.numpy().astype(np.float32))
        body_q_hist.append(state_in.body_q.numpy().astype(np.float32))
        body_qd_hist.append(state_in.body_qd.numpy().astype(np.float32))
        water_points_hist.append(water.render_points.numpy().reshape(-1, 3).astype(np.float32))

        for _ in range(substeps):
            state_in.clear_forces()
            if water.active_impulse_count > 0:
                wp.launch(
                    _compute_body_forces_from_water,
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
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            for spec in meta.object_specs:
                if spec.drag_damping > 0.0:
                    wp.launch(
                        _apply_drag_range,
                        dim=spec.particle_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            spec.particle_start,
                            spec.particle_count,
                            sim_dt,
                            spec.drag_damping,
                        ],
                        device=device,
                    )

        water.body_water_forces.assign(state_in.body_f)
        wp.launch(
            _subtract_body_force_for_water_step,
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
        water.solver.update_particle_frames(water.state_0, water.state_1, frame_dt)
        water.solver.update_render_grains(water.state_0, water.state_1, water.render_points, frame_dt)
        water.state_0, water.state_1 = water.state_1, water.state_0
        _collect_water_impulses(water)

        if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == frames:
            print(f"  frame {frame + 1}/{frames}", flush=True)

    wall = time.perf_counter() - t0
    return {
        "particle_q_all": np.stack(q_hist),
        "body_q": np.stack(body_q_hist),
        "body_qd": np.stack(body_qd_hist),
        "water_points_all": np.stack(water_points_hist),
        "water_point_radii": water.render_radii.numpy().astype(np.float32),
        "water_point_colors": water.render_colors.numpy().astype(np.float32),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time_sec": float(wall),
    }

def _view_specs() -> list[ViewSpec]:
    return [
        ViewSpec("wide", "Establishing Wide", yaw_deg=-34.0, pitch_deg=-11.0, distance=3.9),
        ViewSpec("close_a", "Close View A", yaw_deg=-18.0, pitch_deg=-8.0, distance=3.1),
        ViewSpec("close_b", "Close View B", yaw_deg=26.0, pitch_deg=-7.0, distance=3.0),
        ViewSpec("close_c", "Close View C", yaw_deg=68.0, pitch_deg=-6.0, distance=3.15),
    ]


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

    import newton.viewer

    width = int(args.screen_width)
    height = int(args.screen_height)
    slowdown = max(float(args.slowdown), 1.0e-6)
    fps_out = float(args.render_fps) / slowdown
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
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

    water_points_wp = wp.empty(sim_data["water_points_all"].shape[1], dtype=wp.vec3, device=device)
    water_radii_wp = wp.array(sim_data["water_point_radii"].astype(np.float32, copy=False), dtype=wp.float32, device=device)
    water_colors_wp = wp.array(sim_data["water_point_colors"].astype(np.float32, copy=False), dtype=wp.vec3, device=device)
    bunny_mesh_points = wp.array(
        (meta.mesh_verts_local * float(meta.mesh_scale)).astype(np.float32, copy=False),
        dtype=wp.vec3,
        device=device,
    )
    bunny_mesh_indices = wp.array(meta.mesh_tri_indices.reshape(-1).astype(np.int32, copy=False), dtype=wp.int32, device=device)
    bunny_instance_scales = wp.array(
        np.ones((len(meta.rigid_labels), 3), dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    bunny_palette = np.asarray(
        [
            (0.95, 0.45, 0.28),
            (0.96, 0.68, 0.22),
            (0.88, 0.38, 0.52),
            (0.98, 0.80, 0.34),
        ],
        dtype=np.float32,
    )
    bunny_instance_colors = wp.array(
        np.stack([bunny_palette[i % len(bunny_palette)] for i in range(len(meta.rigid_labels))]).astype(np.float32),
        dtype=wp.vec3,
        device=device,
    )
    bunny_instance_materials = wp.array(
        np.tile(np.asarray([[0.34, 0.0, 0.0, 0.0]], dtype=np.float32), (len(meta.rigid_labels), 1)),
        dtype=wp.vec4,
        device=device,
    )

    point_batches = []
    line_batches = []
    for spec in meta.object_specs:
        pts = wp.empty(spec.particle_count, dtype=wp.vec3, device=device)
        radii = wp.array(spec.render_point_radius.astype(np.float32), dtype=wp.float32, device=device)
        colors = wp.array(
            np.tile(np.asarray(spec.color_rgb, dtype=np.float32), (spec.particle_count, 1)),
            dtype=wp.vec3,
            device=device,
        )
        starts = wp.empty(spec.render_edge_global.shape[0], dtype=wp.vec3, device=device)
        ends = wp.empty(spec.render_edge_global.shape[0], dtype=wp.vec3, device=device)
        point_batches.append((spec, pts, radii, colors))
        line_batches.append((spec, starts, ends))

    viewer = newton.viewer.ViewerGL(
        width=width,
        height=height,
        vsync=False,
        headless=bool(args.viewer_headless),
    )
    ffmpeg_proc = None
    try:
        viewer.set_model(model)
        try:
            viewer.camera.fov = float(args.camera_fov)
        except Exception:
            pass

        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.show_visual = False
        viewer.show_static = False
        viewer.show_collision = False
        viewer.show_contacts = False
        viewer.show_ui = False
        viewer.picking_enabled = False

        try:
            viewer.renderer.sky_upper = (0.16, 0.29, 0.44)
            viewer.renderer.sky_lower = (0.02, 0.05, 0.10)
        except Exception:
            pass

        try:
            shape_colors = {}
            palette = [
                (0.95, 0.45, 0.28),
                (0.96, 0.68, 0.22),
                (0.88, 0.38, 0.52),
                (0.98, 0.80, 0.34),
            ]
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "water_bunny_" in name:
                    bunny_id = int(name.rsplit("_", 1)[1])
                    shape_colors[idx] = palette[bunny_id % len(palette)]
            if shape_colors:
                viewer.update_shape_colors(shape_colors)
        except Exception:
            pass

        target = meta.rigid_cluster_center.astype(np.float32).copy()
        target[2] = float(args.water_level + 0.08)
        cpos = _camera_position(target, view_spec.yaw_deg, view_spec.pitch_deg, view_spec.distance)
        viewer.set_camera(
            wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
            float(view_spec.pitch_deg),
            float(view_spec.yaw_deg),
        )

        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
        if state.body_qd is not None:
            state.body_qd.zero_()

        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["particle_q_all"].shape[0])
        for frame_idx in range(n_frames):
            q_all = sim_data["particle_q_all"][frame_idx].astype(np.float32, copy=False)
            bq = sim_data["body_q"][frame_idx].astype(np.float32, copy=False)
            state.particle_q.assign(q_all)
            state.body_q.assign(bq)

            viewer.begin_frame(float(frame_idx) * float(sim_data["sim_dt"]) * float(sim_data["substeps"]))
            viewer.log_state(state)
            sim_t = float(frame_idx) * float(sim_data["sim_dt"]) * float(sim_data["substeps"])
            water_points_wp.assign(sim_data["water_points_all"][frame_idx].astype(np.float32, copy=False))
            viewer.log_points(
                "/demo/water_particles",
                water_points_wp,
                water_radii_wp,
                water_colors_wp,
                hidden=False,
            )

            viewer.log_mesh(
                "/demo/bunny_mesh",
                bunny_mesh_points,
                bunny_mesh_indices,
                hidden=True,
                backface_culling=True,
            )
            viewer.log_instances(
                "/demo/bunny_instances",
                "/demo/bunny_mesh",
                state.body_q,
                bunny_instance_scales,
                bunny_instance_colors,
                bunny_instance_materials,
                hidden=False,
            )
            for spec, pts_wp, radii_wp, colors_wp in point_batches:
                pts_np = q_all[spec.particle_start : spec.particle_start + spec.particle_count]
                pts_wp.assign(pts_np)
                viewer.log_points(
                    f"/demo/{spec.name}_points",
                    pts_wp,
                    radii_wp,
                    colors_wp,
                    hidden=False,
                )

            for spec, starts_wp, ends_wp in line_batches:
                if spec.render_edge_global.size == 0:
                    continue
                starts_np = q_all[spec.render_edge_global[:, 0]]
                ends_np = q_all[spec.render_edge_global[:, 1]]
                starts_wp.assign(starts_np.astype(np.float32, copy=False))
                ends_wp.assign(ends_np.astype(np.float32, copy=False))
                viewer.log_lines(
                    f"/demo/{spec.name}_springs",
                    starts_wp,
                    ends_wp,
                    spec.color_rgb,
                    width=float(args.spring_line_width),
                    hidden=False,
                )

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                frame = _overlay_text_lines_rgb(
                    frame,
                    [
                        view_spec.label,
                        "Scene: sloth + zebra drop onto bunny rigids in MPM water",
                        f"frame {frame_idx + 1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
                    ],
                    font_size=int(args.label_font_size),
                )
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(frame.tobytes())

        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        code = ffmpeg_proc.wait()
        ffmpeg_proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with non-zero status: {code}")
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


def save_scene_npz(out_dir: Path, prefix: str, sim_data: dict[str, Any], meta: DemoMeta) -> Path:
    scene_npz = out_dir / f"{_output_stem(prefix)}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        body_q=sim_data["body_q"],
        body_qd=sim_data["body_qd"],
        water_points_all=sim_data["water_points_all"],
        water_point_radii=sim_data["water_point_radii"],
        water_point_colors=sim_data["water_point_colors"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        wall_time_sec=np.float32(sim_data["wall_time_sec"]),
        water_level=np.float32(meta.water_level),
        hidden_floor_z=np.float32(meta.hidden_floor_z),
        rigid_mesh_vertices_local=meta.mesh_verts_local,
        rigid_mesh_render_edges=meta.mesh_render_edges,
        rigid_mesh_scale=np.float32(meta.mesh_scale),
        rigid_cluster_center=meta.rigid_cluster_center.astype(np.float32),
        object_ranges=np.asarray(
            [[spec.particle_start, spec.particle_count] for spec in meta.object_specs],
            dtype=np.int32,
        ),
    )
    return scene_npz


def load_scene_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "particle_q_all": np.asarray(data["particle_q_all"], dtype=np.float32),
            "body_q": np.asarray(data["body_q"], dtype=np.float32),
            "body_qd": np.asarray(data["body_qd"], dtype=np.float32),
            "water_points_all": np.asarray(data["water_points_all"], dtype=np.float32),
            "water_point_radii": np.asarray(data["water_point_radii"], dtype=np.float32),
            "water_point_colors": np.asarray(data["water_point_colors"], dtype=np.float32),
            "sim_dt": float(np.asarray(data["sim_dt"]).ravel()[0]),
            "substeps": int(np.asarray(data["substeps"]).ravel()[0]),
            "wall_time_sec": float(np.asarray(data["wall_time_sec"]).ravel()[0])
            if "wall_time_sec" in data
            else 0.0,
        }


def build_summary(
    args: argparse.Namespace,
    sim_data: dict[str, Any],
    meta: DemoMeta,
    panel_paths: list[Path],
    composed_path: Path | None,
) -> dict[str, Any]:
    frame_count = int(sim_data["particle_q_all"].shape[0])
    wall_time_sec = float(sim_data.get("wall_time_sec", 0.0))
    if wall_time_sec <= 0.0:
        existing_summary_path = args.out_dir / f"{_output_stem(args.prefix)}_summary.json"
        if existing_summary_path.exists():
            try:
                with existing_summary_path.open("r", encoding="utf-8") as handle:
                    existing_summary = json.load(handle)
                wall_time_sec = float(existing_summary.get("wall_time_sec", 0.0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                wall_time_sec = 0.0

    return {
        "experiment": "sloth_zebra_water_impact_demo",
        "sloth_ir": str(args.sloth_ir.resolve()),
        "zebra_ir": str(args.zebra_ir.resolve()),
        "frames": frame_count,
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_data["sim_dt"]) * float(sim_data["substeps"]),
        "wall_time_sec": wall_time_sec,
        "gravity_mag": float(args.gravity_mag),
        "water_level": float(args.water_level),
        "drop_gap": float(args.drop_gap),
        "hidden_floor_z": float(args.hidden_floor_z),
        "rigid_count": int(args.rigid_count),
        "rigid_mass": float(args.rigid_mass),
        "rigid_scale": float(args.rigid_scale),
        "rigid_shape": str(args.rigid_shape),
        "initial_rigid_submerge_fraction": float(args.float_submerge_fraction),
        "water_extent": float(args.water_extent),
        "water_depth": float(args.water_depth),
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
        "water_soft_direct_coupling": _water_soft_direct_coupling_supported(),
        "water_soft_direct_coupling_reason": (
            "MPM collider setup reads collision shapes/meshes only; imported PhysTwin soft objects "
            "are particle-spring systems without collider shapes in this demo."
        ),
        "mesh_asset_path": meta.mesh_asset_path,
        "views": [view.label for view in _view_specs()],
        "panel_videos": [str(p) for p in panel_paths],
        "composed_video": str(composed_path) if composed_path is not None else None,
        "soft_objects": [
            {
                "name": spec.name,
                "ir_path": str(spec.ir_path),
                "particle_start": int(spec.particle_start),
                "particle_count": int(spec.particle_count),
                "mass_sum": float(spec.mass_sum),
                "drag_damping": float(spec.drag_damping),
                "bbox_extent": [float(v) for v in spec.bbox_extent],
                "strict_phystwin_export": bool(spec.strict_export),
                "spring_ke_mode": spec.spring_ke_mode,
                "spring_ke_rel_error_max": float(spec.spring_ke_rel_error_max),
            }
            for spec in meta.object_specs
        ],
    }


def _render_subprocess_cmd(
    args: argparse.Namespace,
    scene_npz: Path,
    view_name: str,
) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--sloth-ir",
        str(args.sloth_ir.resolve()),
        "--zebra-ir",
        str(args.zebra_ir.resolve()),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--prefix",
        str(args.prefix),
        "--device",
        str(args.device),
        "--render-only-scene",
        str(scene_npz.resolve()),
        "--views",
        str(view_name),
        "--water-level",
        str(args.water_level),
        "--drop-gap",
        str(args.drop_gap),
        "--hidden-floor-z",
        str(args.hidden_floor_z),
        "--gravity-mag",
        str(args.gravity_mag),
        "--sloth-target-xy",
        str(args.sloth_target_xy[0]),
        str(args.sloth_target_xy[1]),
        "--zebra-target-xy",
        str(args.zebra_target_xy[0]),
        str(args.zebra_target_xy[1]),
        "--rigid-shape",
        str(args.rigid_shape),
        "--rigid-count",
        str(args.rigid_count),
        "--rigid-mass",
        str(args.rigid_mass),
        "--rigid-scale",
        str(args.rigid_scale),
        "--rigid-body-mu",
        str(args.rigid_body_mu),
        "--rigid-body-ke",
        str(args.rigid_body_ke),
        "--rigid-body-kd",
        str(args.rigid_body_kd),
        "--rigid-body-kf",
        str(args.rigid_body_kf),
        "--bunny-asset",
        str(args.bunny_asset),
        "--bunny-prim",
        str(args.bunny_prim),
        "--float-submerge-fraction",
        str(args.float_submerge_fraction),
        "--water-extent",
        str(args.water_extent),
        "--water-depth",
        str(args.water_depth),
        "--water-wall-thickness",
        str(args.water_wall_thickness),
        "--water-wall-height",
        str(args.water_wall_height),
        "--water-voxel-size",
        str(args.water_voxel_size),
        "--water-particles-per-cell",
        str(args.water_particles_per_cell),
        "--water-density",
        str(args.water_density),
        "--water-young-modulus",
        str(args.water_young_modulus),
        "--water-poisson-ratio",
        str(args.water_poisson_ratio),
        "--water-damping",
        str(args.water_damping),
        "--water-hardening",
        str(args.water_hardening),
        "--water-friction",
        str(args.water_friction),
        "--water-yield-pressure",
        str(args.water_yield_pressure),
        "--water-tensile-yield-ratio",
        str(args.water_tensile_yield_ratio),
        "--water-yield-stress",
        str(args.water_yield_stress),
        "--water-air-drag",
        str(args.water_air_drag),
        "--water-grid-type",
        str(args.water_grid_type),
        "--water-max-iterations",
        str(args.water_max_iterations),
        "--water-points-per-particle",
        str(args.water_points_per_particle),
        "--angular-damping",
        str(args.angular_damping),
        "--friction-smoothing",
        str(args.friction_smoothing),
        "--spring-line-width",
        str(args.spring_line_width),
        "--render-edge-stride",
        str(args.render_edge_stride),
        "--particle-radius-vis-scale",
        str(args.particle_radius_vis_scale),
        "--particle-radius-vis-min",
        str(args.particle_radius_vis_min),
        "--render-fps",
        str(args.render_fps),
        "--slowdown",
        str(args.slowdown),
        "--screen-width",
        str(args.screen_width),
        "--screen-height",
        str(args.screen_height),
        "--camera-fov",
        str(args.camera_fov),
        "--label-font-size",
        str(args.label_font_size),
    ]
    if args.viewer_headless:
        cmd.append("--viewer-headless")
    else:
        cmd.append("--no-viewer-headless")
    if args.overlay_label:
        cmd.append("--overlay-label")
    else:
        cmd.append("--no-overlay-label")
    if args.enable_tri_contact:
        cmd.append("--enable-tri-contact")
    else:
        cmd.append("--no-enable-tri-contact")
    return cmd


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    sloth_ir = _load_ir(args.sloth_ir)
    zebra_ir = _load_ir(args.zebra_ir)
    view_names = _parse_view_names(args.views)

    print("Building combined model...", flush=True)
    model, meta = build_model(sloth_ir, zebra_ir, args, device)
    if not _water_soft_direct_coupling_supported():
        print(
            "Note: MPM water couples directly to rigid/static collider shapes only; "
            "the imported PhysTwin soft objects do not directly collide with water in this demo.",
            flush=True,
        )

    if args.render_only_scene is not None:
        sim_data = load_scene_npz(args.render_only_scene.resolve())
        scene_npz = args.render_only_scene.resolve()
        print(f"Using existing scene NPZ: {scene_npz}", flush=True)
    else:
        print("Building native MPM water system...", flush=True)
        water = build_water_system(model, args, device)
        print("Simulating impact scene...", flush=True)
        sim_data = simulate(model, meta, args, sloth_ir, water, device)
        scene_npz = save_scene_npz(args.out_dir, args.prefix, sim_data, meta)
        print(f"  Scene NPZ: {scene_npz}", flush=True)

    panel_paths: list[Path] = []
    composed_path: Path | None = None
    if not args.skip_render:
        selected_views = [view for view in _view_specs() if view.name in view_names]
        print(f"Rendering {len(selected_views)} camera view(s)...", flush=True)

        if os.environ.get("PT_DEMO_CHILD_RENDER") == "1" or len(selected_views) == 1:
            for view in selected_views:
                panel_path = args.out_dir / f"{_output_stem(args.prefix)}_{view.name}.mp4"
                render_view_mp4(
                    model=model,
                    sim_data=sim_data,
                    meta=meta,
                    args=args,
                    device=device,
                    view_spec=view,
                    out_mp4=panel_path,
                )
                panel_paths.append(panel_path)
                print(f"  Saved: {panel_path}", flush=True)
        else:
            for view in selected_views:
                panel_path = args.out_dir / f"{_output_stem(args.prefix)}_{view.name}.mp4"
                cmd = _render_subprocess_cmd(args, scene_npz, view.name)
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PT_DEMO_CHILD_RENDER"] = "1"
                subprocess.run(cmd, check=True, env=env)
                panel_paths.append(panel_path)
                print(f"  Saved: {panel_path}", flush=True)

        if [view.name for view in selected_views] == [view.name for view in _view_specs()]:
            composed_path = args.out_dir / f"{_output_stem(args.prefix)}_2x2.mp4"
            _compose_2x2(
                panel_paths,
                [view.label for view in selected_views],
                composed_path,
            )
            print(f"  Composed: {composed_path}", flush=True)

    summary = build_summary(args, sim_data, meta, panel_paths, composed_path)
    summary_path = args.out_dir / f"{_output_stem(args.prefix)}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
