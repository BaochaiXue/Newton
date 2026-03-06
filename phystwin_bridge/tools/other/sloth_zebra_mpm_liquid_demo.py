#!/usr/bin/env python3
"""MPM liquid + PhysTwin soft body demo.

Scene
-----
- A pool of MPM liquid particles (water-like material) sits above a ground plane.
- Several native Newton bunny rigid bodies float inside the liquid pool.
- Two PhysTwin soft bodies (sloth + zebra) are imported from IR and dropped
  from above into the liquid, colliding with the floating rigids.
- MPM liquid interacts with rigid bodies via Newton's two-way MPM-collider coupling.
- Soft bodies interact with rigid bodies via SemiImplicit particle-body contact.
- Four camera views are rendered and composed into a labeled 2x2 MP4.

Architecture
------------
- **rigid_model** (SolverXPBD): ground plane + floating bunny rigids
- **liquid_model** (SolverImplicitMPM): MPM water particles, colliding with rigids
- **soft_model** (SolverSemiImplicit): PhysTwin spring-mass (sloth + zebra), colliding with rigids
- Two-way coupling: MPM impulses → rigid body forces; rigid positions → MPM collider
"""
from __future__ import annotations

import argparse
import importlib.util
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
from newton.solvers import SolverImplicitMPM  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SoftSpec:
    name: str
    color_rgb: tuple[float, float, float]
    ir_path: Path
    particle_start: int
    particle_count: int
    drag_damping: float
    render_edge_global: np.ndarray
    render_point_radius: np.ndarray
    mass_sum: float


@dataclass
class ViewSpec:
    name: str
    label: str
    yaw_deg: float
    pitch_deg: float
    distance: float


# ═══════════════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════════════


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
def _compute_body_forces_from_mpm(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Convert MPM collider impulses to rigid body forces."""
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def _subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Remove previously applied MPM forces from rigid body velocities."""
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MPM liquid + PhysTwin soft body drop demo."
    )
    p.add_argument(
        "--sloth-ir", type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_sloth_normal/"
            "20260305_163731_full_parity/double_lift_sloth_normal_ir.npz"
        ),
    )
    p.add_argument(
        "--zebra-ir", type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_zebra_normal/"
            "20260305_163731_full_parity/double_lift_zebra_normal_ir.npz"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="mpm_liquid_demo")
    p.add_argument("--device", default=path_defaults.default_device())

    # Timing
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)

    # Gravity
    p.add_argument("--gravity-mag", type=float, default=9.8)

    # Liquid (MPM)
    p.add_argument("--liquid-extent-xy", type=float, default=0.8)
    p.add_argument("--liquid-depth", type=float, default=0.35)
    p.add_argument("--liquid-density", type=float, default=1000.0)
    p.add_argument("--liquid-voxel-size", type=float, default=0.06)
    p.add_argument("--liquid-young-modulus", type=float, default=5.0e4,
                    help="Low = more fluid-like.")
    p.add_argument("--liquid-poisson-ratio", type=float, default=0.3)
    p.add_argument("--liquid-yield-stress", type=float, default=50.0,
                    help="Low = flows easily like liquid.")
    p.add_argument("--liquid-yield-pressure", type=float, default=1.0e10)
    p.add_argument("--liquid-friction", type=float, default=0.0,
                    help="Zero friction = inviscid fluid-like.")
    p.add_argument("--liquid-tensile-yield-ratio", type=float, default=1.0)
    p.add_argument("--liquid-hardening", type=float, default=0.0)

    # Soft body (PhysTwin)
    p.add_argument("--drop-gap", type=float, default=0.6,
                    help="Height above liquid surface to drop soft bodies.")
    p.add_argument("--sloth-target-xy", type=float, nargs=2, default=(-0.18, -0.06))
    p.add_argument("--zebra-target-xy", type=float, nargs=2, default=(0.18, 0.06))
    p.add_argument("--render-edge-stride", type=int, default=14)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.2)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.012)

    # Rigid bodies (floating bunnies)
    p.add_argument("--rigid-count", type=int, default=3)
    p.add_argument("--rigid-mass", type=float, default=40.0)
    p.add_argument("--rigid-scale", type=float, default=0.14)
    p.add_argument("--rigid-body-mu", type=float, default=0.5)
    p.add_argument("--rigid-body-ke", type=float, default=2.0e4)
    p.add_argument("--rigid-body-kd", type=float, default=5.0e2)
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")

    # Solver
    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)
    p.add_argument("--mpm-max-iterations", type=int, default=100)
    p.add_argument("--mpm-tolerance", type=float, default=1.0e-5)

    # Rendering
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--screen-width", type=int, default=960)
    p.add_argument("--screen-height", type=int, default=540)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-fov", type=float, default=50.0)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=26)
    p.add_argument("--spring-line-width", type=float, default=0.008)
    p.add_argument(
        "--views", default="wide,close_a,close_b,close_c",
        help="Comma-separated render views to emit.",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    ir = newton_import_ir.load_ir(path.resolve())
    required = [
        "x0", "v0", "mass", "num_object_points",
        "spring_edges", "spring_rest_length", "spring_ke", "spring_kd",
        "drag_damping", "sim_dt", "sim_substeps", "reverse_z",
    ]
    missing = [key for key in required if key not in ir]
    if missing:
        raise KeyError(f"{path} missing IR fields: {missing}")
    return ir


def _validate_ir_pair(a: dict, b: dict) -> None:
    for key in ("sim_dt", "sim_substeps", "reverse_z"):
        va = newton_import_ir.ir_scalar(a, key) if key != "reverse_z" else newton_import_ir.ir_bool(a, key)
        vb = newton_import_ir.ir_scalar(b, key) if key != "reverse_z" else newton_import_ir.ir_bool(b, key)
        if va != vb:
            raise ValueError(f"IR mismatch for {key}: {va!r} vs {vb!r}")


def quat_to_rotmat(q_xyzw):
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1--2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def _bunny_quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(float(yaw_deg))
    sz, cz = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
    tilt = math.radians(90.0)
    sx, cx = math.sin(0.5 * tilt), math.cos(0.5 * tilt)
    return (sx * cz, sx * sz, cx * sz, cx * cz)


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
        raise ValueError(f"Prim not found: {prim_path}")
    mesh = newton.usd.get_mesh(prim)
    return mesh, np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)


def _camera_position(target, yaw_deg, pitch_deg, distance):
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    front = np.array([
        math.cos(yaw) * math.cos(pitch),
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
    ], dtype=np.float32)
    front /= max(float(np.linalg.norm(front)), 1e-6)
    return np.asarray(target, dtype=np.float32) - front * float(distance)


def _view_specs() -> list[ViewSpec]:
    return [
        ViewSpec("wide", "Wide Shot", yaw_deg=-38.0, pitch_deg=-14.0, distance=3.8),
        ViewSpec("close_a", "Close A", yaw_deg=-18.0, pitch_deg=-10.0, distance=2.8),
        ViewSpec("close_b", "Close B", yaw_deg=30.0, pitch_deg=-8.0, distance=2.8),
        ViewSpec("close_c", "Close C", yaw_deg=75.0, pitch_deg=-7.0, distance=2.9),
    ]


def _parse_view_names(value: str) -> list[str]:
    items = [p.strip() for p in value.split(",") if p.strip()]
    known = {s.name for s in _view_specs()}
    unknown = [i for i in items if i not in known]
    if unknown:
        raise ValueError(f"Unknown views: {unknown}")
    return items


def _overlay_text_lines_rgb(frame: np.ndarray, lines: list[str], font_size: int = 26) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame
    if frame.dtype != np.uint8:
        return frame
    img = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    font = None
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(fp, size=max(10, int(font_size)))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    x, y, pad_x, pad_y, gap = 18, 16, 8, 5, 7
    for line in lines:
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle((x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y), fill=(0, 0, 0, 145))
        draw.text((x, y), line, fill=(255, 255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0, 220))
        y += th + gap
    return np.asarray(img, dtype=np.uint8)


def _compose_2x2(inputs: list[Path], labels: list[str], out_mp4: Path) -> None:
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
    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
        str(out_mp4),
    ])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stdout}")


# ═══════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════


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
    mass = np.asarray(ir["mass"], dtype=np.float32).copy()
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    if reverse_z:
        x0[:, 2] *= -1.0

    pos = x0[:n_obj].copy()
    vel = np.zeros_like(pos, dtype=np.float32)
    mass_obj = mass[:n_obj].copy()
    radius_all, _, _ = newton_import_ir.resolve_collision_radius(ir, x0.shape[0])
    radius_obj = radius_all[:n_obj].astype(np.float32, copy=True)

    bbox_min = pos.min(axis=0)
    center_xy = pos[:, :2].mean(axis=0)
    shift = np.array([
        float(target_xy[0]) - float(center_xy[0]),
        float(target_xy[1]) - float(center_xy[1]),
        float(target_bottom_z) - float(bbox_min[2]),
    ], dtype=np.float32)
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
            i=particle_start + i, j=particle_start + j,
            ke=float(ke[idx]), kd=float(kd[idx]), control=0.0,
        )
        builder.spring_rest_length[-1] = float(rest[idx])

    render_edges = edges_obj[::max(1, int(render_edge_stride))].copy() + particle_start
    point_radius = np.maximum(radius_obj * float(particle_radius_vis_scale), float(particle_radius_vis_min))

    return SoftSpec(
        name=name, color_rgb=color_rgb, ir_path=ir_path.resolve(),
        particle_start=particle_start, particle_count=n_obj,
        drag_damping=float(newton_import_ir.ir_scalar(ir, "drag_damping", default=0.0)),
        render_edge_global=render_edges,
        render_point_radius=point_radius.astype(np.float32),
        mass_sum=float(mass_obj.sum()),
    )


def build_rigid_model(args: argparse.Namespace, device: str):
    """Build Newton model with ground plane + floating bunny rigid bodies."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    builder.default_shape_cfg.mu = float(args.rigid_body_mu)

    # Ground plane
    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.mu = 0.5
    ground_cfg.ke = 1.5e4
    ground_cfg.kd = 5.0e2
    builder.add_ground_plane(cfg=ground_cfg)

    # Load bunny mesh
    mesh, verts_local = load_bunny_mesh(args.bunny_asset, args.bunny_prim)
    mesh_scale = float(args.rigid_scale)
    mesh_scaled = verts_local * mesh_scale

    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.rigid_body_mu)
    rigid_cfg.ke = float(args.rigid_body_ke)
    rigid_cfg.kd = float(args.rigid_body_kd)

    rigid_labels = []
    cluster_xy = [(-0.22, 0.00), (0.22, 0.00), (0.00, 0.20)][:int(args.rigid_count)]
    yaw_values = [18.0, -34.0, 42.0, 0.0]
    mesh_extent = np.ptp(mesh_scaled, axis=0)

    for idx, (px, py) in enumerate(cluster_xy):
        qx, qy, qz, qw = _bunny_quat_xyzw(yaw_values[idx % len(yaw_values)])
        q = wp.quat(qx, qy, qz, qw)
        rotated = (mesh_scaled @ quat_to_rotmat((qx, qy, qz, qw)).T).astype(np.float32)
        z_min = float(rotated[:, 2].min())
        # Place bunny partially submerged: bottom at ~60% depth in liquid
        pos_z = float(args.liquid_depth * 0.4 - z_min)
        label = f"bunny_{idx}"
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(px), float(py), pos_z), q),
            mass=float(args.rigid_mass),
            inertia=wp.mat33(
                float(args.rigid_mass * (mesh_extent[1]**2 + mesh_extent[2]**2) / 12.0), 0.0, 0.0,
                0.0, float(args.rigid_mass * (mesh_extent[0]**2 + mesh_extent[2]**2) / 12.0), 0.0,
                0.0, 0.0, float(args.rigid_mass * (mesh_extent[0]**2 + mesh_extent[1]**2) / 12.0),
            ),
            lock_inertia=True,
            label=label,
        )
        builder.add_shape_mesh(body=body, mesh=mesh, scale=(mesh_scale, mesh_scale, mesh_scale), cfg=rigid_cfg)
        rigid_labels.append(label)

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    return model, rigid_labels, mesh, verts_local


def build_soft_model(
    sloth_ir: dict, zebra_ir: dict, args: argparse.Namespace, device: str,
) -> tuple[newton.Model, list[SoftSpec]]:
    """Build Newton model for PhysTwin spring-mass soft bodies."""
    _validate_ir_pair(sloth_ir, zebra_ir)
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)

    target_bottom_z = float(args.liquid_depth + args.drop_gap)
    specs: list[SoftSpec] = []
    particle_start = 0

    specs.append(_prepare_soft_object(
        builder=builder, ir=sloth_ir, ir_path=args.sloth_ir,
        name="sloth", color_rgb=(0.98, 0.66, 0.18),
        particle_start=particle_start,
        target_xy=tuple(float(v) for v in args.sloth_target_xy),
        target_bottom_z=target_bottom_z,
        render_edge_stride=args.render_edge_stride,
        particle_radius_vis_scale=args.particle_radius_vis_scale,
        particle_radius_vis_min=args.particle_radius_vis_min,
    ))
    particle_start += specs[-1].particle_count

    specs.append(_prepare_soft_object(
        builder=builder, ir=zebra_ir, ir_path=args.zebra_ir,
        name="zebra", color_rgb=(0.30, 0.82, 0.96),
        particle_start=particle_start,
        target_xy=tuple(float(v) for v in args.zebra_target_xy),
        target_bottom_z=target_bottom_z,
        render_edge_stride=args.render_edge_stride,
        particle_radius_vis_scale=args.particle_radius_vis_scale,
        particle_radius_vis_min=args.particle_radius_vis_min,
    ))
    particle_start += specs[-1].particle_count

    # Add rigid body shapes as collision targets for soft particles
    # (ground plane + bunny meshes copied from rigid model)
    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.mu = 0.5
    ground_cfg.ke = 1.5e4
    ground_cfg.kd = 5.0e2
    builder.add_ground_plane(cfg=ground_cfg)

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    return model, specs


def build_liquid_model(args: argparse.Namespace, device: str) -> newton.Model:
    """Build the MPM liquid particle model."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)

    voxel_size = float(args.liquid_voxel_size)
    density = float(args.liquid_density)
    particles_per_cell = 3.0
    extent = float(args.liquid_extent_xy)
    depth = float(args.liquid_depth)

    lo = np.array([-extent, -extent, 0.0])
    hi = np.array([extent, extent, depth])
    res = np.array(np.ceil(particles_per_cell * (hi - lo) / voxel_size), dtype=int)
    cell_size = (hi - lo) / res
    cell_volume = np.prod(cell_size)
    radius = float(np.max(cell_size) * 0.5)
    mass = float(cell_volume * density)

    builder.add_particle_grid(
        pos=wp.vec3(lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
        custom_attributes={
            "mpm:friction": float(args.liquid_friction),
        },
    )

    # Add ground plane for liquid containment
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))

    # Set liquid material properties
    liquid_indices = wp.array(np.arange(model.particle_count, dtype=int), dtype=int, device=device)
    model.mpm.young_modulus[liquid_indices].fill_(float(args.liquid_young_modulus))
    model.mpm.poisson_ratio[liquid_indices].fill_(float(args.liquid_poisson_ratio))
    model.mpm.yield_stress[liquid_indices].fill_(float(args.liquid_yield_stress))
    model.mpm.yield_pressure[liquid_indices].fill_(float(args.liquid_yield_pressure))
    model.mpm.tensile_yield_ratio[liquid_indices].fill_(float(args.liquid_tensile_yield_ratio))
    model.mpm.hardening[liquid_indices].fill_(float(args.liquid_hardening))
    model.mpm.friction[liquid_indices].fill_(float(args.liquid_friction))

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════


def simulate(
    rigid_model: newton.Model,
    soft_model: newton.Model,
    liquid_model: newton.Model,
    soft_specs: list[SoftSpec],
    mpm_solver: SolverImplicitMPM,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    # Rigid solver (XPBD)
    rigid_solver = newton.solvers.SolverXPBD(rigid_model)
    rigid_contacts = rigid_model.contacts()
    rigid_control = rigid_model.control()
    rigid_state_0 = rigid_model.state()
    rigid_state_1 = rigid_model.state()

    # Soft solver (SemiImplicit)
    soft_solver = newton.solvers.SolverSemiImplicit(
        soft_model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
    )
    soft_contacts = soft_model.contacts()
    soft_control = soft_model.control()
    soft_state_0 = soft_model.state()
    soft_state_1 = soft_model.state()

    # Liquid state
    liquid_state_0 = liquid_model.state()
    # Share body state arrays with rigid model for collider coupling
    liquid_state_0.body_q = wp.empty_like(rigid_state_0.body_q)
    liquid_state_0.body_qd = wp.empty_like(rigid_state_0.body_qd)
    liquid_state_0.body_f = wp.empty_like(rigid_state_0.body_f)

    frame_dt = 1.0 / float(args.fps)
    substeps = max(1, int(args.substeps))
    sim_dt = frame_dt / substeps
    frames = max(2, int(args.frames))

    # MPM collider impulse buffers
    max_nodes = 1 << 18
    collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
    collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
    collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=device)
    collider_body_id = mpm_solver.collider_body_index
    body_mpm_forces = wp.zeros_like(rigid_state_0.body_f)

    # Collect initial impulses (all zero)
    _collect_mpm_impulses(mpm_solver, liquid_state_0, collider_impulses, collider_impulse_pos, collider_impulse_ids)

    # History
    soft_q_hist: list[np.ndarray] = []
    rigid_q_hist: list[np.ndarray] = []
    liquid_q_hist: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in range(frames):
        soft_q_hist.append(soft_state_0.particle_q.numpy().astype(np.float32))
        rigid_q_hist.append(rigid_state_0.body_q.numpy().astype(np.float32))
        liquid_q_hist.append(liquid_state_0.particle_q.numpy().astype(np.float32))

        for _ in range(substeps):
            # 1) Apply MPM impulses as forces on rigid bodies
            rigid_state_0.clear_forces()
            wp.launch(
                _compute_body_forces_from_mpm,
                dim=collider_impulse_ids.shape[0],
                inputs=[
                    frame_dt,
                    collider_impulse_ids, collider_impulses, collider_impulse_pos,
                    collider_body_id,
                    rigid_state_0.body_q, rigid_model.body_com, rigid_state_0.body_f,
                ],
                device=device,
            )
            body_mpm_forces.assign(rigid_state_0.body_f)

            # 2) Step rigid bodies
            rigid_model.collide(rigid_state_0, rigid_contacts)
            rigid_solver.step(rigid_state_0, rigid_state_1, rigid_control, rigid_contacts, sim_dt)
            rigid_state_0, rigid_state_1 = rigid_state_1, rigid_state_0

            # 3) Step soft bodies
            soft_state_0.clear_forces()
            soft_model.collide(soft_state_0, soft_contacts)
            soft_solver.step(soft_state_0, soft_state_1, soft_control, soft_contacts, sim_dt)
            soft_state_0, soft_state_1 = soft_state_1, soft_state_0

            # Apply drag on soft bodies
            for spec in soft_specs:
                if spec.drag_damping > 0.0:
                    wp.launch(
                        _apply_drag_range,
                        dim=spec.particle_count,
                        inputs=[
                            soft_state_0.particle_q, soft_state_0.particle_qd,
                            spec.particle_start, spec.particle_count,
                            sim_dt, spec.drag_damping,
                        ],
                        device=device,
                    )

        # 4) Step MPM liquid (once per frame, as in the twoway coupling example)
        # Subtract previously applied impulses from body velocities
        if liquid_state_0.body_q is not None:
            wp.launch(
                _subtract_body_force,
                dim=liquid_state_0.body_q.shape,
                inputs=[
                    frame_dt,
                    rigid_state_0.body_q, rigid_state_0.body_qd, body_mpm_forces,
                    rigid_model.body_inv_inertia, rigid_model.body_inv_mass,
                    liquid_state_0.body_q, liquid_state_0.body_qd,
                ],
                device=device,
            )
        mpm_solver.step(liquid_state_0, liquid_state_0, contacts=None, control=None, dt=frame_dt)
        _collect_mpm_impulses(mpm_solver, liquid_state_0, collider_impulses, collider_impulse_pos, collider_impulse_ids)

        if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == frames:
            print(f"  frame {frame + 1}/{frames}", flush=True)

    wall = time.perf_counter() - t0
    print(f"Simulation done: {frames} frames in {wall:.1f}s", flush=True)

    return {
        "soft_q_all": np.stack(soft_q_hist),
        "rigid_q": np.stack(rigid_q_hist),
        "liquid_q_all": np.stack(liquid_q_hist),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "frame_dt": float(frame_dt),
        "wall_time_sec": float(wall),
    }


def _collect_mpm_impulses(mpm_solver, liquid_state, impulses, pos, ids):
    coll_imp, coll_pos, coll_ids = mpm_solver.collect_collider_impulses(liquid_state)
    ids.fill_(-1)
    n = min(coll_imp.shape[0], impulses.shape[0])
    impulses[:n].assign(coll_imp[:n])
    pos[:n].assign(coll_pos[:n])
    ids[:n].assign(coll_ids[:n])


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════


def render_view_mp4(
    *,
    rigid_model: newton.Model,
    sim_data: dict[str, Any],
    soft_specs: list[SoftSpec],
    liquid_model: newton.Model,
    args: argparse.Namespace,
    device: str,
    view_spec: ViewSpec,
    out_mp4: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    import newton.viewer

    width, height = int(args.screen_width), int(args.screen_height)
    slowdown = max(float(args.slowdown), 1e-6)
    fps_out = float(args.render_fps) / slowdown
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps_out:.6f}", "-i", "-", "-an",
        "-vcodec", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", str(out_mp4),
    ]

    viewer = newton.viewer.ViewerGL(
        width=width, height=height, vsync=False, headless=bool(args.viewer_headless),
    )
    ffmpeg_proc = None
    try:
        viewer.set_model(rigid_model)
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
            viewer.renderer.sky_upper = (0.14, 0.26, 0.42)
            viewer.renderer.sky_lower = (0.02, 0.05, 0.10)
        except Exception:
            pass

        # Camera
        target = np.array([0.0, 0.0, float(args.liquid_depth) * 0.5], dtype=np.float32)
        cpos = _camera_position(target, view_spec.yaw_deg, view_spec.pitch_deg, view_spec.distance)
        viewer.set_camera(
            wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
            float(view_spec.pitch_deg), float(view_spec.yaw_deg),
        )

        # Prepare soft body render buffers
        soft_render = []
        for spec in soft_specs:
            pts = wp.empty(spec.particle_count, dtype=wp.vec3, device=device)
            radii = wp.array(spec.render_point_radius, dtype=wp.float32, device=device)
            colors = wp.array(
                np.tile(np.asarray(spec.color_rgb, dtype=np.float32), (spec.particle_count, 1)),
                dtype=wp.vec3, device=device,
            )
            starts = wp.empty(spec.render_edge_global.shape[0], dtype=wp.vec3, device=device)
            ends = wp.empty(spec.render_edge_global.shape[0], dtype=wp.vec3, device=device)
            soft_render.append((spec, pts, radii, colors, starts, ends))

        # Liquid render colors
        n_liquid = liquid_model.particle_count
        liquid_colors = wp.array(
            np.tile(np.asarray([0.15, 0.55, 0.90], dtype=np.float32), (n_liquid, 1)),
            dtype=wp.vec3, device=device,
        )
        liquid_pts = wp.empty(n_liquid, dtype=wp.vec3, device=device)

        state = rigid_model.state()
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["soft_q_all"].shape[0])

        for fi in range(n_frames):
            state.body_q.assign(sim_data["rigid_q"][fi].astype(np.float32, copy=False))

            viewer.begin_frame(float(fi) * float(sim_data["frame_dt"]))
            viewer.log_state(state)

            # Liquid particles
            lq = sim_data["liquid_q_all"][fi].astype(np.float32, copy=False)
            liquid_pts.assign(lq)
            viewer.log_points(
                "/demo/liquid", liquid_pts, liquid_model.particle_radius, liquid_colors, hidden=False,
            )

            # Soft body particles and springs
            sq = sim_data["soft_q_all"][fi].astype(np.float32, copy=False)
            for spec, pts_wp, radii_wp, colors_wp, starts_wp, ends_wp in soft_render:
                pts_np = sq[spec.particle_start:spec.particle_start + spec.particle_count]
                pts_wp.assign(pts_np)
                viewer.log_points(
                    f"/demo/{spec.name}_pts", pts_wp, radii_wp, colors_wp, hidden=False,
                )
                if spec.render_edge_global.size > 0:
                    starts_wp.assign(sq[spec.render_edge_global[:, 0]].astype(np.float32, copy=False))
                    ends_wp.assign(sq[spec.render_edge_global[:, 1]].astype(np.float32, copy=False))
                    viewer.log_lines(
                        f"/demo/{spec.name}_springs", starts_wp, ends_wp,
                        spec.color_rgb, width=float(args.spring_line_width), hidden=False,
                    )

            viewer.end_frame()
            frame_pixels = viewer.get_frame(render_ui=False).numpy()
            if args.overlay_label:
                sim_t = float(fi) * float(sim_data["frame_dt"])
                frame_pixels = _overlay_text_lines_rgb(frame_pixels, [
                    view_spec.label,
                    "MPM Liquid + PhysTwin Soft Bodies",
                    f"frame {fi+1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
                ], font_size=int(args.label_font_size))
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(frame_pixels.tobytes())

        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        code = ffmpeg_proc.wait()
        ffmpeg_proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code}")
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    sloth_ir = _load_ir(args.sloth_ir)
    zebra_ir = _load_ir(args.zebra_ir)
    view_names = _parse_view_names(args.views)

    print("Building rigid model...", flush=True)
    rigid_model, rigid_labels, bunny_mesh, bunny_verts = build_rigid_model(args, device)

    print("Building soft model...", flush=True)
    soft_model, soft_specs = build_soft_model(sloth_ir, zebra_ir, args, device)

    print("Building MPM liquid model...", flush=True)
    liquid_model = build_liquid_model(args, device)
    print(f"  Liquid particle count: {liquid_model.particle_count}", flush=True)

    # Setup MPM solver with rigid model as collider
    mpm_options = SolverImplicitMPM.Options()
    mpm_options.voxel_size = float(args.liquid_voxel_size)
    mpm_options.tolerance = float(args.mpm_tolerance)
    mpm_options.max_iterations = int(args.mpm_max_iterations)
    mpm_options.grid_type = "sparse"
    mpm_options.transfer_scheme = "apic"
    mpm_options.strain_basis = "P0"
    mpm_solver = SolverImplicitMPM(liquid_model, mpm_options)
    mpm_solver.setup_collider(model=rigid_model)

    print("Simulating...", flush=True)
    sim_data = simulate(rigid_model, soft_model, liquid_model, soft_specs, mpm_solver, args, device)

    # Render
    if not args.skip_render:
        selected_views = [v for v in _view_specs() if v.name in view_names]
        panel_paths: list[Path] = []
        print(f"Rendering {len(selected_views)} view(s)...", flush=True)

        for view in selected_views:
            panel_path = args.out_dir / f"{args.prefix}_{view.name}.mp4"
            render_view_mp4(
                rigid_model=rigid_model, sim_data=sim_data,
                soft_specs=soft_specs, liquid_model=liquid_model,
                args=args, device=device, view_spec=view, out_mp4=panel_path,
            )
            panel_paths.append(panel_path)
            print(f"  Saved: {panel_path}", flush=True)

        if len(selected_views) == 4:
            composed = args.out_dir / f"{args.prefix}_2x2.mp4"
            _compose_2x2(panel_paths, [v.label for v in selected_views], composed)
            print(f"  Composed: {composed}", flush=True)

    # Summary
    summary = {
        "experiment": "mpm_liquid_demo",
        "frames": int(args.frames),
        "frame_dt": float(sim_data["frame_dt"]),
        "wall_time_sec": float(sim_data["wall_time_sec"]),
        "liquid_particle_count": int(liquid_model.particle_count),
        "rigid_count": int(args.rigid_count),
        "soft_objects": [
            {"name": s.name, "particle_count": s.particle_count, "mass_sum": s.mass_sum}
            for s in soft_specs
        ],
    }
    summary_path = args.out_dir / f"{args.prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
