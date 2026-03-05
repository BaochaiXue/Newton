#!/usr/bin/env python3
"""Rope-Bunny drop experiment for PhysTwin->Newton bridge.

Scenario
--------
A PhysTwin-learned rope (spring-mass) is placed 5 m above a Stanford Bunny
mesh sitting on a table.  The rope falls under gravity and collides with the
bunny.  The experiment runs inside Newton's SemiImplicit solver.

Requirements addressed
----------------------
1. Fixed camera (no rotation/tracking during playback).
2. Rigid object = Stanford Bunny mesh (canonical Newton example asset).
3. Table (ground plane) present.
4. Rope drops from 5 m with zero initial velocity.
5. Slow-motion rendering with labeled slowdown factor.
6. Correct visual orientation and Newton built-in rendering support.

Usage
-----
    /home/xinjie/miniconda3/bin/python rope_bunny_drop_experiment.py \
        --ir <path_to_rope_ir.npz> \
        --out-dir <output_directory> \
        --rigid-mass 5.0          # A-run
    /home/xinjie/miniconda3/bin/python rope_bunny_drop_experiment.py \
        --ir <path_to_rope_ir.npz> \
        --out-dir <output_directory> \
        --rigid-mass 500.0        # B-run
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
from typing import Any
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bridge core imports
# ---------------------------------------------------------------------------
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

import warp as wp  # noqa: E402
import newton  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rope drops from 5 m onto a Stanford Bunny sitting on a table."
    )
    p.add_argument("--ir", type=Path, required=True, help="Path to rope IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="rope_bunny_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    # Timing
    p.add_argument("--frames", type=int, default=600,
                    help="Total number of rendered frames.")
    p.add_argument("--sim-dt", type=float, default=None,
                    help="Override sim dt (default: from IR).")
    p.add_argument("--substeps", type=int, default=None,
                    help="Override substeps per frame (default: from IR).")

    # Rope
    p.add_argument("--drop-height", type=float, default=5.0,
                    help="Height (metres) above the table from which the rope drops.")
    p.add_argument("--object-mass", type=float, default=1.0,
                    help="Per-particle mass for the rope object.")
    p.add_argument("--drop-controller-springs", action=argparse.BooleanOptionalAction,
                    default=True)
    p.add_argument("--freeze-controllers", action=argparse.BooleanOptionalAction,
                    default=True)
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply PhysTwin drag damping correction in Newton loop.")
    p.add_argument("--drag-damping-scale", type=float, default=1.0,
                    help="Scale factor for drag_damping from IR.")

    # Bunny
    p.add_argument("--rigid-mass", type=float, default=5.0)
    p.add_argument("--bunny-scale", type=float, default=0.12,
                    help="Scale factor for the bunny mesh.")
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")
    p.add_argument(
        "--bunny-quat-xyzw",
        type=float,
        nargs=4,
        default=(0.70710678, 0.0, 0.0, 0.70710678),
        metavar=("X", "Y", "Z", "W"),
        help="Bunny world orientation as quaternion (xyzw).",
    )

    # Contact params
    p.add_argument("--body-mu", type=float, default=0.5)
    p.add_argument("--body-ke", type=float, default=5e4)
    p.add_argument("--body-kd", type=float, default=5e2)
    p.add_argument("--controller-radius", type=float, default=1e-5)

    # Solver
    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)

    # Rendering
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=2.0,
                    help="Slow-motion factor (e.g. 2 = 2x slower).")
    p.add_argument("--render-backend", choices=["newton_gl", "matplotlib"],
                    default="newton_gl",
                    help="Renderer backend. 'newton_gl' uses Newton built-in ViewerGL.")
    p.add_argument("--screen-width", type=int, default=1920,
                    help="Output width (px) for Newton ViewerGL rendering.")
    p.add_argument("--screen-height", type=int, default=1080,
                    help="Output height (px) for Newton ViewerGL rendering.")
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True,
                    help="Use headless mode for Newton ViewerGL.")
    p.add_argument("--camera-pos", type=float, nargs=3, default=(4.5, -6.0, 2.6),
                    metavar=("X", "Y", "Z"),
                    help="Fixed camera world position for Newton ViewerGL.")
    p.add_argument("--camera-pitch", type=float, default=-8.0,
                    help="Fixed camera pitch (deg) for Newton ViewerGL.")
    p.add_argument("--camera-yaw", type=float, default=-42.0,
                    help="Fixed camera yaw (deg) for Newton ViewerGL.")
    p.add_argument("--camera-distance-scale", type=float, default=1.2,
                    help="Distance multiplier for fixed-camera auto fit from trajectory bounds.")
    p.add_argument("--camera-fov", type=float, default=55.0,
                    help="Viewer camera FOV (degrees).")
    p.add_argument("--auto-frame-camera", action=argparse.BooleanOptionalAction, default=False,
                    help="Frame camera once from scene bounds before rendering.")
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5,
                    help="Visualization-only multiplier for particle radii in ViewerGL.")
    p.add_argument("--particle-radius-vis-min", type=float, default=0.02,
                    help="Visualization-only min particle radius in ViewerGL.")
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True,
                    help="Overlay slow-motion label text on output video.")
    p.add_argument("--label-font-size", type=int, default=28,
                    help="Overlay label font size in pixels.")
    p.add_argument("--rope-line-width", type=float, default=0.01,
                    help="ViewerGL rope spring line width.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=14.0)
    p.add_argument("--height", type=float, default=9.0)
    p.add_argument("--elev", type=float, default=15.0,
                    help="Camera elevation (degrees).")
    p.add_argument("--azim", type=float, default=-60.0,
                    help="Camera azimuth (degrees).")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--spring-stride", type=int, default=20)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bunny mesh helpers
# ---------------------------------------------------------------------------
def load_bunny_mesh(asset_name: str, prim_path: str, scale: float):
    """Load Stanford Bunny from Newton USD examples."""
    try:
        import newton.examples  # noqa: F811
        import newton.usd
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "Bunny mesh requires newton.examples, newton.usd, and pxr."
        ) from exc

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

    # Build render edges (unique triangle edges, subsampled).
    e01 = tri_indices[:, [0, 1]]
    e12 = tri_indices[:, [1, 2]]
    e20 = tri_indices[:, [2, 0]]
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0).astype(np.int32, copy=False)
    return mesh, points, tri_indices, all_edges, str(asset_path)


def quat_to_rotmat(q_xyzw):
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def transform_mesh_verts(pos, quat_xyzw, verts_local):
    """Transform mesh vertices to world frame."""
    R = quat_to_rotmat(quat_xyzw)
    return verts_local @ R.T + np.asarray(pos).reshape(1, 3)


def _output_stem(args: argparse.Namespace) -> str:
    return f"{args.prefix}_m{int(args.rigid_mass)}"


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(ir: dict, args: argparse.Namespace, device: str):
    """Build Newton model: rope + bunny + ground plane."""
    # Normalize to Newton convention (+Z up, gravity -Z) for rendering/sanity.
    # If IR was generated with reverse_z=True, flip Z once at load time.

    up_axis = newton.Axis.from_any("Z")
    builder = newton.ModelBuilder(up_axis=up_axis, gravity=0.0)

    x0 = np.asarray(ir["x0"], dtype=np.float32).copy()
    v0 = np.asarray(ir["v0"], dtype=np.float32).copy()
    mass = np.asarray(ir["mass"], dtype=np.float32).copy()
    reverse_z = newton_import_ir.ir_bool(ir, "reverse_z", default=False)
    if reverse_z:
        x0[:, 2] *= -1.0
        v0[:, 2] *= -1.0
    n_total = x0.shape[0]
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])

    # Shift rope: place rope COM at (0, 0, +drop_height), i.e. 5m above table.
    obj_com = x0[:n_obj].mean(axis=0)
    shift = np.array([0.0, 0.0, args.drop_height], dtype=np.float32) - obj_com
    x0[:n_obj] += shift
    # Also shift controller points if present
    ctrl_idx = np.asarray(ir.get("controller_idx", np.zeros((0,), dtype=np.int32)),
                          dtype=np.int32).ravel()
    if ctrl_idx.size:
        x0[ctrl_idx] += shift

    # Zero initial velocity (drop from rest)
    v0[:] = 0.0

    # Set object masses
    mass[:n_obj] = float(args.object_mass)
    if ctrl_idx.size:
        mass[ctrl_idx] = 0.0  # Controller particles are kinematic

    # Collision radius
    radius, _, _ = newton_import_ir.resolve_collision_radius(ir, n_total)
    if "contact_collision_dist" in ir:
        dist = newton_import_ir.ir_scalar(ir, "contact_collision_dist")
        radius[:n_obj] = max(dist * 0.5, newton_import_ir.EPSILON)
    if ctrl_idx.size:
        radius[ctrl_idx] = float(args.controller_radius)

    flags = np.full(n_total, int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
    builder.add_particles(
        pos=[tuple(r.tolist()) for r in x0],
        vel=[tuple(r.tolist()) for r in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
        flags=flags.astype(int).tolist(),
    )

    # Springs (object-only, drop controller springs)
    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).ravel()
    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel()
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel()

    if args.drop_controller_springs:
        keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
        edges, rest, ke, kd = edges[keep], rest[keep], ke[keep], kd[keep]

    for idx in range(edges.shape[0]):
        i, j = int(edges[idx, 0]), int(edges[idx, 1])
        builder.add_spring(i=i, j=j, ke=float(ke[idx]), kd=float(kd[idx]), control=0.0)
        builder.spring_rest_length[-1] = float(rest[idx])

    # Ground plane (table at z=0), Newton canonical orientation.
    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.mu = 0.5
    ground_cfg.ke = 1e4
    ground_cfg.kd = 1e2
    builder.add_ground_plane(cfg=ground_cfg)

    # Bunny mesh
    mesh, mesh_verts_local, mesh_tris, mesh_render_edges, mesh_asset_path = (
        load_bunny_mesh(args.bunny_asset, args.bunny_prim, args.bunny_scale)
    )

    # Position bunny on table with explicit, configurable orientation.
    qx, qy, qz, qw = [float(v) for v in args.bunny_quat_xyzw]
    bunny_quat_xyzw = [qx, qy, qz, qw]
    bunny_quat = wp.quat(qx, qy, qz, qw)

    # Compute bunny bounds in world frame with this orientation
    R_bunny = quat_to_rotmat(bunny_quat_xyzw)
    verts_rotated = (mesh_verts_local * float(args.bunny_scale)) @ R_bunny.T

    # In +Z-up convention, bunny bottom is min z; place that at z=0.
    bunny_z_max = verts_rotated[:, 2].max()
    bunny_z_min = verts_rotated[:, 2].min()
    bunny_pos_z = -bunny_z_min  # shift so that bottom aligns with z=0 table

    bunny_pos = np.array([0.0, 0.0, bunny_pos_z], dtype=np.float32)

    body = builder.add_body(
        xform=wp.transform(wp.vec3(*bunny_pos.tolist()), bunny_quat),
        mass=float(args.rigid_mass),
        inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
        lock_inertia=True,
        label="bunny",
    )
    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.body_mu)
    rigid_cfg.ke = float(args.body_ke)
    rigid_cfg.kd = float(args.body_kd)
    builder.add_shape_mesh(
        body=body, mesh=mesh,
        scale=(args.bunny_scale, args.bunny_scale, args.bunny_scale),
        cfg=rigid_cfg,
    )

    model = builder.finalize(device=device)
    # Newton canonical gravity: downward along -Z.
    model.set_gravity((0.0, 0.0, -9.8))

    meta = {
        "bunny_pos": bunny_pos.tolist(),
        "bunny_z_extent": [float(bunny_z_min), float(bunny_z_max)],
        "mesh_verts_local": mesh_verts_local,
        "mesh_scale": float(args.bunny_scale),
        "mesh_render_edges": mesh_render_edges,
        "mesh_asset_path": mesh_asset_path,
        "bunny_quat_xyzw": bunny_quat_xyzw,
        "rope_com_start": x0[:n_obj].mean(axis=0).tolist(),
    }
    return model, n_obj, mass, meta, x0


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
def simulate(model, ir, args, n_obj, mass, x0_shifted, device):
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
    )
    contacts = model.contacts()
    control = model.control()
    state_in = model.state()
    state_out = model.state()

    # Freeze controllers at shifted frame-0 position
    ctrl_idx = np.asarray(
        ir.get("controller_idx", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    ).ravel()
    ctrl_idx_wp = ctrl_target_wp = ctrl_vel_wp = None
    if args.freeze_controllers and ctrl_idx.size:
        ctrl_targets = x0_shifted[ctrl_idx]
        ctrl_idx_wp = wp.array(ctrl_idx, dtype=wp.int32, device=device)
        ctrl_target_wp = wp.array(ctrl_targets, dtype=wp.vec3, device=device)
        ctrl_vel_wp = wp.zeros(ctrl_idx.size, dtype=wp.vec3, device=device)

    sim_dt = float(args.sim_dt) if args.sim_dt else float(newton_import_ir.ir_scalar(ir, "sim_dt"))
    substeps = int(args.substeps) if args.substeps else int(newton_import_ir.ir_scalar(ir, "sim_substeps"))
    substeps = max(1, substeps)
    n_frames = max(2, int(args.frames))

    # Histories for visualization
    all_q_hist = []
    obj_q_hist = []
    body_q_hist = []
    body_vel_hist = []

    drag = 0.0
    if args.apply_drag and "drag_damping" in ir:
        drag = (
            newton_import_ir.ir_scalar(ir, "drag_damping")
            * float(args.drag_damping_scale)
        )

    t0 = time.perf_counter()
    for frame in range(n_frames):
        # Capture state
        q = state_in.particle_q.numpy().astype(np.float32)
        body_q_raw = state_in.body_q.numpy().astype(np.float32)
        body_qd_raw = state_in.body_qd.numpy().astype(np.float32)

        all_q_hist.append(q.copy())
        obj_q_hist.append(q[:n_obj].copy())
        body_q_hist.append(body_q_raw[0].copy())
        body_vel_hist.append(body_qd_raw[0, :3].copy())

        for _sub in range(substeps):
            state_in.clear_forces()
            if ctrl_idx_wp is not None:
                wp.launch(
                    newton_import_ir._write_kinematic_state,
                    dim=ctrl_idx.size,
                    inputs=[state_in.particle_q, state_in.particle_qd,
                            ctrl_idx_wp, ctrl_target_wp, ctrl_vel_wp],
                    device=device,
                )
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=n_obj,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        n_obj,
                        sim_dt,
                        drag,
                    ],
                    device=device,
                )

        if (frame + 1) % 50 == 0:
            print(f"  frame {frame+1}/{n_frames}")

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s")

    return {
        "particle_q_all": np.stack(all_q_hist),     # [F, N_total, 3]
        "particle_q_object": np.stack(obj_q_hist),  # [F, N_obj, 3]
        "body_q": np.stack(body_q_hist),             # [F, 7]
        "body_vel": np.stack(body_vel_hist),          # [F, 3]
        "sim_dt": sim_dt,
        "substeps": substeps,
        "wall_time": wall_time,
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _build_rope_render_edges(ir: dict, n_obj: int, stride: int) -> np.ndarray | None:
    if "spring_edges" not in ir:
        return None
    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        return None
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    edges = edges[keep]
    if edges.size == 0:
        return None
    stride = max(1, int(stride))
    return edges[::stride].copy()


def _overlay_text_lines_rgb(frame: np.ndarray, lines: list[str], font_size: int = 28) -> np.ndarray:
    """Overlay simple text labels on RGB frames (best-effort; no hard dependency)."""
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

    x = 20
    y = 18
    pad_x = 8
    pad_y = 4
    gap = 8
    for line in lines:
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle(
            (x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y),
            fill=(0, 0, 0, 140),
        )
        draw.text(
            (x, y),
            line,
            fill=(255, 170, 80, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 200),
        )
        y += th + gap

    return np.asarray(img, dtype=np.uint8)


def _compute_fixed_camera_pose(sim_data, meta, args) -> tuple[np.ndarray, float, float]:
    """Fit a static camera from trajectory bounds, then apply yaw/pitch style."""
    q_obj = sim_data["particle_q_object"].reshape(-1, 3)
    body_pos = sim_data["body_q"][:, :3]
    pts_min = np.minimum(q_obj.min(axis=0), body_pos.min(axis=0))
    pts_max = np.maximum(q_obj.max(axis=0), body_pos.max(axis=0))

    # Expand by bunny local extents so rigid geometry stays inside frame.
    mesh_local = np.asarray(meta["mesh_verts_local"], dtype=np.float32) * float(meta.get("mesh_scale", 1.0))
    half = np.max(np.abs(mesh_local), axis=0)
    pts_min = pts_min - half
    pts_max = pts_max + half

    all_center = 0.5 * (pts_min + pts_max)
    bunny_center = body_pos[0].astype(np.float32, copy=False)
    # Bias framing toward bunny so contact details are visible.
    center = 0.75 * bunny_center + 0.25 * all_center

    ext = pts_max - pts_min
    span_xy = float(max(ext[0], ext[1]))
    span_z = float(ext[2])
    span = max(span_xy * 1.25, span_z * 0.9, 0.5)

    yaw = float(args.camera_yaw)
    pitch = float(args.camera_pitch)
    yr = math.radians(yaw)
    pr = math.radians(pitch)
    front = np.array(
        [
            math.cos(yr) * math.cos(pr),
            math.sin(yr) * math.cos(pr),
            math.sin(pr),
        ],
        dtype=np.float32,
    )
    front_norm = max(float(np.linalg.norm(front)), 1e-6)
    front /= front_norm
    dist = span * float(args.camera_distance_scale)
    cam_pos = center - front * dist
    return cam_pos, pitch, yaw


def render_video_newton_gl(model, sim_data, ir, meta, args, n_obj, device):
    """Render with Newton built-in ViewerGL (headless), export MP4 via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH; required for MP4 encoding.")

    import newton.viewer

    width = int(args.screen_width)
    height = int(args.screen_height)
    slowdown = max(float(args.slowdown), 1e-6)
    fps_out = float(args.render_fps) / slowdown
    out_mp4 = Path(args.out_dir) / f"{_output_stem(args)}.mp4"
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

        # Viewer style tuned for this experiment.
        viewer.show_particles = True
        viewer.show_triangles = True
        viewer.show_visual = True
        viewer.show_static = True
        viewer.show_collision = True
        viewer.show_contacts = False
        viewer.show_ui = False
        viewer.picking_enabled = False

        # Fix camera once to keep view deterministic and static.
        if args.auto_frame_camera:
            try:
                viewer._frame_camera_on_model()  # noqa: SLF001
            except Exception:
                cpos, cpitch, cyaw = _compute_fixed_camera_pose(sim_data, meta, args)
                viewer.set_camera(
                    wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
                    float(cpitch),
                    float(cyaw),
                )
        else:
            cpos, cpitch, cyaw = _compute_fixed_camera_pose(sim_data, meta, args)
            viewer.set_camera(
                wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
                float(cpitch),
                float(cyaw),
            )

        # Make rope particles visually thicker (render-only, after simulation).
        vis_scale = float(args.particle_radius_vis_scale)
        vis_min = float(args.particle_radius_vis_min)
        radii = model.particle_radius.numpy().astype(np.float32, copy=False)
        radii = np.maximum(radii * vis_scale, vis_min)
        model.particle_radius.assign(radii)

        # Apply stable, readable colors for bunny and table.
        try:
            shape_colors = {}
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "bunny" in name:
                    shape_colors[idx] = (0.88, 0.35, 0.28)
                elif "ground" in name or "plane" in name or "table" in name:
                    shape_colors[idx] = (0.23, 0.26, 0.31)
            if shape_colors:
                viewer.update_shape_colors(shape_colors)
        except Exception:
            pass

        rope_edges = _build_rope_render_edges(ir, n_obj=n_obj, stride=args.spring_stride)
        starts_wp = ends_wp = None
        starts_np = ends_np = None
        if rope_edges is not None and rope_edges.size > 0:
            n_lines = int(rope_edges.shape[0])
            starts_wp = wp.empty(n_lines, dtype=wp.vec3, device=device)
            ends_wp = wp.empty(n_lines, dtype=wp.vec3, device=device)
            starts_np = np.zeros((n_lines, 3), dtype=np.float32)
            ends_np = np.zeros((n_lines, 3), dtype=np.float32)

        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
        if state.body_qd is not None:
            state.body_qd.zero_()

        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        n_frames = int(sim_data["particle_q_all"].shape[0])
        for f in range(n_frames):
            # Restore snapshot state for this frame.
            state.particle_q.assign(sim_data["particle_q_all"][f].astype(np.float32, copy=False))
            state.body_q.assign(sim_data["body_q"][f : f + 1].astype(np.float32, copy=False))

            viewer.begin_frame(float(f) * float(sim_data["sim_dt"]) * float(sim_data["substeps"]))
            viewer.log_state(state)

            if rope_edges is not None and starts_wp is not None and ends_wp is not None:
                q_obj = sim_data["particle_q_object"][f]
                starts_np[:] = q_obj[rope_edges[:, 0]]
                ends_np[:] = q_obj[rope_edges[:, 1]]
                starts_wp.assign(starts_np)
                ends_wp.assign(ends_np)
                viewer.log_lines(
                    "/bridge/rope_springs",
                    starts_wp,
                    ends_wp,
                    (0.60, 0.80, 0.98),
                    width=float(args.rope_line_width),
                    hidden=False,
                )

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()

            if args.overlay_label:
                sim_t = float(f) * float(sim_data["sim_dt"]) * float(sim_data["substeps"])
                frame = _overlay_text_lines_rgb(
                    frame,
                    [
                        f"SLOW MOTION {float(args.slowdown):.1f}x",
                        f"frame {f+1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
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

    print(f"  Video saved (Newton ViewerGL): {out_mp4}")
    return str(out_mp4)


def render_video_matplotlib(sim_data, meta, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    q_obj = sim_data["particle_q_object"]   # [F, N, 3]
    body_q = sim_data["body_q"]             # [F, 7]
    body_vel = sim_data["body_vel"]         # [F, 3]
    mesh_scale = float(meta.get("mesh_scale", 1.0))
    mesh_verts_local = meta["mesh_verts_local"] * mesh_scale
    mesh_render_edges = meta["mesh_render_edges"]
    frames = q_obj.shape[0]

    # In PhysTwin convention positive z = down.
    # For correct visual: invert z-axis so gravity appears downward on screen.
    # We negate z for all rendering.
    def flip_z(pts):
        """Flip z-axis for correct visual orientation."""
        out = pts.copy()
        if out.ndim == 1:
            out[2] = -out[2]
        else:
            out[:, 2] = -out[:, 2]
        return out

    # Compute global bounds for fixed camera
    all_obj = q_obj.reshape(-1, 3)
    all_body_pos = body_q[:, :3]

    # Also include bunny mesh extent
    bunny_extent = float(np.max(np.abs(mesh_verts_local))) * 1.2
    all_pts_z_flipped = flip_z(all_obj)
    body_z_flipped = flip_z(all_body_pos)

    xyz_min = np.minimum(all_pts_z_flipped.min(axis=0), body_z_flipped.min(axis=0)) - bunny_extent
    xyz_max = np.maximum(all_pts_z_flipped.max(axis=0), body_z_flipped.max(axis=0)) + bunny_extent

    # Ensure symmetric-ish for nice camera framing
    center = 0.5 * (xyz_min + xyz_max)
    half_span = max(0.5 * np.max(xyz_max - xyz_min), 0.5)

    # Load spring edges for rope visualization
    ir_path = args.ir.resolve()
    spring_edges = None
    with np.load(ir_path, allow_pickle=True) as ir:
        if "spring_edges" in ir and "num_object_points" in ir:
            edges = np.asarray(ir["spring_edges"], dtype=np.int32)
            n_ir_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
            keep = (edges[:, 0] < n_ir_obj) & (edges[:, 1] < n_ir_obj)
            edges = edges[keep]
            if edges.shape[0] > 0:
                spring_edges = edges[::args.spring_stride].copy()

    # Setup figure
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.06, 0.07, 0.09))
    fig.set_facecolor((0.06, 0.07, 0.09))

    # Fixed camera
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z (up)", color="white")
    ax.tick_params(colors="white")

    # Initial content
    pts0 = flip_z(q_obj[0])
    scatter = ax.scatter(
        pts0[:, 0], pts0[:, 1], pts0[:, 2],
        s=args.point_size, c="#58B0E0", alpha=0.85, depthshade=False,
    )

    # Bunny mesh wireframe (initial)
    bpos0 = body_q[0, :3]
    bquat0 = body_q[0, 3:7]
    v0_world = transform_mesh_verts(bpos0, bquat0, mesh_verts_local)
    v0_flipped = flip_z(v0_world)
    bunny_segments = v0_flipped[mesh_render_edges[::args.spring_stride]]
    bunny_lc = Line3DCollection(bunny_segments, colors="#E84040", linewidths=0.6, alpha=0.85)
    ax.add_collection3d(bunny_lc)

    # Rope spring lines
    rope_lc = None
    if spring_edges is not None and spring_edges.shape[0] > 0:
        rope_segs = pts0[spring_edges]
        rope_lc = Line3DCollection(rope_segs, colors="#7FB8E0", linewidths=0.3, alpha=0.3)
        ax.add_collection3d(rope_lc)

    # Ground plane (table) at z=0 (after z-flip: z=0)
    table_half = max(half_span * 0.8, 0.3)
    table_verts = np.array([
        [center[0]-table_half, center[1]-table_half, 0],
        [center[0]+table_half, center[1]-table_half, 0],
        [center[0]+table_half, center[1]+table_half, 0],
        [center[0]-table_half, center[1]+table_half, 0],
    ])
    table_poly = Poly3DCollection(
        [table_verts], facecolors="#3A4050", edgecolors="#6A7080",
        linewidths=1.0, alpha=0.6,
    )
    ax.add_collection3d(table_poly)

    # Text annotations
    slowdown = float(args.slowdown)
    effective_fps = args.render_fps / slowdown
    title_text = ax.set_title(
        f"Rope Drop onto Stanford Bunny (Newton SemiImplicit)  |  bunny mass={args.rigid_mass:.0f} kg",
        fontsize=13, pad=12, color="white",
    )
    meta_text = fig.text(
        0.02, 0.03, "", fontsize=10, color="#AABBCC",
        fontfamily="monospace",
    )
    slow_label = fig.text(
        0.98, 0.03, f"SLOW MOTION {slowdown:.0f}×",
        fontsize=12, color="#FF8844", fontweight="bold",
        ha="right", fontfamily="sans-serif",
    )

    # Render
    out_mp4 = Path(args.out_dir) / f"{_output_stem(args)}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=effective_fps, codec="libx264", bitrate=8000)
    with writer.saving(fig, str(out_mp4), dpi=args.dpi):
        for f in range(frames):
            pts = flip_z(q_obj[f])
            bpos = body_q[f, :3]
            bquat = body_q[f, 3:7]

            # Update scatter
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

            # Update rope structural lines
            if rope_lc is not None:
                rope_lc.set_segments(pts[spring_edges])

            # Update bunny wireframe
            v_world = transform_mesh_verts(bpos, bquat, mesh_verts_local)
            v_flipped = flip_z(v_world)
            bunny_lc.set_segments(v_flipped[mesh_render_edges[::args.spring_stride]])

            # Metrics
            bspeed = float(np.linalg.norm(body_vel[f]))
            rope_com = pts.mean(axis=0)
            bpos_vis = flip_z(bpos)
            dist = float(np.linalg.norm(bpos_vis - rope_com))

            sim_time = f * sim_data["sim_dt"] * sim_data["substeps"]
            meta_text.set_text(
                f"frame {f+1:03d}/{frames}  |  t={sim_time:.3f}s  |  "
                f"bunny_speed={bspeed:.4f} m/s  |  dist(bunny,rope_COM)={dist:.3f} m"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"  Video saved: {out_mp4}")
    return str(out_mp4)


# ---------------------------------------------------------------------------
# Outputs / summary
# ---------------------------------------------------------------------------
def save_scene_npz(
    args: argparse.Namespace,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
) -> Path:
    scene_npz = args.out_dir / f"{_output_stem(args)}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        particle_q_object=sim_data["particle_q_object"],
        body_q=sim_data["body_q"],
        body_vel=sim_data["body_vel"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        rigid_mesh_vertices_local=meta["mesh_verts_local"],
        rigid_mesh_scale=np.float32(meta.get("mesh_scale", args.bunny_scale)),
        rigid_mesh_render_edges=meta["mesh_render_edges"],
        rigid_mass=np.float32(args.rigid_mass),
        drop_height=np.float32(args.drop_height),
        num_object_points=np.int32(n_obj),
    )
    return scene_npz


def build_summary(
    args: argparse.Namespace,
    ir: dict[str, Any],
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
) -> dict[str, Any]:
    return {
        "experiment": "rope_bunny_drop",
        "ir_path": str(args.ir.resolve()),
        "drop_height_m": float(args.drop_height),
        "object_mass_per_particle": float(args.object_mass),
        "n_object_particles": n_obj,
        "total_object_mass": float(n_obj * args.object_mass),
        "rigid_mass": float(args.rigid_mass),
        "mass_ratio": float(n_obj * args.object_mass / args.rigid_mass),
        "bunny_scale": float(args.bunny_scale),
        "bunny_quat_xyzw": [float(v) for v in meta.get("bunny_quat_xyzw", list(args.bunny_quat_xyzw))],
        "reverse_z": bool(newton_import_ir.ir_bool(ir, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "frames": int(args.frames),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "wall_time_sec": float(sim_data["wall_time"]),
        "slowdown_factor": float(args.slowdown),
        "body_speed_initial": float(np.linalg.norm(sim_data["body_vel"][0])),
        "body_speed_final": float(np.linalg.norm(sim_data["body_vel"][-1])),
        "body_speed_max": float(np.max(np.linalg.norm(sim_data["body_vel"], axis=1))),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "render_backend": str(args.render_backend),
        "camera_pos": [float(v) for v in args.camera_pos],
        "camera_pitch": float(args.camera_pitch),
        "camera_yaw": float(args.camera_yaw),
        "camera_fov": float(args.camera_fov),
        "camera_distance_scale": float(args.camera_distance_scale),
        "auto_frame_camera": bool(args.auto_frame_camera),
    }


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    summary_path = args.out_dir / f"{_output_stem(args)}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def render_experiment_video(
    args: argparse.Namespace,
    model,
    sim_data: dict[str, Any],
    ir: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
    device: str,
) -> str:
    if args.render_backend == "newton_gl":
        return render_video_newton_gl(
            model=model,
            sim_data=sim_data,
            ir=ir,
            meta=meta,
            args=args,
            n_obj=n_obj,
            device=device,
        )
    return render_video_matplotlib(sim_data, meta, args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    ir = newton_import_ir.load_ir(args.ir.resolve())

    print(f"Building model: rope drops {args.drop_height}m onto bunny (mass={args.rigid_mass})")
    model, n_obj, mass, meta, x0_shifted = build_model(ir, args, device)

    print("Running simulation...")
    sim_data = simulate(model, ir, args, n_obj, mass, x0_shifted, device)

    scene_npz = save_scene_npz(args, sim_data, meta, n_obj)
    print(f"  Scene NPZ: {scene_npz}")

    summary = build_summary(args, ir, sim_data, meta, n_obj)
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}")
    print(json.dumps(summary, indent=2))

    print("Rendering video...")
    _ = render_experiment_video(args, model, sim_data, ir, meta, n_obj, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
