#!/usr/bin/env python3
"""Object-only rope drop onto a Stanford Bunny in Newton SemiImplicit.

This version is intentionally aligned with the main PhysTwin -> Newton importer:

- load a PhysTwin IR bundle
- drop controller particles entirely
- keep only object particles + object-object springs
- reuse importer-side contact/drag helpers where possible
- run a single Newton SemiImplicit simulation
- render one deterministic ViewerGL video

The scenario is:
1. put the Stanford Bunny on the ground plane
2. place the rope directly above it
3. set the rope bottom to `bunny_top_z + drop_height`
4. drop from rest under gravity
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from demo_common import CORE_DIR, load_core_module, overlay_text_lines_rgb

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
NEWTON_PY_ROOT = CORE_DIR.parents[2] / "newton"
if str(NEWTON_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWTON_PY_ROOT))

path_defaults = load_core_module("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = load_core_module("newton_import_ir", CORE_DIR / "newton_import_ir.py")

import newton  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-only rope drops onto a Stanford Bunny.")
    p.add_argument("--ir", type=Path, required=True, help="Path to PhysTwin IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="rope_bunny_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument("--drop-height", type=float, default=5.0, help="Rope bottom height above bunny top, in meters.")
    p.add_argument("--object-mass", type=float, default=1.0, help="Per-particle rope object mass.")
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument("--disable-particle-contact-kernel", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--rigid-mass", type=float, default=5.0)
    p.add_argument("--body-mu", type=float, default=0.5)
    p.add_argument("--body-ke", type=float, default=5.0e4)
    p.add_argument("--body-kd", type=float, default=5.0e2)
    p.add_argument("--bunny-scale", type=float, default=0.12)
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")
    p.add_argument(
        "--bunny-quat-xyzw",
        type=float,
        nargs=4,
        default=(0.70710678, 0.0, 0.0, 0.70710678),
        metavar=("X", "Y", "Z", "W"),
    )

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=2.0)
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-pos", type=float, nargs=3, default=(4.5, -6.0, 2.6), metavar=("X", "Y", "Z"))
    p.add_argument("--camera-pitch", type=float, default=-8.0)
    p.add_argument("--camera-yaw", type=float, default=-42.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.02)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--rope-line-width", type=float, default=0.01)
    p.add_argument("--spring-stride", type=int, default=20)
    return p.parse_args()


def load_ir(path: Path) -> dict[str, np.ndarray]:
    return newton_import_ir.load_ir(path.resolve())


def load_bunny_mesh(asset_name: str, prim_path: str):
    try:
        import newton.examples  # noqa: PLC0415
        import newton.usd  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415
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
        raise ValueError(f"Bunny mesh index length must be multiple of 3, got {indices.size}")
    tri_indices = indices.reshape(-1, 3)
    e01 = tri_indices[:, [0, 1]]
    e12 = tri_indices[:, [1, 2]]
    e20 = tri_indices[:, [2, 0]]
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0).astype(np.int32, copy=False)
    return mesh, points, tri_indices, all_edges, str(asset_path)


def quat_to_rotmat(q_xyzw: list[float] | tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = max((x * x + y * y + z * z + w * w) ** 0.5, 1.0e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _copy_object_only_ir(ir: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
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
    v0[:] = 0.0

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = np.full(n_obj, float(args.object_mass), dtype=np.float32)
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
    return ir_demo


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
        disable_particle_contact_kernel=bool(args.disable_particle_contact_kernel),
        shape_contacts=True,
        add_ground_plane=True,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        device=device,
    )

    checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)
    particle_contacts = newton_import_ir._resolve_particle_contacts(cfg, ir_obj)

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    radius, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    newton_import_ir._add_ground_plane(builder, ir_obj, cfg, checks)

    mesh, mesh_verts_local, mesh_tris, mesh_render_edges, mesh_asset_path = load_bunny_mesh(args.bunny_asset, args.bunny_prim)
    qx, qy, qz, qw = [float(v) for v in args.bunny_quat_xyzw]
    bunny_quat_xyzw = [qx, qy, qz, qw]
    bunny_quat = wp.quat(qx, qy, qz, qw)

    verts_rotated = (mesh_verts_local * float(args.bunny_scale)) @ quat_to_rotmat(bunny_quat_xyzw).T
    bunny_z_min = float(verts_rotated[:, 2].min())
    bunny_z_max = float(verts_rotated[:, 2].max())
    bunny_pos = np.array([0.0, 0.0, -bunny_z_min], dtype=np.float32)
    bunny_top_z = bunny_pos[2] + bunny_z_max

    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    rope_center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    shift = np.array(
        [
            -float(rope_center_xy[0]),
            -float(rope_center_xy[1]),
            bunny_top_z + float(args.drop_height) - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]

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
        body=body,
        mesh=mesh,
        scale=(float(args.bunny_scale), float(args.bunny_scale), float(args.bunny_scale)),
        cfg=rigid_cfg,
    )

    model = builder.finalize(device=device)
    if not (particle_contacts and (not cfg.disable_particle_contact_kernel)):
        model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "mesh_verts_local": mesh_verts_local.astype(np.float32, copy=False),
        "mesh_scale": float(args.bunny_scale),
        "mesh_render_edges": mesh_render_edges.astype(np.int32, copy=False),
        "mesh_asset_path": mesh_asset_path,
        "bunny_quat_xyzw": bunny_quat_xyzw,
        "bunny_pos": bunny_pos.astype(np.float32, copy=False),
        "bunny_top_z": float(bunny_top_z),
        "rope_shift": shift.astype(np.float32, copy=False),
        "render_edges": render_edges,
    }
    return model, meta, n_obj, shifted_q


def simulate(model: newton.Model, ir_obj: dict[str, Any], args: argparse.Namespace, n_obj: int, device: str) -> dict[str, Any]:
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=bool(args.disable_particle_contact_kernel),
        shape_contacts=True,
        add_ground_plane=True,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
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
    contacts = model.contacts() if newton_import_ir._use_collision_pipeline(cfg, ir_obj) else None

    sim_dt = float(args.sim_dt) if args.sim_dt is not None else float(newton_import_ir.ir_scalar(ir_obj, "sim_dt"))
    substeps = int(args.substeps) if args.substeps is not None else int(newton_import_ir.ir_scalar(ir_obj, "sim_substeps"))
    substeps = max(1, substeps)

    particle_contacts = newton_import_ir._resolve_particle_contacts(cfg, ir_obj)
    particle_contact_kernel = particle_contacts and (not cfg.disable_particle_contact_kernel)
    particle_grid = model.particle_grid if particle_contact_kernel else None
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))

    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)

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
        body_q.append(state_in.body_q.numpy().astype(np.float32).copy())
        body_vel.append(state_in.body_qd.numpy().astype(np.float32)[:, :3].copy())

        for _ in range(substeps):
            state_in.clear_forces()

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if contacts is not None:
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
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": np.stack(particle_q_all),
        "particle_q_object": np.stack(particle_q_object),
        "body_q": np.stack(body_q),
        "body_vel": np.stack(body_vel),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": float(wall_time),
    }


def render_video(model: newton.Model, sim_data: dict[str, Any], meta: dict[str, Any], args: argparse.Namespace, device: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps) / max(float(args.slowdown), 1.0e-6)
    out_mp4 = args.out_dir / f"{args.prefix}_m{int(args.rigid_mass)}.mp4"
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
        viewer.show_particles = True
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
        viewer.set_camera(
            wp.vec3(float(args.camera_pos[0]), float(args.camera_pos[1]), float(args.camera_pos[2])),
            float(args.camera_pitch),
            float(args.camera_yaw),
        )

        radii = model.particle_radius.numpy().astype(np.float32, copy=False)
        radii = np.maximum(radii * float(args.particle_radius_vis_scale), float(args.particle_radius_vis_min))
        model.particle_radius.assign(radii)

        try:
            shape_colors = {}
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "bunny" in name:
                    shape_colors[idx] = (0.88, 0.35, 0.28)
                elif "ground" in name or "plane" in name:
                    shape_colors[idx] = (0.23, 0.26, 0.31)
            if shape_colors:
                viewer.update_shape_colors(shape_colors)
        except Exception:
            pass

        rope_edges = np.asarray(meta["render_edges"], dtype=np.int32)
        starts_wp = wp.empty(len(rope_edges), dtype=wp.vec3, device=device) if rope_edges.size else None
        ends_wp = wp.empty(len(rope_edges), dtype=wp.vec3, device=device) if rope_edges.size else None

        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
        if state.body_qd is not None:
            state.body_qd.zero_()

        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(sim_data["particle_q_all"].shape[0])
        for f in range(n_frames):
            state.particle_q.assign(sim_data["particle_q_all"][f].astype(np.float32, copy=False))
            state.body_q.assign(sim_data["body_q"][f].astype(np.float32, copy=False))

            viewer.begin_frame(float(f) * float(sim_data["sim_dt"]) * float(sim_data["substeps"]))
            viewer.log_state(state)

            if rope_edges.size and starts_wp is not None and ends_wp is not None:
                q_obj = sim_data["particle_q_object"][f]
                starts_wp.assign(q_obj[rope_edges[:, 0]].astype(np.float32, copy=False))
                ends_wp.assign(q_obj[rope_edges[:, 1]].astype(np.float32, copy=False))
                viewer.log_lines(
                    "/demo/rope_springs",
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
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        f"SLOW MOTION {float(args.slowdown):.1f}x",
                        f"frame {f + 1:03d}/{n_frames:03d}  t={sim_t:.3f}s",
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

    print(f"  Video saved: {out_mp4}", flush=True)
    return out_mp4


def save_scene_npz(args: argparse.Namespace, sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int) -> Path:
    scene_npz = args.out_dir / f"{args.prefix}_m{int(args.rigid_mass)}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        particle_q_object=sim_data["particle_q_object"],
        body_q=sim_data["body_q"],
        body_vel=sim_data["body_vel"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        rigid_mesh_vertices_local=meta["mesh_verts_local"],
        rigid_mesh_scale=np.float32(meta["mesh_scale"]),
        rigid_mesh_render_edges=meta["mesh_render_edges"],
        rigid_mass=np.float32(args.rigid_mass),
        drop_height=np.float32(args.drop_height),
        num_object_points=np.int32(n_obj),
    )
    return scene_npz


def build_summary(args: argparse.Namespace, ir_obj: dict[str, Any], sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int, out_mp4: Path) -> dict[str, Any]:
    body_speed = np.linalg.norm(sim_data["body_vel"][:, 0, :], axis=1)
    return {
        "experiment": "rope_bunny_drop_object_only",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "drop_height_m": float(args.drop_height),
        "object_mass_per_particle": float(args.object_mass),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(n_obj * args.object_mass),
        "rigid_mass": float(args.rigid_mass),
        "mass_ratio": float(n_obj * args.object_mass / max(args.rigid_mass, 1.0e-8)),
        "bunny_scale": float(args.bunny_scale),
        "bunny_quat_xyzw": [float(v) for v in meta["bunny_quat_xyzw"]],
        "bunny_top_z": float(meta["bunny_top_z"]),
        "reverse_z": bool(newton_import_ir.ir_bool(ir_obj, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "frames": int(args.frames),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "wall_time_sec": float(sim_data["wall_time"]),
        "slowdown_factor": float(args.slowdown),
        "body_speed_initial": float(body_speed[0]),
        "body_speed_final": float(body_speed[-1]),
        "body_speed_max": float(np.max(body_speed)),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "camera_pos": [float(v) for v in args.camera_pos],
        "camera_pitch": float(args.camera_pitch),
        "camera_yaw": float(args.camera_yaw),
        "camera_fov": float(args.camera_fov),
        "render_video": str(out_mp4),
    }


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    summary_path = args.out_dir / f"{args.prefix}_m{int(args.rigid_mass)}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    ir = load_ir(args.ir)
    ir_obj = _copy_object_only_ir(ir, args)

    if args.sim_dt is None:
        args.sim_dt = float(newton_import_ir.ir_scalar(ir_obj, "sim_dt"))
    if args.substeps is None:
        args.substeps = int(newton_import_ir.ir_scalar(ir_obj, "sim_substeps"))

    print(f"Building object-only rope+bunny model from {args.ir.resolve()}", flush=True)
    model, meta, n_obj, _shifted_q = build_model(ir_obj, args, device)

    print("Running simulation...", flush=True)
    sim_data = simulate(model, ir_obj, args, n_obj, device)

    scene_npz = save_scene_npz(args, sim_data, meta, n_obj)
    print(f"  Scene NPZ: {scene_npz}", flush=True)

    print("Rendering video...", flush=True)
    out_mp4 = render_video(model, sim_data, meta, args, device)

    summary = build_summary(args, ir_obj, sim_data, meta, n_obj, out_mp4)
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
