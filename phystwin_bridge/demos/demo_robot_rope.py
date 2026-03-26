#!/usr/bin/env python3
"""Active rigid pusher demo: a robot proxy actively pushes a rope.

This is a minimum active-control / two-way-coupling demo for the PhysTwin ->
Newton bridge:

- load one rope from PhysTwin IR
- drop controllers and keep one object-only rope
- pin the two rope endpoints in space so the rope hangs freely
- add one dynamic rigid sphere as a robot end-effector proxy
- drive the sphere with a translational PD controller toward a target path
- let native particle-shape contact produce the two-way coupling

The intent is not to build a full robot tonight. The rigid sphere is the
smallest controlled actuator that still demonstrates:

1. active control input
2. deformable response
3. rigid-body reaction / tracking lag under contact load
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

NEWTON_PY_ROOT = Path(__file__).resolve().parents[2] / "newton"
if str(NEWTON_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWTON_PY_ROOT))

from demo_rope_bunny_drop import (
    _apply_drag_correction_ignore_axis,
    _copy_object_only_ir,
    _effective_spring_scales,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    load_ir,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
)
from demo_shared import compute_visual_particle_radii, temporary_particle_radius_override

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_ROOT = Path(__file__).resolve().parents[1]


@wp.kernel
def _apply_body_pd_force_translation(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_index: int,
    target_pos: wp.vec3,
    target_vel: wp.vec3,
    kp: float,
    kd: float,
    force_limit: float,
):
    pos = wp.transform_get_translation(body_q[body_index])
    vel = wp.spatial_top(body_qd[body_index])

    force = (target_pos - pos) * kp + (target_vel - vel) * kd
    force_norm = wp.length(force)
    if force_limit > 0.0 and force_norm > force_limit:
        force = force * (force_limit / (force_norm + 1.0e-8))

    body_f[body_index] = body_f[body_index] + wp.spatial_vector(force, wp.vec3(0.0))


def _default_rope_ir() -> Path:
    return BRIDGE_ROOT / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active robot proxy pushes a hanging rope with real two-way deformable-rigid coupling."
    )
    p.add_argument("--ir", type=Path, default=_default_rope_ir(), help="Path to rope PhysTwin IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_push_rope")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--sim-dt", type=float, default=1.0e-4)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--gravity-mag", type=float, default=9.8)

    p.add_argument("--object-mass", type=float, default=1.0, help="Fallback per-particle rope mass.")
    p.add_argument(
        "--auto-set-weight",
        type=float,
        default=None,
        help=(
            "Target total deformable mass [kg]. If provided, auto-compute the needed "
            "weight_scale so mass + spring + contact all follow the same ratio."
        ),
    )
    p.add_argument(
        "--mass-spring-scale",
        type=float,
        default=None,
        help=(
            "Single scale factor applied consistently to object mass, spring_ke, and spring_kd. "
            "Use this instead of separately changing mass / spring-ke-scale / spring-kd-scale."
        ),
    )
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument(
        "--drag-ignore-gravity-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply drag only orthogonal to gravity so free-fall acceleration is preserved.",
    )

    p.add_argument("--anchor-height", type=float, default=1.20)
    p.add_argument("--anchor-count-per-end", type=int, default=1)

    p.add_argument("--robot-mass", type=float, default=3.0)
    p.add_argument("--robot-radius", type=float, default=0.18)
    p.add_argument("--robot-contact-mu", type=float, default=0.4)
    p.add_argument("--robot-contact-ke", type=float, default=8.0e4)
    p.add_argument("--robot-contact-kd", type=float, default=8.0e2)
    p.add_argument("--robot-start-offset", type=float, default=0.55)
    p.add_argument("--robot-end-offset", type=float, default=-0.12)
    p.add_argument("--robot-z-offset", type=float, default=0.0)
    p.add_argument("--robot-kp", type=float, default=3000.0)
    p.add_argument("--robot-kd", type=float, default=250.0)
    p.add_argument("--robot-force-limit", type=float, default=1500.0)
    p.add_argument("--settle-seconds", type=float, default=0.01)
    p.add_argument("--push-seconds", type=float, default=0.03)
    p.add_argument("--hold-seconds", type=float, default=0.10)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=8.0)
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(-1.6, 1.35, 1.05),
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=-10.0)
    p.add_argument("--camera-yaw", type=float, default=-40.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.004)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--rope-line-width", type=float, default=0.01)
    p.add_argument("--spring-stride", type=int, default=20)
    return p.parse_args()


def _rope_endpoints(edges: np.ndarray, n_obj: int) -> np.ndarray:
    degree = np.zeros((n_obj,), dtype=np.int32)
    for i, j in np.asarray(edges, dtype=np.int32):
        if 0 <= i < n_obj and 0 <= j < n_obj:
            degree[i] += 1
            degree[j] += 1
    endpoints = np.flatnonzero(degree == 1)
    if endpoints.size >= 2:
        if endpoints.size == 2:
            return endpoints.astype(np.int32, copy=False)
        pts = None
        best_pair = (int(endpoints[0]), int(endpoints[1]))
        best_dist = -1.0
        for i_idx in range(endpoints.size):
            for j_idx in range(i_idx + 1, endpoints.size):
                i = int(endpoints[i_idx])
                j = int(endpoints[j_idx])
                if pts is not None:
                    pass
                best_pair = (i, j)
                break
            if best_dist >= 0.0:
                break
        return np.asarray(best_pair, dtype=np.int32)
    return np.asarray([0, max(0, n_obj - 1)], dtype=np.int32)


def _anchor_particle_indices(
    shifted_q: np.ndarray, endpoint_indices: np.ndarray, count_per_end: int
) -> np.ndarray:
    count_per_end = max(1, int(count_per_end))
    selected: list[int] = []
    for endpoint in endpoint_indices.tolist():
        endpoint_pos = shifted_q[int(endpoint)]
        d = np.linalg.norm(shifted_q - endpoint_pos[None, :], axis=1)
        nearest = np.argsort(d)[:count_per_end]
        selected.extend(int(v) for v in nearest.tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1.0e-8:
        return np.asarray(fallback, dtype=np.float32).copy()
    return (np.asarray(v, dtype=np.float32) / n).astype(np.float32, copy=False)


def _robot_target_state(t: float, meta: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(meta["robot_start_pos"], dtype=np.float32)
    end = np.asarray(meta["robot_end_pos"], dtype=np.float32)
    zero = np.zeros((3,), dtype=np.float32)
    settle = float(args.settle_seconds)
    push = max(float(args.push_seconds), 1.0e-8)
    hold = max(float(args.hold_seconds), 0.0)
    if t <= settle:
        return start.copy(), zero
    if t <= settle + push:
        alpha = float((t - settle) / push)
        pos = (1.0 - alpha) * start + alpha * end
        vel = (end - start) / push
        return pos.astype(np.float32), vel.astype(np.float32)
    if t <= settle + push + hold:
        return end.copy(), zero
    return end.copy(), zero


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
    raw_ir = load_ir(args.ir)
    _validate_scaling_args(args)
    _maybe_autoset_mass_spring_scale(args, raw_ir)
    ir_obj = _copy_object_only_ir(raw_ir, args)

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)

    endpoint_indices = _rope_endpoints(edges, n_obj)
    endpoint_mid = 0.5 * (x0[int(endpoint_indices[0])] + x0[int(endpoint_indices[1])])
    shift = np.array(
        [
            -float(endpoint_mid[0]),
            -float(endpoint_mid[1]),
            float(args.anchor_height) - float(endpoint_mid[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift

    anchor_indices = _anchor_particle_indices(
        shifted_q, endpoint_indices=endpoint_indices, count_per_end=int(args.anchor_count_per_end)
    )

    rope_axis = _safe_normalize(
        shifted_q[int(endpoint_indices[1])] - shifted_q[int(endpoint_indices[0])],
        fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    push_dir = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), rope_axis)
    push_dir = _safe_normalize(push_dir, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    rope_center = shifted_q.mean(axis=0).astype(np.float32, copy=False)
    robot_start_pos = (
        rope_center - push_dir * float(args.robot_start_offset) + np.array([0.0, 0.0, float(args.robot_z_offset)], dtype=np.float32)
    ).astype(np.float32)
    robot_end_pos = (
        rope_center + push_dir * float(args.robot_end_offset) + np.array([0.0, 0.0, float(args.robot_z_offset)], dtype=np.float32)
    ).astype(np.float32)

    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(spring_ke_scale),
        spring_kd_scale=float(spring_kd_scale),
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=False,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
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

    checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    _, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=False)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]
    for idx in anchor_indices.tolist():
        builder.particle_flags[idx] = int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)
        builder.particle_mass[idx] = 0.0

    robot_body = builder.add_body(
        xform=wp.transform(wp.vec3(*robot_start_pos.tolist()), wp.quat_identity()),
        mass=float(args.robot_mass),
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
        lock_inertia=True,
        label="robot_pusher",
    )
    robot_cfg = builder.default_shape_cfg.copy()
    robot_cfg.mu = float(args.robot_contact_mu)
    robot_cfg.ke = float(args.robot_contact_ke)
    robot_cfg.kd = float(args.robot_contact_kd)
    robot_shape = builder.add_shape_sphere(body=robot_body, radius=float(args.robot_radius), cfg=robot_cfg, label="robot_pusher_sphere")

    model = builder.finalize(device=device)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)

    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "robot_body": int(robot_body),
        "robot_shape": int(robot_shape),
        "robot_radius": float(args.robot_radius),
        "robot_start_pos": robot_start_pos.astype(np.float32),
        "robot_end_pos": robot_end_pos.astype(np.float32),
        "endpoint_indices": endpoint_indices.astype(np.int32),
        "anchor_indices": anchor_indices.astype(np.int32),
        "anchor_positions": shifted_q[anchor_indices].astype(np.float32),
        "rope_center": rope_center.astype(np.float32),
        "push_dir": push_dir.astype(np.float32),
        "render_edges": render_edges,
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
    }
    return model, ir_obj, meta, n_obj


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(spring_ke_scale),
        spring_kd_scale=float(spring_kd_scale),
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=False,
        disable_particle_contact_kernel=True,
        shape_contacts=True,
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
    substeps = max(1, int(args.substeps))
    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

    n_frames = max(2, int(args.frames))
    particle_q_all: list[np.ndarray] = []
    particle_q_object: list[np.ndarray] = []
    body_q: list[np.ndarray] = []
    body_vel: list[np.ndarray] = []
    robot_target_pos: list[np.ndarray] = []
    robot_target_vel: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in range(n_frames):
        q = state_in.particle_q.numpy().astype(np.float32)
        particle_q_all.append(q.copy())
        particle_q_object.append(q[:n_obj].copy())
        body_q.append(state_in.body_q.numpy().astype(np.float32).copy())
        body_vel.append(state_in.body_qd.numpy().astype(np.float32)[:, :3].copy())
        target_pos_frame, target_vel_frame = _robot_target_state(
            float(frame) * sim_dt * float(substeps), meta, args
        )
        robot_target_pos.append(target_pos_frame.copy())
        robot_target_vel.append(target_vel_frame.copy())

        for sub in range(substeps):
            state_in.clear_forces()
            sim_t = (float(frame) * float(substeps) + float(sub)) * sim_dt
            target_pos, target_vel = _robot_target_state(sim_t, meta, args)
            wp.launch(
                _apply_body_pd_force_translation,
                dim=1,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    int(meta["robot_body"]),
                    wp.vec3(*target_pos.tolist()),
                    wp.vec3(*target_vel.tolist()),
                    float(args.robot_kp),
                    float(args.robot_kd),
                    float(args.robot_force_limit),
                ],
                device=device,
            )

            if contacts is not None:
                model.collide(state_in, contacts)

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis,
                        dim=n_obj,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            n_obj,
                            sim_dt,
                            drag,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        newton_import_ir._apply_drag_correction,
                        dim=n_obj,
                        inputs=[state_in.particle_q, state_in.particle_qd, n_obj, sim_dt, drag],
                        device=device,
                    )

        if (frame + 1) % 50 == 0 or frame == n_frames - 1:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": np.stack(particle_q_all),
        "particle_q_object": np.stack(particle_q_object),
        "body_q": np.stack(body_q),
        "body_vel": np.stack(body_vel),
        "robot_target_pos": np.stack(robot_target_pos),
        "robot_target_vel": np.stack(robot_target_vel),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": float(wall_time),
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
        viewer.show_particles = True
        viewer.show_triangles = False
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
            viewer.update_shape_colors({int(meta["robot_shape"]): (0.92, 0.34, 0.24)})
        except Exception:
            pass

        render_radii = compute_visual_particle_radii(
            model.particle_radius.numpy(),
            radius_scale=float(args.particle_radius_vis_scale),
            radius_cap=float(args.particle_radius_vis_min),
        )

        rope_edges = np.asarray(meta["render_edges"], dtype=np.int32)
        rope_line_starts_wp = wp.empty(len(rope_edges), dtype=wp.vec3, device=device) if rope_edges.size else None
        rope_line_ends_wp = wp.empty(len(rope_edges), dtype=wp.vec3, device=device) if rope_edges.size else None

        anchor_positions = np.asarray(meta["anchor_positions"], dtype=np.float32)
        anchor_xforms_wp = wp.array(
            [wp.transform(wp.vec3(*row.tolist()), wp.quat_identity()) for row in anchor_positions],
            dtype=wp.transform,
            device=device,
        )
        anchor_colors_wp = wp.array(
            [wp.vec3(0.34, 0.90, 0.52) for _ in range(anchor_positions.shape[0])],
            dtype=wp.vec3,
            device=device,
        )

        state = model.state()
        if state.body_qd is not None:
            state.body_qd.zero_()

        with temporary_particle_radius_override(model, render_radii):
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
                render_indices = np.clip(
                    np.rint(sample_times / sim_frame_dt).astype(np.int32), 0, n_sim_frames - 1
                )

            for out_idx, sim_idx in enumerate(render_indices):
                state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
                state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))

                sim_t = float(sim_idx) * sim_frame_dt
                viewer.begin_frame(sim_t)
                viewer.log_state(state)

                viewer.log_shapes(
                    "/demo/robot_target",
                    newton.GeoType.SPHERE,
                    float(meta["robot_radius"]),
                    wp.array(
                        [
                            wp.transform(
                                wp.vec3(*sim_data["robot_target_pos"][sim_idx].astype(np.float32).tolist()),
                                wp.quat_identity(),
                            )
                        ],
                        dtype=wp.transform,
                        device=device,
                    ),
                    wp.array([wp.vec3(1.0, 0.84, 0.18)], dtype=wp.vec3, device=device),
                )
                viewer.log_shapes(
                    "/demo/anchors",
                    newton.GeoType.SPHERE,
                    float(meta["robot_radius"]) * 0.45,
                    anchor_xforms_wp,
                    anchor_colors_wp,
                )

                if rope_edges.size and rope_line_starts_wp is not None and rope_line_ends_wp is not None:
                    q_obj = sim_data["particle_q_object"][sim_idx]
                    rope_line_starts_wp.assign(q_obj[rope_edges[:, 0]].astype(np.float32, copy=False))
                    rope_line_ends_wp.assign(q_obj[rope_edges[:, 1]].astype(np.float32, copy=False))
                    viewer.log_lines(
                        "/demo/rope_springs",
                        rope_line_starts_wp,
                        rope_line_ends_wp,
                        (0.62, 0.84, 0.98),
                        width=float(args.rope_line_width),
                        hidden=False,
                    )

                viewer.end_frame()
                frame = viewer.get_frame(render_ui=False).numpy()
                if args.overlay_label:
                    robot_pos = sim_data["body_q"][sim_idx, int(meta["robot_body"]), :3]
                    target_pos = sim_data["robot_target_pos"][sim_idx]
                    tracking_err = float(np.linalg.norm(robot_pos - target_pos))
                    speed = float(np.linalg.norm(sim_data["body_vel"][sim_idx, int(meta["robot_body"])]))
                    frame = overlay_text_lines_rgb(
                        frame,
                        [
                            "Active rigid pusher -> hanging rope",
                            f"tracking err: {tracking_err:.3f} m | robot speed: {speed:.3f} m/s",
                            f"rope total mass: {float(meta['total_object_mass']):.3g} kg",
                        ],
                        font_size=int(args.label_font_size),
                    )
                ffmpeg_proc.stdin.write(frame.tobytes())
                if (out_idx + 1) % max(1, int(fps_out)) == 0:
                    print(f"  rendered {out_idx + 1}/{len(render_indices)} frames", flush=True)

            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            ffmpeg_proc = None
    finally:
        if ffmpeg_proc is not None:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        viewer.close()
    return out_mp4


def build_summary(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    out_mp4: Path,
) -> dict[str, Any]:
    particle_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    body_vel = np.asarray(sim_data["body_vel"], dtype=np.float32)
    target_pos = np.asarray(sim_data["robot_target_pos"], dtype=np.float32)

    robot_body = int(meta["robot_body"])
    robot_pos = body_q[:, robot_body, :3]
    tracking_error = np.linalg.norm(robot_pos - target_pos, axis=1)

    rope_com = particle_q.mean(axis=1)
    rope_com_disp = float(np.linalg.norm(rope_com[-1] - rope_com[0]))

    robot_radius = float(meta["robot_radius"])
    particle_radius = model.particle_radius.numpy().astype(np.float32)[: particle_q.shape[1]]
    contact_frames = []
    min_clearance = []
    for frame_idx in range(particle_q.shape[0]):
        d = np.linalg.norm(particle_q[frame_idx] - robot_pos[frame_idx][None, :], axis=1) - (
            particle_radius + robot_radius
        )
        min_d = float(np.min(d))
        min_clearance.append(min_d)
        if min_d <= 0.0:
            contact_frames.append(frame_idx)

    first_contact_frame = int(contact_frames[0]) if contact_frames else None
    frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    return {
        "ir_path": str(args.ir.resolve()),
        "output_mp4": str(out_mp4),
        "frames": int(particle_q.shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(frame_dt),
        "wall_time_sec": float(sim_data["wall_time"]),
        "rope_total_mass": float(meta["total_object_mass"]),
        "anchor_count": int(meta["anchor_indices"].shape[0]),
        "robot_mass": float(args.robot_mass),
        "robot_radius": float(meta["robot_radius"]),
        "robot_force_limit": float(args.robot_force_limit),
        "robot_kp": float(args.robot_kp),
        "robot_kd": float(args.robot_kd),
        "tracking_error_mean_m": float(np.mean(tracking_error)),
        "tracking_error_max_m": float(np.max(tracking_error)),
        "robot_speed_max_m_s": float(np.max(np.linalg.norm(body_vel[:, robot_body, :], axis=1))),
        "rope_com_displacement_m": rope_com_disp,
        "rope_com_z_min_m": float(np.min(rope_com[:, 2])),
        "first_contact_frame": first_contact_frame,
        "contact_active_frames": int(len(contact_frames)),
        "min_clearance_min_m": float(np.min(np.asarray(min_clearance, dtype=np.float32))),
        "min_clearance_final_m": float(min_clearance[-1]),
    }


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    summary_path = args.out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.device)

    model, ir_obj, meta, n_obj = build_model(args, device)
    sim_data = simulate(model, ir_obj, meta, args, n_obj, device)
    out_mp4 = render_video(model, sim_data, meta, args, device)
    summary = build_summary(model, sim_data, meta, args, out_mp4)
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
