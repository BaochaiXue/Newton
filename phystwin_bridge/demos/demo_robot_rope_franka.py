#!/usr/bin/env python3
"""Native Newton Franka Panda pushes a PhysTwin rope.

This demo keeps the deformable side on the PhysTwin -> Newton bridge, but
replaces the proxy pusher with a native Newton robotics asset:

- load one rope from PhysTwin IR
- keep it object-only and pin the rope endpoints
- add the native Franka Panda URDF through ``ModelBuilder.add_urdf()``
- drive the end-effector through a short Cartesian push using Newton IK
- solve the combined robot + rope scene with ``SolverSemiImplicit``

The goal is a native robot asset baseline that is easier to defend in a
meeting than a free-floating box / capsule proxy.
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

import newton.ik as ik
import newton.utils

from demo_robot_rope import (
    _anchor_particle_indices,
    _quat_conjugate,
    _quat_multiply,
    _rope_endpoints,
    _validate_scaling_args,
    _maybe_autoset_mass_spring_scale,
    _effective_spring_scales,
)
from demo_rope_bunny_drop import (
    _apply_drag_correction_ignore_axis,
    _copy_object_only_ir,
    load_ir,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
)
from demo_shared import compute_visual_particle_radii, temporary_particle_radius_override

BRIDGE_ROOT = Path(__file__).resolve().parents[1]

FRANKA_INIT_Q = np.asarray(
    [
        -3.6802115e-03,
        2.3901723e-02,
        3.6804110e-03,
        -2.3683236e00,
        -1.2918962e-04,
        2.3922248e00,
        7.8549200e-01,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)


def _default_rope_ir() -> Path:
    return BRIDGE_ROOT / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native Newton Franka Panda pushes a hanging PhysTwin rope."
    )
    p.add_argument("--ir", type=Path, default=_default_rope_ir(), help="Path to rope PhysTwin IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_push_rope_franka")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=800)
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

    p.add_argument("--anchor-height", type=float, default=0.72)
    p.add_argument("--anchor-count-per-end", type=int, default=1)

    p.add_argument(
        "--robot-base-offset",
        type=float,
        nargs=3,
        default=(-0.58, 0.0, -0.32),
        metavar=("X", "Y", "Z"),
        help="Franka base offset from the rope center [m].",
    )
    p.add_argument("--ee-start-x-offset", type=float, default=-0.18)
    p.add_argument("--ee-end-x-offset", type=float, default=0.06)
    p.add_argument("--ee-z-offset", type=float, default=-0.02)
    p.add_argument("--ee-contact-radius", type=float, default=0.055)
    p.add_argument("--ik-iters", type=int, default=24)
    p.add_argument("--joint-target-ke", type=float, default=400.0)
    p.add_argument("--joint-target-kd", type=float, default=40.0)
    p.add_argument("--finger-target-ke", type=float, default=100.0)
    p.add_argument("--finger-target-kd", type=float, default=10.0)
    p.add_argument("--gripper-open", type=float, default=0.04)
    p.add_argument("--settle-seconds", type=float, default=0.02)
    p.add_argument("--push-seconds", type=float, default=0.08)
    p.add_argument("--hold-seconds", type=float, default=0.16)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=8.0)
    p.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gif-fps", type=float, default=10.0)
    p.add_argument("--gif-width", type=int, default=960)
    p.add_argument("--gif-max-colors", type=int, default=128)
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(-0.90, 0.55, 0.95),
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=-8.0)
    p.add_argument("--camera-yaw", type=float, default=-42.0)
    p.add_argument("--camera-fov", type=float, default=34.0)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.004)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=24)
    p.add_argument("--rope-line-width", type=float, default=0.02)
    p.add_argument("--spring-stride", type=int, default=20)
    return p.parse_args()


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    vec_quat = np.asarray([v[0], v[1], v[2], 0.0], dtype=np.float32)
    rotated = _quat_multiply(_quat_multiply(q, vec_quat), _quat_conjugate(q))
    return rotated[:3]


def _ee_world_position(body_q_row: np.ndarray, offset_local: np.ndarray) -> np.ndarray:
    body_q_row = np.asarray(body_q_row, dtype=np.float32)
    pos = body_q_row[:3]
    quat = body_q_row[3:7]
    return pos + _quat_rotate_vector(quat, offset_local)


def _gripper_center_world_position(body_q: np.ndarray, left_finger_idx: int, right_finger_idx: int) -> np.ndarray:
    body_q = np.asarray(body_q, dtype=np.float32)
    left = body_q[int(left_finger_idx), :3]
    right = body_q[int(right_finger_idx), :3]
    return 0.5 * (left + right)


def _robot_target_state(t: float, meta: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(meta["ee_target_start"], dtype=np.float32)
    end = np.asarray(meta["ee_target_end"], dtype=np.float32)
    target_rot = np.asarray(meta["ee_target_quat"], dtype=np.float32)
    settle = float(args.settle_seconds)
    push = max(float(args.push_seconds), 1.0e-8)
    hold = max(float(args.hold_seconds), 0.0)
    if t <= settle:
        return start.copy(), target_rot.copy()
    if t <= settle + push:
        alpha = float((t - settle) / push)
        pos = (1.0 - alpha) * start + alpha * end
        return pos.astype(np.float32), target_rot.copy()
    if t <= settle + push + hold:
        return end.copy(), target_rot.copy()
    return end.copy(), target_rot.copy()


def _find_index_by_suffix(labels: list[str], suffix: str) -> int:
    for idx, label in enumerate(labels):
        if label.endswith(suffix):
            return idx
    raise KeyError(f"Could not find label suffix {suffix!r} in {labels}")


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
    rope_center = shifted_q.mean(axis=0).astype(np.float32, copy=False)

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

    robot_base_pos = rope_center + np.asarray(args.robot_base_offset, dtype=np.float32)
    franka_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    builder.add_urdf(
        franka_asset,
        xform=wp.transform(wp.vec3(*robot_base_pos.tolist()), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )

    builder.joint_q[:9] = FRANKA_INIT_Q.tolist()
    builder.joint_target_pos[:9] = FRANKA_INIT_Q.tolist()
    builder.joint_target_ke[:7] = [float(args.joint_target_ke)] * 7
    builder.joint_target_kd[:7] = [float(args.joint_target_kd)] * 7
    builder.joint_target_ke[7:9] = [float(args.finger_target_ke)] * 2
    builder.joint_target_kd[7:9] = [float(args.finger_target_kd)] * 2
    builder.joint_target_pos[7:9] = [float(args.gripper_open)] * 2
    builder.joint_armature[:7] = [0.3] * 4 + [0.11] * 3
    builder.joint_armature[7:9] = [0.15] * 2
    builder.joint_effort_limit[:7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    builder.joint_effort_limit[7:9] = [20.0, 20.0]

    ee_body_index = _find_index_by_suffix(builder.body_label, "/fr3_link7")
    left_finger_index = _find_index_by_suffix(builder.body_label, "/fr3_leftfinger")
    right_finger_index = _find_index_by_suffix(builder.body_label, "/fr3_rightfinger")
    ee_offset_local = np.asarray([0.0, 0.0, 0.22], dtype=np.float32)
    ee_target_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ee_target_start = rope_center + np.asarray(
        [float(args.ee_start_x_offset), 0.0, float(args.ee_z_offset)], dtype=np.float32
    )
    ee_target_end = rope_center + np.asarray(
        [float(args.ee_end_x_offset), 0.0, float(args.ee_z_offset)], dtype=np.float32
    )

    model = builder.finalize(device=device)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)

    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "robot_geometry": "native_franka",
        "robot_base_pos": robot_base_pos.astype(np.float32),
        "ee_body_index": int(ee_body_index),
        "left_finger_index": int(left_finger_index),
        "right_finger_index": int(right_finger_index),
        "ee_offset_local": ee_offset_local.astype(np.float32),
        "ee_target_quat": ee_target_quat.astype(np.float32),
        "ee_target_start": ee_target_start.astype(np.float32),
        "ee_target_end": ee_target_end.astype(np.float32),
        "joint_q_init": FRANKA_INIT_Q.astype(np.float32),
        "endpoint_indices": endpoint_indices.astype(np.int32),
        "anchor_indices": anchor_indices.astype(np.int32),
        "anchor_positions": shifted_q[anchor_indices].astype(np.float32),
        "rope_center": rope_center.astype(np.float32),
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
    contacts = model.contacts() if newton_import_ir._use_collision_pipeline(cfg, ir_obj) else None
    prev_joint_q = np.asarray(meta["joint_q_init"], dtype=np.float32).copy()
    state_in.joint_q.assign(prev_joint_q)
    state_in.joint_qd.zero_()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

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

    ik_joint_q = wp.array(np.asarray(meta["joint_q_init"], dtype=np.float32).reshape(1, -1), dtype=wp.float32, device=device)
    pos_obj = ik.IKObjectivePosition(
        link_index=int(meta["ee_body_index"]),
        link_offset=wp.vec3(*np.asarray(meta["ee_offset_local"], dtype=np.float32).tolist()),
        target_positions=wp.array([wp.vec3(*np.asarray(meta["ee_target_start"], dtype=np.float32).tolist())], dtype=wp.vec3, device=device),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=int(meta["ee_body_index"]),
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*np.asarray(meta["ee_target_quat"], dtype=np.float32).tolist())], dtype=wp.vec4, device=device),
    )
    joint_limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    ik_solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, joint_limit_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )

    n_frames = max(2, int(args.frames))
    particle_q_all: list[np.ndarray] = []
    particle_q_object: list[np.ndarray] = []
    body_q: list[np.ndarray] = []
    body_vel: list[np.ndarray] = []
    ee_target_pos: list[np.ndarray] = []

    t0 = time.perf_counter()
    for frame in range(n_frames):
        q = state_in.particle_q.numpy().astype(np.float32)
        particle_q_all.append(q.copy())
        particle_q_object.append(q[:n_obj].copy())
        body_q.append(state_in.body_q.numpy().astype(np.float32).copy())
        body_vel.append(state_in.body_qd.numpy().astype(np.float32)[:, :3].copy())
        target_pos_frame, _ = _robot_target_state(float(frame) * sim_dt * float(substeps), meta, args)
        ee_target_pos.append(target_pos_frame.copy())

        for sub in range(substeps):
            sim_t = (float(frame) * float(substeps) + float(sub)) * sim_dt
            target_pos, target_quat = _robot_target_state(sim_t, meta, args)
            pos_obj.set_target_position(0, wp.vec3(*target_pos.tolist()))
            rot_obj.set_target_rotation(0, wp.vec4(*target_quat.tolist()))
            ik_solver.step(ik_joint_q, ik_joint_q, iterations=int(args.ik_iters))

            joint_target_np = ik_joint_q.numpy().reshape(-1).astype(np.float32)
            joint_target_np[7:9] = float(args.gripper_open)
            joint_target_qd = (joint_target_np - prev_joint_q) / sim_dt
            prev_joint_q = joint_target_np.copy()

            state_in.clear_forces()
            state_in.joint_q.assign(joint_target_np)
            state_in.joint_qd.assign(joint_target_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            if contacts is not None:
                model.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, sim_dt)
            state_in, state_out = state_out, state_in
            state_in.joint_q.assign(joint_target_np)
            state_in.joint_qd.assign(joint_target_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

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
        "ee_target_pos": np.stack(ee_target_pos),
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

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=bool(args.viewer_headless))
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
        viewer.set_camera(wp.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])), float(args.camera_pitch), float(args.camera_yaw))

        particle_radius_sim = model.particle_radius.numpy().astype(np.float32)
        render_radii = compute_visual_particle_radii(
            particle_radius_sim,
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
        anchor_colors_wp = wp.array([wp.vec3(0.34, 0.90, 0.52) for _ in range(anchor_positions.shape[0])], dtype=wp.vec3, device=device)
        left_finger_idx = int(meta["left_finger_index"])
        right_finger_idx = int(meta["right_finger_index"])

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
                render_indices = np.clip(np.rint(sample_times / sim_frame_dt).astype(np.int32), 0, n_sim_frames - 1)

            for out_idx, sim_idx in enumerate(render_indices):
                state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
                state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))

                sim_t = float(sim_idx) * sim_frame_dt
                viewer.begin_frame(sim_t)
                viewer.log_state(state)
                viewer.log_shapes(
                    "/demo/ee_target",
                    newton.GeoType.SPHERE,
                    float(args.ee_contact_radius) * 0.55,
                    wp.array(
                        [wp.transform(wp.vec3(*sim_data["ee_target_pos"][sim_idx].astype(np.float32).tolist()), wp.quat_identity())],
                        dtype=wp.transform,
                        device=device,
                    ),
                    wp.array([wp.vec3(1.0, 0.84, 0.18)], dtype=wp.vec3, device=device),
                )
                gripper_center = _gripper_center_world_position(
                    sim_data["body_q"][sim_idx], left_finger_idx, right_finger_idx
                ).astype(np.float32, copy=False)
                viewer.log_shapes(
                    "/demo/gripper_center",
                    newton.GeoType.SPHERE,
                    0.014,
                    wp.array(
                        [wp.transform(wp.vec3(*gripper_center.tolist()), wp.quat_identity())],
                        dtype=wp.transform,
                        device=device,
                    ),
                    wp.array([wp.vec3(0.22, 0.90, 0.96)], dtype=wp.vec3, device=device),
                )
                viewer.log_shapes(
                    "/demo/anchors",
                    newton.GeoType.SPHERE,
                    0.018,
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
                    target_pos = sim_data["ee_target_pos"][sim_idx]
                    tracking_err = float(np.linalg.norm(gripper_center - target_pos))
                    if sim_idx > 0:
                        prev_gripper_center = _gripper_center_world_position(
                            sim_data["body_q"][sim_idx - 1], left_finger_idx, right_finger_idx
                        ).astype(np.float32, copy=False)
                        speed = float(np.linalg.norm(gripper_center - prev_gripper_center) / sim_frame_dt)
                    else:
                        speed = 0.0
                    q_obj = sim_data["particle_q_object"][sim_idx]
                    particle_radius = particle_radius_sim[: q_obj.shape[0]]
                    min_clearance = float(
                        np.min(
                            np.linalg.norm(q_obj - gripper_center[None, :], axis=1)
                            - (particle_radius + float(args.ee_contact_radius))
                        )
                    )
                    contact_state = "ON" if min_clearance <= 0.0 else "OFF"
                    frame = overlay_text_lines_rgb(
                        frame,
                        [
                            "Native Newton Franka Panda + hanging rope | yellow=target, cyan=gripper center",
                            f"tracking err: {tracking_err:.3f} m | gripper speed: {speed:.3f} m/s",
                            f"contact: {contact_state} | approx gripper clearance: {1000.0 * min_clearance:.1f} mm",
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
    target_pos = np.asarray(sim_data["ee_target_pos"], dtype=np.float32)
    particle_radius = model.particle_radius.numpy().astype(np.float32)[: particle_q.shape[1]]

    left_finger_idx = int(meta["left_finger_index"])
    right_finger_idx = int(meta["right_finger_index"])
    gripper_center = np.stack(
        [_gripper_center_world_position(body_q[i], left_finger_idx, right_finger_idx) for i in range(body_q.shape[0])]
    )
    tracking_error = np.linalg.norm(gripper_center - target_pos, axis=1)
    gripper_speed = np.linalg.norm(np.diff(gripper_center, axis=0), axis=1) / float(sim_data["sim_dt"] * sim_data["substeps"])
    if gripper_speed.size == 0:
        gripper_speed = np.zeros((1,), dtype=np.float32)

    rope_com = particle_q.mean(axis=1)
    rope_com_disp = float(np.linalg.norm(rope_com[-1] - rope_com[0]))

    contact_frames = []
    min_clearance = []
    for frame_idx in range(particle_q.shape[0]):
        d = np.linalg.norm(particle_q[frame_idx] - gripper_center[frame_idx][None, :], axis=1) - (
            particle_radius + float(args.ee_contact_radius)
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
        "robot_geometry": "native_franka",
        "ee_body_index": int(meta["ee_body_index"]),
        "gripper_center_tracking_error_mean_m": float(np.mean(tracking_error)),
        "gripper_center_tracking_error_max_m": float(np.max(tracking_error)),
        "gripper_center_speed_max_m_s": float(np.max(gripper_speed)),
        "rope_com_displacement_m": rope_com_disp,
        "rope_com_z_min_m": float(np.min(rope_com[:, 2])),
        "first_contact_frame": first_contact_frame,
        "contact_active_frames": int(len(contact_frames)),
        "min_clearance_min_m": float(np.min(np.asarray(min_clearance, dtype=np.float32))),
        "min_clearance_final_m": float(min_clearance[-1]),
    }


def make_gif(args: argparse.Namespace, out_mp4: Path) -> Path | None:
    if not bool(args.make_gif):
        return None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    out_gif = args.out_dir / f"{args.prefix}.gif"
    palette = args.out_dir / f"{args.prefix}_palette.png"
    width = max(160, int(args.gif_width))
    fps = max(1.0, float(args.gif_fps))
    max_colors = max(16, int(args.gif_max_colors))
    vf = f"fps={fps:.6f},scale={width}:-1:flags=lanczos"

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(out_mp4),
            "-vf",
            f"{vf},palettegen=max_colors={max_colors}:stats_mode=diff",
            str(palette),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(out_mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
            "-loop",
            "0",
            str(out_gif),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        palette.unlink()
    except FileNotFoundError:
        pass
    return out_gif


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
    out_gif = make_gif(args, out_mp4)
    summary = build_summary(model, sim_data, meta, args, out_mp4)
    if out_gif is not None:
        summary["output_gif"] = str(out_gif)
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
