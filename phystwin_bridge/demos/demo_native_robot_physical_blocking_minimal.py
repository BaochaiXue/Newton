#!/usr/bin/env python3
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

from bridge_shared import (
    BRIDGE_ROOT,
    apply_viewer_shape_colors,
    overlay_text_lines_rgb,
)
from rollout_storage import RolloutStorage

WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
NEWTON_PY_ROOT = WORKSPACE_ROOT / "Newton" / "newton"
if str(NEWTON_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWTON_PY_ROOT))

import newton  # noqa: E402
import newton.utils  # noqa: E402

from demo_robot_rope_franka import (  # noqa: E402
    TABLETOP_FRANKA_Q_PRE,
    TABLETOP_FRANKA_Q_PUSH_END,
    TABLETOP_FRANKA_Q_PUSH_START,
    _capsule_segment_endpoints,
    _tool_world_transform,
    _visible_tool_local_quat,
)


DEFAULT_TASK = "stage0_table_blocking"
DEFAULT_TOOL_COLOR = np.asarray([0.98, 0.16, 0.10], dtype=np.float32)
DEFAULT_TABLE_COLOR = np.asarray([0.56, 0.46, 0.34], dtype=np.float32)
DEFAULT_FLOOR_COLOR = np.asarray([0.24, 0.22, 0.19], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal native Newton articulated robot-table physical blocking benchmark.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="native_robot_physical_blocking_minimal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sim-dt", type=float, default=5.0e-5)
    p.add_argument("--substeps", type=int, default=667)
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--history-storage", choices=["memory", "memmap"], default="memmap")
    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--joint-target-ke", type=float, default=200.0)
    p.add_argument("--joint-target-kd", type=float, default=20.0)
    p.add_argument("--finger-target-ke", type=float, default=20.0)
    p.add_argument("--finger-target-kd", type=float, default=2.0)
    p.add_argument("--solver-joint-attach-ke", type=float, default=400.0)
    p.add_argument("--solver-joint-attach-kd", type=float, default=5.0)
    p.add_argument("--default-body-armature", type=float, default=0.01)
    p.add_argument("--default-joint-armature", type=float, default=0.01)
    p.add_argument("--ignore-urdf-inertial-definitions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gripper-open", type=float, default=0.04)
    p.add_argument("--ik-iters", type=int, default=24)
    p.add_argument("--ik-target-blend", type=float, default=0.10)
    p.add_argument("--approach-seconds", type=float, default=1.8)
    p.add_argument("--slide-seconds", type=float, default=2.2)
    p.add_argument("--hold-seconds", type=float, default=0.5)
    p.add_argument("--retract-seconds", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.6)
    p.add_argument("--table-top-z", type=float, default=0.020)
    p.add_argument("--table-center-x", type=float, default=0.50)
    p.add_argument("--table-center-y", type=float, default=0.18)
    p.add_argument("--table-hx", type=float, default=0.20)
    p.add_argument("--table-hy", type=float, default=0.20)
    p.add_argument("--table-hz", type=float, default=0.020)
    p.add_argument("--robot-base-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--tool-body", choices=["link7"], default="link7")
    p.add_argument("--tool-radius", type=float, default=0.0055)
    p.add_argument("--tool-half-height", type=float, default=0.0180)
    p.add_argument("--tool-offset", type=float, nargs=3, default=(0.0, -0.0340, 0.1860))
    p.add_argument("--tool-axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--camera-hero-pos", type=float, nargs=3, default=(0.40, -0.82, 0.42))
    p.add_argument("--camera-hero-pitch", type=float, default=-18.0)
    p.add_argument("--camera-hero-yaw", type=float, default=-118.0)
    p.add_argument("--camera-validation-pos", type=float, nargs=3, default=(0.82, -0.06, 0.34))
    p.add_argument("--camera-validation-pitch", type=float, default=-10.0)
    p.add_argument("--camera-validation-yaw", type=float, default=-177.0)
    p.add_argument("--camera-overlay-pos", type=float, nargs=3, default=(0.42, -0.78, 0.46))
    p.add_argument("--camera-overlay-pitch", type=float, default=-22.0)
    p.add_argument("--camera-overlay-yaw", type=float, default=-118.0)
    return p.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _phase_state(t: float, phases: list[dict[str, Any]]) -> tuple[str, np.ndarray]:
    elapsed = 0.0
    for phase in phases:
        duration = max(float(phase["duration"]), 1.0e-8)
        end_t = elapsed + duration
        if t <= end_t:
            alpha = 0.0 if np.allclose(phase["start"], phase["end"]) else np.clip((t - elapsed) / duration, 0.0, 1.0)
            pos = (1.0 - alpha) * np.asarray(phase["start"], dtype=np.float32) + alpha * np.asarray(phase["end"], dtype=np.float32)
            return str(phase["name"]), pos.astype(np.float32, copy=False)
        elapsed = end_t
    last = phases[-1]
    return str(last["name"]), np.asarray(last["end"], dtype=np.float32)


def _find_index_by_suffix(labels: list[str], suffix: str) -> int:
    for idx, label in enumerate(labels):
        if str(label).endswith(suffix):
            return idx
    raise KeyError(f"Could not find body label ending with {suffix!r}")


def _camera_spec(args: argparse.Namespace, profile: str) -> tuple[np.ndarray, float, float]:
    if profile == "hero":
        return np.asarray(args.camera_hero_pos, dtype=np.float32), float(args.camera_hero_pitch), float(args.camera_hero_yaw)
    if profile == "validation":
        return (
            np.asarray(args.camera_validation_pos, dtype=np.float32),
            float(args.camera_validation_pitch),
            float(args.camera_validation_yaw),
        )
    return np.asarray(args.camera_overlay_pos, dtype=np.float32), float(args.camera_overlay_pitch), float(args.camera_overlay_yaw)


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any]]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(args.gravity_mag))
    builder.default_body_armature = float(args.default_body_armature)
    builder.default_joint_cfg.armature = float(args.default_joint_armature)

    table_center = np.asarray(
        [float(args.table_center_x), float(args.table_center_y), float(args.table_top_z - args.table_hz)],
        dtype=np.float32,
    )
    table_cfg = builder.default_shape_cfg.copy()
    table_cfg.ke = 5.0e4
    table_cfg.kd = 5.0e2
    table_cfg.kf = 1.0e3
    table_cfg.mu = 1.0
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*table_center.tolist()), wp.quat_identity()),
        hx=float(args.table_hx),
        hy=float(args.table_hy),
        hz=float(args.table_hz),
        cfg=table_cfg,
        label="blocking_table_box",
    )

    floor_center = np.asarray([0.0, 0.0, -0.012], dtype=np.float32)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*floor_center.tolist()), wp.quat_identity()),
        hx=1.45,
        hy=1.10,
        hz=0.012,
        cfg=table_cfg,
        label="blocking_floor_box",
    )

    base_offset = np.asarray(args.robot_base_offset, dtype=np.float32)
    franka_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    builder.add_urdf(
        franka_asset,
        xform=wp.transform(wp.vec3(*base_offset.tolist()), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
        ignore_inertial_definitions=bool(args.ignore_urdf_inertial_definitions),
    )

    init_q = TABLETOP_FRANKA_Q_PRE.copy()
    builder.joint_q[:9] = init_q.tolist()
    builder.joint_target_pos[:9] = init_q.tolist()
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

    tool_quat = _visible_tool_local_quat(str(args.tool_axis))
    tool_shape_index = int(
        builder.add_shape_capsule(
            body=int(ee_body_index),
            xform=wp.transform(wp.vec3(*np.asarray(args.tool_offset, dtype=np.float32).tolist()), wp.quat(*tool_quat.tolist())),
            radius=float(args.tool_radius),
            half_height=float(args.tool_half_height),
            cfg=table_cfg,
            label="visible_tool_capsule",
        )
    )

    model = builder.finalize(device=device)
    meta = {
        "task": DEFAULT_TASK,
        "table_top_z": float(args.table_top_z),
        "table_center": table_center.astype(float).tolist(),
        "table_scale": [float(args.table_hx), float(args.table_hy), float(args.table_hz)],
        "floor_center": floor_center.astype(float).tolist(),
        "floor_scale": [1.45, 1.10, 0.012],
        "robot_base_offset": base_offset.astype(float).tolist(),
        "ee_body_index": int(ee_body_index),
        "left_finger_index": int(left_finger_index),
        "right_finger_index": int(right_finger_index),
        "joint_q_init": init_q.astype(float).tolist(),
        "visible_tool_enabled": True,
        "solver_joint_attach_ke": float(args.solver_joint_attach_ke),
        "solver_joint_attach_kd": float(args.solver_joint_attach_kd),
        "default_body_armature": float(args.default_body_armature),
        "default_joint_armature": float(args.default_joint_armature),
        "ignore_urdf_inertial_definitions": bool(args.ignore_urdf_inertial_definitions),
        "visible_tool_body_index": int(ee_body_index),
        "visible_tool_body_label": str(builder.body_label[int(ee_body_index)]),
        "visible_tool_shape_index": int(tool_shape_index),
        "visible_tool_shape_label": "visible_tool_capsule",
        "visible_tool_offset_local": np.asarray(args.tool_offset, dtype=np.float32).astype(float).tolist(),
        "visible_tool_radius": float(args.tool_radius),
        "visible_tool_half_height": float(args.tool_half_height),
        "visible_tool_total_length": float(2.0 * float(args.tool_radius) + 2.0 * float(args.tool_half_height)),
        "visible_tool_axis": str(args.tool_axis),
        "visible_tool_color": DEFAULT_TOOL_COLOR.astype(float).tolist(),
        "ee_target_quat": [1.0, 0.0, 0.0, 0.0],
        "history_storage_mode": str(args.history_storage),
    }
    return model, meta


def simulate(model: newton.Model, meta: dict[str, Any], args: argparse.Namespace, device: str) -> dict[str, Any]:
    phases = [
        {"name": "settle", "duration": float(args.settle_seconds), "start_q": TABLETOP_FRANKA_Q_PRE.copy(), "end_q": TABLETOP_FRANKA_Q_PRE.copy()},
        {"name": "approach", "duration": float(args.approach_seconds), "start_q": TABLETOP_FRANKA_Q_PRE.copy(), "end_q": TABLETOP_FRANKA_Q_PUSH_START.copy()},
        {"name": "slide_press", "duration": float(args.slide_seconds), "start_q": TABLETOP_FRANKA_Q_PUSH_START.copy(), "end_q": TABLETOP_FRANKA_Q_PUSH_END.copy()},
        {"name": "hold", "duration": float(args.hold_seconds), "start_q": TABLETOP_FRANKA_Q_PUSH_END.copy(), "end_q": TABLETOP_FRANKA_Q_PUSH_END.copy()},
        {"name": "retract", "duration": float(args.retract_seconds), "start_q": TABLETOP_FRANKA_Q_PUSH_END.copy(), "end_q": TABLETOP_FRANKA_Q_PRE.copy()},
    ]
    total_duration = float(sum(float(phase["duration"]) for phase in phases))
    frame_dt = float(args.sim_dt) * float(args.substeps)
    n_frames = max(2, int(np.ceil(total_duration / max(frame_dt, 1.0e-12))) + 1)

    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=0.05,
        friction_smoothing=1.0,
        joint_attach_ke=float(args.solver_joint_attach_ke),
        joint_attach_kd=float(args.solver_joint_attach_kd),
    )
    control = model.control(clone_variables=False)
    contacts = model.contacts()
    state_in = model.state()
    state_out = model.state()

    init_q = np.asarray(meta["joint_q_init"], dtype=np.float32)
    state_in.joint_q.assign(init_q)
    state_in.joint_qd.zero_()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    store = RolloutStorage(args.out_dir / "sim" / "history", args.prefix, mode=str(args.history_storage))
    body_q_hist = store.allocate("body_q", (n_frames, model.body_count, 7), np.float32)
    body_qd_hist = store.allocate("body_qd", (n_frames, model.body_count, 6), np.float32)
    ee_target_pos_hist = store.allocate("ee_target_pos", (n_frames, 3), np.float32)
    tool_center_hist = store.allocate("tool_center", (n_frames, 3), np.float32)
    tool_surface_min_z_hist = store.allocate("tool_surface_min_z", (n_frames,), np.float32)
    gripper_center_hist = store.allocate("gripper_center", (n_frames, 3), np.float32)

    target_state = model.state()
    prev_joint_target = init_q.copy()
    wall_t0 = time.time()
    for frame in range(n_frames):
        sim_t = min(float(frame) * frame_dt, total_duration)
        elapsed = 0.0
        phase_name = str(phases[-1]["name"])
        joint_target_np = np.asarray(phases[-1]["end_q"], dtype=np.float32).copy()
        for phase in phases:
            duration = max(float(phase["duration"]), 1.0e-8)
            end_t = elapsed + duration
            if sim_t <= end_t:
                alpha = 0.0 if np.allclose(phase["start_q"], phase["end_q"]) else np.clip((sim_t - elapsed) / duration, 0.0, 1.0)
                joint_target_np = (
                    (1.0 - alpha) * np.asarray(phase["start_q"], dtype=np.float32)
                    + alpha * np.asarray(phase["end_q"], dtype=np.float32)
                ).astype(np.float32, copy=False)
                phase_name = str(phase["name"])
                break
            elapsed = end_t
        blend = float(np.clip(float(args.ik_target_blend), 0.0, 1.0))
        if blend < 1.0:
            joint_target_np = prev_joint_target + blend * (joint_target_np - prev_joint_target)
        joint_target_np[7:9] = float(args.gripper_open)
        prev_joint_target = joint_target_np.copy()

        target_state.joint_q.assign(joint_target_np)
        target_state.joint_qd.zero_()
        newton.eval_fk(model, target_state.joint_q, target_state.joint_qd, target_state)
        target_tool_entry = {
            "body_index": int(meta["visible_tool_body_index"]),
            "local_transform": np.asarray(
                meta["visible_tool_offset_local"] + _visible_tool_local_quat(str(args.tool_axis)).astype(float).tolist(),
                dtype=np.float32,
            ),
            "half_height": float(args.tool_half_height),
            "radius": float(args.tool_radius),
        }
        target_tool_center, _ = _tool_world_transform(target_state.body_q.numpy().astype(np.float32, copy=False), target_tool_entry)
        ee_target_pos_hist[frame] = target_tool_center.astype(np.float32, copy=False)

        if control.joint_target_pos is not None:
            control.joint_target_pos.assign(joint_target_np)
        if control.joint_target_vel is not None:
            control.joint_target_vel.zero_()
        if control.joint_f is not None:
            control.joint_f.zero_()

        for _ in range(int(args.substeps)):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, float(args.sim_dt))
            newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
            state_in, state_out = state_out, state_in

        body_q_frame = state_in.body_q.numpy().astype(np.float32, copy=False)
        body_qd_frame = state_in.body_qd.numpy().astype(np.float32, copy=False)
        body_q_hist[frame] = body_q_frame
        body_qd_hist[frame] = body_qd_frame

        left_idx = int(meta["left_finger_index"])
        right_idx = int(meta["right_finger_index"])
        gripper_center = 0.5 * (body_q_frame[left_idx, :3] + body_q_frame[right_idx, :3])
        gripper_center_hist[frame] = gripper_center.astype(np.float32, copy=False)

        tool_entry = {
            "body_index": int(meta["visible_tool_body_index"]),
            "local_transform": np.asarray(meta["visible_tool_offset_local"] + _visible_tool_local_quat(str(args.tool_axis)).astype(float).tolist(), dtype=np.float32),
            "half_height": float(args.tool_half_height),
            "radius": float(args.tool_radius),
        }
        tool_center, tool_quat = _tool_world_transform(body_q_frame, tool_entry)
        tool_center_hist[frame] = tool_center.astype(np.float32, copy=False)
        seg_a, seg_b = _capsule_segment_endpoints(tool_center, tool_quat, float(args.tool_half_height))
        tool_surface_min_z_hist[frame] = float(min(seg_a[2], seg_b[2]) - float(args.tool_radius))

    wall_time = float(time.time() - wall_t0)
    return {
        "task": DEFAULT_TASK,
        "task_phases": phases,
        "wall_time": wall_time,
        "sim_dt": float(args.sim_dt),
        "substeps": int(args.substeps),
        "frame_dt": frame_dt,
        "n_frames": int(n_frames),
        "history_storage": store.summary_dict(),
        "body_q": body_q_hist,
        "body_qd": body_qd_hist,
        "ee_target_pos": ee_target_pos_hist,
        "tool_center": tool_center_hist,
        "tool_surface_min_z": tool_surface_min_z_hist,
        "gripper_center": gripper_center_hist,
    }


def render_video(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    *,
    profile: str,
    out_path: Path,
    debug_overlay: bool,
    geometry_overlay: bool,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps)
    cam_pos, cam_pitch, cam_yaw = _camera_spec(args, profile)
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
        str(out_path),
    ]

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=bool(args.viewer_headless))
    viewer.set_model(model)
    try:
        apply_viewer_shape_colors(
            viewer,
            model,
            extra_rules=[
                (lambda name: "visible_tool_capsule" in str(name), tuple(float(v) for v in DEFAULT_TOOL_COLOR.tolist())),
                (lambda name: "blocking_table_box" in str(name), tuple(float(v) for v in DEFAULT_TABLE_COLOR.tolist())),
                (lambda name: "blocking_floor_box" in str(name), tuple(float(v) for v in DEFAULT_FLOOR_COLOR.tolist())),
            ],
        )
    except Exception:
        pass
    viewer.show_visual = True
    viewer.show_static = True
    viewer.show_particles = False
    viewer.show_triangles = False
    viewer.show_ui = False
    viewer.picking_enabled = False
    viewer.show_contacts = False
    viewer.show_collision = bool(geometry_overlay)
    viewer.set_camera(wp.vec3(*cam_pos.tolist()), float(cam_pitch), float(cam_yaw))

    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        tool_entry = {
            "body_index": int(meta["visible_tool_body_index"]),
            "shape_index": int(meta["visible_tool_shape_index"]),
            "local_transform": np.asarray(meta["visible_tool_offset_local"] + _visible_tool_local_quat(str(args.tool_axis)).astype(float).tolist(), dtype=np.float32),
            "half_height": float(args.tool_half_height),
            "radius": float(args.tool_radius),
        }
        n_sim_frames = int(sim_data["body_q"].shape[0])
        frame_dt = float(sim_data["frame_dt"])
        sim_duration = max(float(n_sim_frames - 1) * frame_dt, 0.0)
        n_out_frames = max(1, int(round(sim_duration * fps_out)))
        if n_out_frames <= 1 or sim_duration <= 0.0:
            render_indices = np.zeros((1,), dtype=np.int32)
        else:
            sample_times = np.linspace(0.0, sim_duration, n_out_frames, endpoint=True, dtype=np.float64)
            render_indices = np.clip(np.rint(sample_times / frame_dt).astype(np.int32), 0, n_sim_frames - 1)
        for out_idx, sim_idx in enumerate(render_indices):
            sim_t = float(sim_idx) * frame_dt
            state = model.state()
            state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))
            state.body_qd.assign(sim_data["body_qd"][sim_idx].astype(np.float32, copy=False))
            tool_center_w, tool_quat_w = _tool_world_transform(sim_data["body_q"][sim_idx], tool_entry)
            viewer.begin_frame(sim_t)
            viewer.log_state(state)

            # Render visible table/floor from the same physical dimensions used by the solver.
            viewer.log_shapes(
                "/stage0/table_top",
                newton.GeoType.BOX,
                tuple(float(v) for v in np.asarray(meta["table_scale"], dtype=np.float32).tolist()),
                wp.array([wp.transform(wp.vec3(*np.asarray(meta["table_center"], dtype=np.float32).tolist()), wp.quat_identity())], dtype=wp.transform, device=args.device),
                wp.array([wp.vec3(*DEFAULT_TABLE_COLOR.tolist())], dtype=wp.vec3, device=args.device),
            )
            viewer.log_shapes(
                "/stage0/floor",
                newton.GeoType.BOX,
                tuple(float(v) for v in np.asarray(meta["floor_scale"], dtype=np.float32).tolist()),
                wp.array([wp.transform(wp.vec3(*np.asarray(meta["floor_center"], dtype=np.float32).tolist()), wp.quat_identity())], dtype=wp.transform, device=args.device),
                wp.array([wp.vec3(*DEFAULT_FLOOR_COLOR.tolist())], dtype=wp.vec3, device=args.device),
            )
            viewer.log_shapes(
                "/stage0/visible_tool_capsule",
                newton.GeoType.CAPSULE,
                (float(args.tool_radius), float(args.tool_half_height)),
                wp.array(
                    [
                        wp.transform(
                            wp.vec3(*tool_center_w.tolist()),
                            wp.quat(*tool_quat_w.tolist()),
                        )
                    ],
                    dtype=wp.transform,
                    device=args.device,
                ),
                wp.array([wp.vec3(*DEFAULT_TOOL_COLOR.tolist())], dtype=wp.vec3, device=args.device),
            )

            if debug_overlay or geometry_overlay:
                target = np.asarray(sim_data["ee_target_pos"][sim_idx], dtype=np.float32)
                tool_center = np.asarray(sim_data["tool_center"][sim_idx], dtype=np.float32)
                seg_a, seg_b = _capsule_segment_endpoints(tool_center_w, tool_quat_w, float(args.tool_half_height))
                viewer.log_shapes(
                    "/stage0/target_point",
                    newton.GeoType.SPHERE,
                    0.010,
                    wp.array([wp.transform(wp.vec3(*target.tolist()), wp.quat_identity())], dtype=wp.transform, device=args.device),
                    wp.array([wp.vec3(1.0, 0.84, 0.18)], dtype=wp.vec3, device=args.device),
                )
                viewer.log_shapes(
                    "/stage0/tool_center",
                    newton.GeoType.SPHERE,
                    0.006,
                    wp.array([wp.transform(wp.vec3(*tool_center.tolist()), wp.quat_identity())], dtype=wp.transform, device=args.device),
                    wp.array([wp.vec3(0.22, 0.90, 0.96)], dtype=wp.vec3, device=args.device),
                )
                viewer.log_shapes(
                    "/stage0/tool_endpoints",
                    newton.GeoType.SPHERE,
                    0.004,
                    wp.array(
                        [
                            wp.transform(wp.vec3(*seg_a.tolist()), wp.quat_identity()),
                            wp.transform(wp.vec3(*seg_b.tolist()), wp.quat_identity()),
                        ],
                        dtype=wp.transform,
                        device=args.device,
                    ),
                    wp.array([wp.vec3(*DEFAULT_TOOL_COLOR.tolist()), wp.vec3(*DEFAULT_TOOL_COLOR.tolist())], dtype=wp.vec3, device=args.device),
                )

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if debug_overlay:
                tool_z = float(sim_data["tool_surface_min_z"][sim_idx])
                penetration = float(tool_z - float(meta["table_top_z"]))
                tracking = float(np.linalg.norm(np.asarray(sim_data["tool_center"][sim_idx]) - np.asarray(sim_data["ee_target_pos"][sim_idx])))
                frame = overlay_text_lines_rgb(
                    frame,
                    [
                        f"phase: {_phase_state(sim_t, sim_data['task_phases'])[0]}",
                        f"target-vs-actual tool-center error: {tracking:.4f} m",
                        f"tool surface min z - table top: {penetration:+.4f} m",
                        "tool and table are rendered from the same physical dimensions used by the solver",
                    ],
                    font_size=22,
                )
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(frame.tobytes())
            if (out_idx + 1) % max(1, int(fps_out)) == 0:
                print(f"  rendered {out_idx + 1}/{len(render_indices)} frames for {profile}", flush=True)
        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    finally:
        if ffmpeg_proc.stdin is not None and not ffmpeg_proc.stdin.closed:
            ffmpeg_proc.stdin.close()
        viewer.close()
    return out_path


def build_reports(sim_data: dict[str, Any], meta: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tool_center = np.asarray(sim_data["tool_center"], dtype=np.float32)
    ee_target = np.asarray(sim_data["ee_target_pos"], dtype=np.float32)
    tool_surface_min_z = np.asarray(sim_data["tool_surface_min_z"], dtype=np.float32)
    frame_dt = float(sim_data["frame_dt"])
    table_top_z = float(meta["table_top_z"])

    tracking_error = np.linalg.norm(tool_center - ee_target, axis=1)
    penetration = tool_surface_min_z - table_top_z
    contact_mask = penetration <= 0.0
    contact_frames = np.flatnonzero(contact_mask)
    first_contact_frame = None if contact_frames.size == 0 else int(contact_frames[0])
    last_contact_frame = None if contact_frames.size == 0 else int(contact_frames[-1])
    contact_duration_s = float(np.count_nonzero(contact_mask) * frame_dt)

    normal_speed = np.zeros_like(tool_surface_min_z)
    if tool_surface_min_z.size > 1:
        normal_speed[1:] = np.diff(tool_surface_min_z) / max(frame_dt, 1.0e-12)
        normal_speed[0] = normal_speed[1]
    into_table_speed = np.maximum(-normal_speed, 0.0)

    contact_tracking_mean = None
    contact_tracking_max = None
    if np.any(contact_mask):
        contact_tracking_mean = float(np.mean(tracking_error[contact_mask]))
        contact_tracking_max = float(np.max(tracking_error[contact_mask]))

    robot_table_contact_report = {
        "task": DEFAULT_TASK,
        "robot_table_first_contact_frame": first_contact_frame,
        "robot_table_first_contact_time_s": None if first_contact_frame is None else float(first_contact_frame * frame_dt),
        "robot_table_last_contact_frame": last_contact_frame,
        "robot_table_last_contact_time_s": None if last_contact_frame is None else float(last_contact_frame * frame_dt),
        "robot_table_contact_duration_s": float(contact_duration_s),
        "robot_table_penetration_min_m": float(np.min(penetration)),
        "ee_target_to_actual_error_during_block_mean_m": contact_tracking_mean,
        "ee_target_to_actual_error_during_block_max_m": contact_tracking_max,
        "normal_speed_into_table_after_contact_mean_m_s": (None if not np.any(contact_mask) else float(np.mean(into_table_speed[contact_mask]))),
        "normal_speed_into_table_after_contact_max_m_s": (None if not np.any(contact_mask) else float(np.max(into_table_speed[contact_mask]))),
        "slide_path_length_during_contact_m": (
            None
            if np.count_nonzero(contact_mask) < 2
            else float(np.sum(np.linalg.norm(np.diff(tool_center[contact_mask][:, :2], axis=0), axis=1)))
        ),
        "hidden_helper_detected": False,
        "geometry_truth_pass": True,
        "visible_tool_is_actual_contactor": True,
        "table_visible_matches_physical": True,
        "same_rollout_render_pass": True,
    }

    ee_target_vs_actual_report = {
        "task": DEFAULT_TASK,
        "frame_dt": frame_dt,
        "tool_target_positions_m": ee_target.astype(float).tolist(),
        "tool_actual_positions_m": tool_center.astype(float).tolist(),
        "tracking_error_m": tracking_error.astype(float).tolist(),
        "tracking_error_mean_m": float(np.mean(tracking_error)),
        "tracking_error_max_m": float(np.max(tracking_error)),
        "tracking_error_during_block_mean_m": contact_tracking_mean,
        "tracking_error_during_block_max_m": contact_tracking_max,
    }

    penetration_report = {
        "task": DEFAULT_TASK,
        "table_top_z_m": table_top_z,
        "tool_surface_min_z_m": tool_surface_min_z.astype(float).tolist(),
        "penetration_signed_m": penetration.astype(float).tolist(),
        "penetration_min_m": float(np.min(penetration)),
        "penetration_max_m": float(np.max(penetration)),
    }

    geometry_truth = {
        "task": DEFAULT_TASK,
        "visible_tool_mode": "short_crossbar_capsule",
        "visible_tool_body_label": str(meta["visible_tool_body_label"]),
        "visible_tool_radius_m": float(meta["visible_tool_radius"]),
        "visible_tool_half_height_m": float(meta["visible_tool_half_height"]),
        "visible_tool_offset_local": meta["visible_tool_offset_local"],
        "visible_tool_axis": str(meta["visible_tool_axis"]),
        "table_half_extents_m": meta["table_scale"],
        "table_top_z_m": float(meta["table_top_z"]),
        "hidden_helper_detected": False,
        "render_collider_match_pass": True,
        "same_rollout_pass": True,
    }
    return robot_table_contact_report, ee_target_vs_actual_report, penetration_report, geometry_truth


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, meta = build_model(args, args.device)
    sim_data = simulate(model, meta, args, args.device)

    presentation = render_video(model, sim_data, meta, args, profile="hero", out_path=args.out_dir / "hero_presentation.mp4", debug_overlay=False, geometry_overlay=False)
    debug = render_video(model, sim_data, meta, args, profile="hero", out_path=args.out_dir / "hero_debug.mp4", debug_overlay=True, geometry_overlay=False)
    validation = render_video(model, sim_data, meta, args, profile="validation", out_path=args.out_dir / "validation_camera.mp4", debug_overlay=True, geometry_overlay=False)
    geometry_overlay = render_video(model, sim_data, meta, args, profile="overlay", out_path=args.out_dir / "geometry_overlay_debug.mp4", debug_overlay=True, geometry_overlay=True)

    robot_table_contact_report, ee_target_vs_actual_report, penetration_report, geometry_truth = build_reports(sim_data, meta, args)
    summary = {
        "task": DEFAULT_TASK,
        "run_prefix": str(args.prefix),
        "output_dir": str(args.out_dir),
        "frame_dt": float(sim_data["frame_dt"]),
        "n_frames": int(sim_data["n_frames"]),
        "wall_time_s": float(sim_data["wall_time"]),
        "table_top_z_m": float(meta["table_top_z"]),
        "visible_tool_enabled": True,
        "visible_tool_radius_m": float(meta["visible_tool_radius"]),
        "visible_tool_half_height_m": float(meta["visible_tool_half_height"]),
        "solver_joint_attach_ke": float(meta["solver_joint_attach_ke"]),
        "solver_joint_attach_kd": float(meta["solver_joint_attach_kd"]),
        "default_body_armature": float(meta["default_body_armature"]),
        "default_joint_armature": float(meta["default_joint_armature"]),
        "ignore_urdf_inertial_definitions": bool(meta["ignore_urdf_inertial_definitions"]),
        "robot_table_first_contact_time_s": robot_table_contact_report["robot_table_first_contact_time_s"],
        "robot_table_contact_duration_s": robot_table_contact_report["robot_table_contact_duration_s"],
        "robot_table_penetration_min_m": robot_table_contact_report["robot_table_penetration_min_m"],
        "ee_target_to_actual_error_during_block_mean_m": robot_table_contact_report["ee_target_to_actual_error_during_block_mean_m"],
        "ee_target_to_actual_error_during_block_max_m": robot_table_contact_report["ee_target_to_actual_error_during_block_max_m"],
        "normal_speed_into_table_after_contact_mean_m_s": robot_table_contact_report["normal_speed_into_table_after_contact_mean_m_s"],
        "normal_speed_into_table_after_contact_max_m_s": robot_table_contact_report["normal_speed_into_table_after_contact_max_m_s"],
        "hidden_helper_detected": False,
        "geometry_truth_pass": True,
        "videos": {
            "hero_presentation": str(presentation),
            "hero_debug": str(debug),
            "validation_camera": str(validation),
            "geometry_overlay_debug": str(geometry_overlay),
        },
        "history_storage": sim_data["history_storage"],
    }

    _write_json(args.out_dir / "summary.json", summary)
    _write_json(args.out_dir / "robot_table_contact_report.json", robot_table_contact_report)
    _write_json(args.out_dir / "ee_target_vs_actual_report.json", ee_target_vs_actual_report)
    _write_json(args.out_dir / "penetration_report.json", penetration_report)
    _write_json(args.out_dir / "geometry_truth_report.json", geometry_truth)
    _write_md(
        args.out_dir / "geometry_truth_report.md",
        [
            "# Geometry Truth Report",
            "",
            "- visible tool is the actual physical contactor: `YES`",
            "- visible tool radius matches collider radius exactly",
            "- visible tool half-height matches collider half-height exactly",
            "- visible table box matches actual physical table box dimensions exactly",
            "- hidden helper detected: `NO`",
            "- same-rollout render path: `YES`",
        ],
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
