#!/usr/bin/env python3
"""split_v3 Stage-0 robot-first direct-finger table blocking demo.

This demo is intentionally rigid-only:

- native Newton Franka
- native Newton table
- no rope
- no support box
- no visible tool

The purpose is to prove a robot-first native execution path where the table
physically blocks the finger without startup collapse.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import demo_robot_rope_franka as legacy

newton = legacy.newton
JointTargetMode = newton.JointTargetMode
ik = legacy.ik
path_defaults = legacy.path_defaults

TABLETOP_PUSH_TASK = legacy.TABLETOP_PUSH_TASK
FRANKA_INIT_Q = legacy.FRANKA_INIT_Q.copy()

Q_PRE = legacy.TABLETOP_BLOCKING_Q_HIGH_PRE.copy()
Q_APPROACH_SEED = legacy.TABLETOP_FRANKA_Q_PRE.copy()
Q_CONTACT_SEED = legacy.TABLETOP_BLOCKING_Q_PUSH_START.copy()
Q_PUSH_SEED = legacy.TABLETOP_BLOCKING_Q_PUSH_END.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="split_v3 Stage-0 robot-first direct-finger blocking demo.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_rope_tabletop_hero")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--task", choices=[TABLETOP_PUSH_TASK], default=TABLETOP_PUSH_TASK)
    p.add_argument("--blocking-stage", choices=["rigid_only"], default="rigid_only")
    p.add_argument("--tabletop-control-mode", choices=["robot_first_native", "joint_target_drive"], default="robot_first_native")

    p.add_argument("--render-mode", choices=["debug", "presentation"], default="presentation")
    p.add_argument("--camera-profile", choices=["hero", "validation"], default="hero")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=5.0e-5)
    p.add_argument("--substeps", type=int, default=667)
    p.add_argument("--history-storage", choices=["memory", "memmap"], default="memmap")
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gif-width", type=int, default=960)
    p.add_argument("--gif-fps", type=float, default=15.0)
    p.add_argument("--gif-max-colors", type=int, default=256)
    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-pos", nargs=3, type=float, default=None)
    p.add_argument("--camera-pitch", type=float, default=None)
    p.add_argument("--camera-yaw", type=float, default=None)
    p.add_argument("--camera-fov", type=float, default=None)
    p.add_argument("--camera-track-mode", choices=["none", "tabletop_follow"], default="none")
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--label-font-size", type=int, default=22)
    p.add_argument("--particle-radius-vis-scale", type=float, default=None)
    p.add_argument("--particle-radius-vis-min", type=float, default=None)
    p.add_argument("--rope-line-width", type=float, default=0.024)
    p.add_argument("--load-history-from-dir", type=Path, default=None)
    p.add_argument("--load-history-prefix", default=None)

    p.add_argument("--joint-target-ke", type=float, default=650.0)
    p.add_argument("--joint-target-kd", type=float, default=100.0)
    p.add_argument("--finger-target-ke", type=float, default=40.0)
    p.add_argument("--finger-target-kd", type=float, default=10.0)
    p.add_argument("--solver-iterations", type=int, default=15)
    p.add_argument("--solver-ls-iterations", type=int, default=100)
    p.add_argument("--default-body-armature", type=float, default=0.1)
    p.add_argument("--default-joint-armature", type=float, default=0.1)
    p.add_argument("--ignore-urdf-inertial-definitions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gripper-open", type=float, default=0.04)
    p.add_argument("--ik-iters", type=int, default=48)
    p.add_argument("--gravity-mag", type=float, default=9.81)
    p.add_argument("--pre-release-settle-damping-scale", type=float, default=1.0)
    p.add_argument("--ee-contact-radius", type=float, default=0.04)

    p.add_argument("--robot-base-pos", nargs=3, type=float, default=(-0.56, -0.22, 0.10))
    p.add_argument("--table-center", nargs=3, type=float, default=(-0.02, -0.19, 0.18))
    p.add_argument("--table-hx", type=float, default=0.20)
    p.add_argument("--table-hy", type=float, default=0.16)
    p.add_argument("--table-hz", type=float, default=0.02)
    p.add_argument("--ee-yaw-deg", type=float, default=-18.0)

    p.add_argument("--settle-seconds", type=float, default=0.60)
    p.add_argument("--pre-seconds", type=float, default=0.50)
    p.add_argument("--approach-seconds", type=float, default=1.00)
    p.add_argument("--contact-seconds", type=float, default=0.40)
    p.add_argument("--push-seconds", type=float, default=1.20)
    p.add_argument("--retract-seconds", type=float, default=1.20)

    p.add_argument("--contact-depth", type=float, default=0.015)
    p.add_argument("--push-x-shift", type=float, default=0.06)
    p.add_argument("--park-lift-z", type=float, default=0.10)
    p.add_argument("--pre-lift-z", type=float, default=0.0)
    p.add_argument("--approach-lift-z", type=float, default=0.0)
    p.add_argument("--retract-lift-z", type=float, default=0.08)
    return p.parse_args()


def _total_task_duration(args: argparse.Namespace) -> float:
    return float(
        args.settle_seconds
        + args.pre_seconds
        + args.approach_seconds
        + args.contact_seconds
        + args.push_seconds
        + args.retract_seconds
    )


def _resolve_runtime_defaults(args: argparse.Namespace) -> None:
    if args.frames is None:
        frame_dt = float(args.sim_dt) * max(1, int(args.substeps))
        args.frames = max(120, int(np.ceil(_total_task_duration(args) / max(frame_dt, 1.0e-8))) + 1)
    if args.overlay_label is None:
        args.overlay_label = str(args.render_mode) == "debug"


def _target_quat(args: argparse.Namespace) -> np.ndarray:
    down = legacy._quat_from_axis_angle_np(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.pi)
    yaw = legacy._quat_from_axis_angle_np(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32), np.deg2rad(float(args.ee_yaw_deg))
    )
    return legacy._quat_multiply(down, yaw).astype(np.float32, copy=False)


def _measure_link7_gripper_offset(
    model: newton.Model,
    ee_body_index: int,
    left_finger_index: int,
    right_finger_index: int,
) -> np.ndarray:
    state = model.state()
    state.joint_q.assign(FRANKA_INIT_Q.astype(np.float32))
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy().astype(np.float32)
    gripper_center = 0.5 * (
        body_q[int(left_finger_index), :3] + body_q[int(right_finger_index), :3]
    )
    link7_pos = body_q[int(ee_body_index), :3]
    link7_quat = body_q[int(ee_body_index), 3:7]
    return legacy._quat_inverse_rotate_vector(link7_quat, gripper_center - link7_pos).astype(np.float32, copy=False)


def _gripper_center_from_q(
    model: newton.Model,
    joint_q: np.ndarray,
    left_finger_index: int,
    right_finger_index: int,
) -> np.ndarray:
    return legacy._fk_gripper_center_from_joint_q(
        model,
        np.asarray(joint_q, dtype=np.float32),
        left_finger_idx=int(left_finger_index),
        right_finger_idx=int(right_finger_index),
    ).astype(np.float32, copy=False)


def _solve_q_for_pose(
    model: newton.Model,
    ee_body_index: int,
    ee_offset_local: np.ndarray,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    ik_iters: int,
    gripper_open: float,
    device: str,
) -> np.ndarray:
    q_work = wp.array(np.asarray(q_seed, dtype=np.float32).reshape(1, -1), dtype=wp.float32, device=device)
    pos_obj = ik.IKObjectivePosition(
        link_index=int(ee_body_index),
        link_offset=wp.vec3(*np.asarray(ee_offset_local, dtype=np.float32).tolist()),
        target_positions=wp.array([wp.vec3(*np.asarray(target_pos, dtype=np.float32).tolist())], dtype=wp.vec3, device=device),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=int(ee_body_index),
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*np.asarray(target_quat, dtype=np.float32).tolist())], dtype=wp.vec4, device=device),
    )
    joint_limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, joint_limit_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(q_work, q_work, iterations=int(ik_iters))
    q = q_work.numpy().reshape(-1).astype(np.float32)
    q[7:9] = float(gripper_open)
    return q


def _build_joint_phase_targets(
    model: newton.Model,
    meta: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, Any]]:
    ee_body_index = int(meta["ee_body_index"])
    ee_offset_local = np.asarray(meta["ee_offset_local"], dtype=np.float32)
    left_finger_index = int(meta["left_finger_index"])
    right_finger_index = int(meta["right_finger_index"])
    table_center = np.asarray(meta["table_center"], dtype=np.float32)
    table_top_z = float(meta["table_top_z"])
    target_quat = np.asarray(meta["ee_target_quat"], dtype=np.float32)

    q_pre = Q_PRE.copy()
    q_pre[7:9] = float(args.gripper_open)
    pre_pos = _gripper_center_from_q(model, q_pre, left_finger_index, right_finger_index)

    park_pos = pre_pos.astype(np.float32, copy=True)
    park_pos[2] = float(park_pos[2] + float(args.park_lift_z))
    q_park = _solve_q_for_pose(
        model,
        ee_body_index,
        ee_offset_local,
        q_pre,
        park_pos,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )

    q_approach = Q_APPROACH_SEED.copy()
    q_approach[7:9] = float(args.gripper_open)
    approach_seed_pos = _gripper_center_from_q(model, q_approach, left_finger_index, right_finger_index)
    approach_pos = approach_seed_pos.astype(np.float32, copy=True)
    approach_pos[0] = float(table_center[0] - 0.03)
    approach_pos[1] = float(table_center[1] - 0.02)
    approach_pos[2] = float(max(approach_pos[2], table_top_z + 0.09 + float(args.approach_lift_z)))
    q_approach = _solve_q_for_pose(
        model,
        ee_body_index,
        ee_offset_local,
        q_pre,
        approach_pos,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )

    contact_pos = approach_pos.astype(np.float32, copy=True)
    contact_pos[0] = float(table_center[0] - 0.01)
    contact_pos[1] = float(table_center[1] + 0.00)
    contact_pos[2] = float(table_top_z - float(args.contact_depth))
    q_contact = _solve_q_for_pose(
        model,
        ee_body_index,
        ee_offset_local,
        q_approach,
        contact_pos,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )

    push_pos = contact_pos.astype(np.float32, copy=True)
    push_pos[0] = float(push_pos[0] + float(args.push_x_shift))
    q_push = _solve_q_for_pose(
        model,
        ee_body_index,
        ee_offset_local,
        q_contact,
        push_pos,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )

    retract_pos = pre_pos.astype(np.float32, copy=True)
    retract_pos[2] = float(retract_pos[2] + float(args.retract_lift_z))
    q_retract = _solve_q_for_pose(
        model,
        ee_body_index,
        ee_offset_local,
        q_push,
        retract_pos,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )

    return [
        {"name": "settle", "duration": float(args.settle_seconds), "start_q": q_park.copy(), "end_q": q_park.copy()},
        {"name": "pre", "duration": float(args.pre_seconds), "start_q": q_park.copy(), "end_q": q_park.copy()},
        {"name": "approach", "duration": float(args.approach_seconds), "start_q": q_park.copy(), "end_q": q_approach.copy()},
        {"name": "contact", "duration": float(args.contact_seconds), "start_q": q_approach.copy(), "end_q": q_contact.copy()},
        {"name": "push", "duration": float(args.push_seconds), "start_q": q_contact.copy(), "end_q": q_push.copy()},
        {"name": "retract", "duration": float(args.retract_seconds), "start_q": q_push.copy(), "end_q": q_retract.copy()},
    ]


def _joint_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray]:
    phases = meta["joint_phase_targets"]
    elapsed = 0.0
    for phase in phases:
        duration = max(float(phase["duration"]), 1.0e-8)
        end_t = elapsed + duration
        if t <= end_t:
            if np.allclose(phase["start_q"], phase["end_q"]):
                alpha = 0.0
            else:
                alpha = np.clip((t - elapsed) / duration, 0.0, 1.0)
            q = (1.0 - alpha) * np.asarray(phase["start_q"], dtype=np.float32) + alpha * np.asarray(phase["end_q"], dtype=np.float32)
            return str(phase["name"]), q.astype(np.float32, copy=False)
        elapsed = end_t
    last = phases[-1]
    return str(last["name"]), np.asarray(last["end_q"], dtype=np.float32)


def _task_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    phase_name, q = _joint_phase_state(t, meta)
    target_pos = _gripper_center_from_q(
        meta["fk_model"],
        q,
        int(meta["left_finger_index"]),
        int(meta["right_finger_index"]),
    )
    return phase_name, target_pos.astype(np.float32, copy=False), np.asarray(meta["ee_target_quat"], dtype=np.float32)


def _camera_presets(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    target = np.asarray(
        [
            float(meta["table_center"][0] + 0.01),
            float(meta["table_center"][1] + 0.01),
            float(meta["table_top_z"] + 0.07),
        ],
        dtype=np.float32,
    )
    hero_pos = legacy.camera_position(target, yaw_deg=-30.0, pitch_deg=-15.0, distance=1.10)
    validation_pos = legacy.camera_position(target, yaw_deg=0.0, pitch_deg=-15.0, distance=1.10)
    return {
        "hero": {
            "pos": hero_pos.astype(np.float32, copy=False).tolist(),
            "pitch": -15.0,
            "yaw": -30.0,
            "fov": 40.0,
            "target": target.astype(np.float32, copy=False).tolist(),
        },
        "validation": {
            "pos": validation_pos.astype(np.float32, copy=False).tolist(),
            "pitch": -15.0,
            "yaw": 0.0,
            "fov": 48.0,
            "target": target.astype(np.float32, copy=False).tolist(),
        },
    }


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    builder.default_body_armature = float(args.default_body_armature)
    builder.default_joint_cfg.armature = float(args.default_joint_armature)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    franka_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    builder.add_urdf(
        franka_asset,
        xform=wp.transform(wp.vec3(*np.asarray(args.robot_base_pos, dtype=np.float32).tolist()), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
        ignore_inertial_definitions=bool(args.ignore_urdf_inertial_definitions),
    )

    for i in range(9):
        builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
    builder.joint_q[:9] = Q_PRE.tolist()
    builder.joint_target_pos[:9] = Q_PRE.tolist()
    builder.joint_target_ke[:7] = [float(args.joint_target_ke)] * 7
    builder.joint_target_kd[:7] = [float(args.joint_target_kd)] * 7
    builder.joint_target_ke[7:9] = [float(args.finger_target_ke)] * 2
    builder.joint_target_kd[7:9] = [float(args.finger_target_kd)] * 2
    builder.joint_target_pos[7:9] = [float(args.gripper_open)] * 2
    builder.joint_armature[:7] = [float(args.default_joint_armature)] * 7
    builder.joint_armature[7:9] = [0.15, 0.15]
    builder.joint_effort_limit[:7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    builder.joint_effort_limit[7:9] = [20.0, 20.0]

    ee_body_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_link7")
    left_finger_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_leftfinger")
    right_finger_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_rightfinger")

    table_center = np.asarray(args.table_center, dtype=np.float32)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*table_center.tolist()), wp.quat_identity()),
        hx=float(args.table_hx),
        hy=float(args.table_hy),
        hz=float(args.table_hz),
        cfg=builder.default_shape_cfg.copy(),
        label="tabletop_table_box",
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))

    ee_offset_local = _measure_link7_gripper_offset(model, ee_body_index, left_finger_index, right_finger_index)
    meta: dict[str, Any] = {
        "task": TABLETOP_PUSH_TASK,
        "blocking_stage": "rigid_only",
        "tabletop_control_mode": str(args.tabletop_control_mode),
        "particle_contacts": False,
        "particle_contact_kernel": False,
        "left_finger_index": int(left_finger_index),
        "right_finger_index": int(right_finger_index),
        "ee_body_index": int(ee_body_index),
        "ee_offset_local": ee_offset_local.astype(np.float32),
        "ee_target_quat": _target_quat(args),
        "table_center": table_center.astype(np.float32),
        "table_top_z": float(table_center[2] + float(args.table_hz)),
        "table_center_m": table_center.astype(np.float32),
        "stage_center": table_center.astype(np.float32),
        "stage_scale": np.asarray([float(args.table_hx), float(args.table_hy), float(args.table_hz)], dtype=np.float32),
        "anchor_bar_center": table_center.astype(np.float32),
        "anchor_bar_scale": np.asarray([float(args.table_hx), float(args.table_hy), float(args.table_hz)], dtype=np.float32),
        "support_point": table_center.astype(np.float32),
        "floor_z": 0.0,
        "floor_center": np.asarray([0.0, 0.0, -0.012], dtype=np.float32),
        "floor_scale": np.asarray([1.45, 1.10, 0.012], dtype=np.float32),
        "robot_base_center": np.asarray([float(args.robot_base_pos[0]), float(args.robot_base_pos[1]), max(0.18, float(args.robot_base_pos[2]))], dtype=np.float32),
        "robot_base_scale": np.asarray([0.17, 0.17, max(0.18, float(args.robot_base_pos[2]))], dtype=np.float32),
        "support_box_enabled": False,
        "support_box_physical": False,
        "support_box_shape_index": None,
        "support_box_center": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "support_box_scale": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "support_box_default_center": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "support_box_default_scale": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "support_box_geometry_source": "disabled_in_stage0",
        "support_box_normal_axis": "x",
        "support_box_normal_axis_index": 0,
        "support_box_normal_sign": 1.0,
        "visible_tool_enabled": False,
        "visible_tool_mode": "none",
        "visible_tool_body_index": None,
        "visible_tool_body_label": None,
        "visible_tool_shape_index": None,
        "visible_tool_shape_label": None,
        "visible_tool_axis": "z",
        "visible_tool_offset_local": np.zeros((3,), dtype=np.float32),
        "visible_tool_radius": None,
        "visible_tool_half_height": None,
        "visible_tool_total_length": None,
        "visible_tool_color": np.asarray([0.98, 0.18, 0.10], dtype=np.float32),
        "anchor_indices": np.zeros((0,), dtype=np.int32),
        "anchor_positions": np.zeros((0, 3), dtype=np.float32),
        "support_patch_indices": np.zeros((0,), dtype=np.int32),
        "support_patch_center_m": table_center.astype(np.float32),
        "visible_support_center_m": table_center.astype(np.float32),
        "render_edges": np.zeros((0, 2), dtype=np.int32),
        "rope_center": np.asarray([float(table_center[0]), float(table_center[1]), float(table_center[2])], dtype=np.float32),
        "total_object_mass": 0.0,
    }
    joint_phase_targets = _build_joint_phase_targets(model, meta, args, device)
    meta["joint_phase_targets"] = joint_phase_targets
    meta["joint_q_init"] = np.asarray(joint_phase_targets[0]["start_q"], dtype=np.float32)
    meta["fk_model"] = model
    task_phases: list[dict[str, Any]] = []
    for phase in joint_phase_targets:
        task_phases.append(
            {
                "name": str(phase["name"]),
                "duration": float(phase["duration"]),
                "start": _gripper_center_from_q(
                    model,
                    np.asarray(phase["start_q"], dtype=np.float32),
                    int(left_finger_index),
                    int(right_finger_index),
                ).astype(np.float32, copy=False),
                "end": _gripper_center_from_q(
                    model,
                    np.asarray(phase["end_q"], dtype=np.float32),
                    int(left_finger_index),
                    int(right_finger_index),
                ).astype(np.float32, copy=False),
                "quat": np.asarray(meta["ee_target_quat"], dtype=np.float32),
            }
        )
    meta["task_phases"] = task_phases
    meta["camera_presets"] = _camera_presets(meta)
    return model, {}, meta, 0


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        njmax=500,
        nconmax=500,
        iterations=int(args.solver_iterations),
        ls_iterations=int(args.solver_ls_iterations),
        impratio=1000.0,
    )
    collision_pipeline = newton.CollisionPipeline(model, reduce_contacts=True, broad_phase="explicit")
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_target_pos_buf = wp.zeros(control.joint_target_pos.shape, dtype=wp.float32, device=device)
    joint_target_vel_buf = (
        wp.zeros(control.joint_target_vel.shape, dtype=wp.float32, device=device)
        if getattr(control, "joint_target_vel", None) is not None
        else None
    )

    state_0 = model.state()
    state_1 = model.state()
    q_init = np.asarray(meta["joint_q_init"], dtype=np.float32)
    for state in (state_0, state_1):
        state.joint_q.assign(q_init)
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_target_pos_buf.assign(q_init)
    wp.copy(control.joint_target_pos, joint_target_pos_buf)
    if joint_target_vel_buf is not None:
        joint_target_vel_buf.zero_()
        wp.copy(control.joint_target_vel, joint_target_vel_buf)

    sim_dt = float(args.sim_dt)
    substeps = max(1, int(args.substeps))
    frame_dt = sim_dt * float(substeps)
    n_frames = max(2, int(args.frames))
    store = legacy.RolloutStorage(args.out_dir, args.prefix, mode=str(args.history_storage))
    particle_q_all = store.allocate("particle_q_all", (n_frames, 0, 3), np.float32)
    particle_q_object = store.allocate("particle_q_object", (n_frames, 0, 3), np.float32)
    body_q = store.allocate("body_q", (n_frames, model.body_count, 7), np.float32)
    body_vel = store.allocate("body_vel", (n_frames, model.body_count, 3), np.float32)
    ee_target_pos = store.allocate("ee_target_pos", (n_frames, 3), np.float32)

    t0 = time.perf_counter()
    for frame in range(n_frames):
        body_q[frame] = state_0.body_q.numpy().astype(np.float32)
        body_vel[frame] = state_0.body_qd.numpy().astype(np.float32)[:, :3]
        particle_q_all[frame] = np.zeros((0, 3), dtype=np.float32)
        particle_q_object[frame] = np.zeros((0, 3), dtype=np.float32)
        phase_name, target_q = _joint_phase_state(float(frame) * frame_dt, meta)
        ee_target_pos[frame] = _gripper_center_from_q(
            model,
            target_q,
            int(meta["left_finger_index"]),
            int(meta["right_finger_index"]),
        )
        for sub in range(substeps):
            sim_t = (float(frame) * float(substeps) + float(sub)) * sim_dt
            _, target_q_sub = _joint_phase_state(sim_t, meta)
            state_0.clear_forces()
            state_1.clear_forces()
            joint_target_pos_buf.assign(np.asarray(target_q_sub, dtype=np.float32))
            wp.copy(control.joint_target_pos, joint_target_pos_buf)
            if joint_target_vel_buf is not None:
                joint_target_vel_buf.zero_()
                wp.copy(control.joint_target_vel, joint_target_vel_buf)
            if getattr(control, "joint_f", None) is not None:
                control.joint_f.zero_()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    wall_time = float(time.perf_counter() - t0)
    return {
        "particle_q_all": particle_q_all,
        "particle_q_object": particle_q_object,
        "body_q": body_q,
        "body_vel": body_vel,
        "ee_target_pos": ee_target_pos,
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": wall_time,
        "robot_motion_mode": "split_v3_robot_first_joint_target_pos",
        "replay_source": None,
        "history_storage_mode": store.mode,
        "history_storage_files": store.files,
    }


def build_summary(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    out_mp4: Path,
) -> dict[str, Any]:
    frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    return {
        "task": TABLETOP_PUSH_TASK,
        "blocking_stage": "rigid_only",
        "tabletop_control_mode": str(args.tabletop_control_mode),
        "robot_motion_mode": str(sim_data["robot_motion_mode"]),
        "output_mp4": str(out_mp4),
        "frames": int(sim_data["body_q"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(frame_dt),
        "task_duration_s": float(_total_task_duration(args)),
        "wall_time_sec": float(sim_data["wall_time"]),
        "render_mode": str(args.render_mode),
        "camera_profile": str(args.camera_profile),
        "history_storage_mode": str(sim_data["history_storage_mode"]),
        "history_storage_files": sim_data["history_storage_files"],
        "recommended_hero_camera": meta["camera_presets"]["hero"],
        "recommended_validation_camera": meta["camera_presets"]["validation"],
        "tabletop_initial_pose": "split_v3_stage0",
        "tabletop_settle_seconds": float(args.settle_seconds),
        "tabletop_support_box_mode": "none",
        "support_box_enabled": False,
        "support_box_physical": False,
        "visible_tool_enabled": False,
        "visible_tool_mode": "none",
        "proof_surface": "actual_multi_box_finger_colliders",
        "robot_geometry": "native_franka",
        "visible_rope_present": False,
        "particle_contacts": False,
        "particle_contact_kernel": False,
        "gravity_mag_m_s2": float(args.gravity_mag),
        "joint_target_ke": float(args.joint_target_ke),
        "joint_target_kd": float(args.joint_target_kd),
        "finger_target_ke": float(args.finger_target_ke),
        "finger_target_kd": float(args.finger_target_kd),
        "default_body_armature": float(args.default_body_armature),
        "default_joint_armature": float(args.default_joint_armature),
        "table_center": np.asarray(meta["table_center"], dtype=np.float32).astype(float).tolist(),
        "table_top_z_m": float(meta["table_top_z"]),
    }


def build_physics_validation(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": TABLETOP_PUSH_TASK,
        "blocking_stage": "rigid_only",
        "rope_present": False,
        "robot_geometry": "native_franka",
        "table_top_z_m": float(meta["table_top_z"]),
        "summary_contact_proxy_mode": "actual_multi_box_finger_colliders",
        "no_hidden_helper": True,
    }


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    _resolve_runtime_defaults(args)

    model, ir_obj, meta, n_obj = build_model(args, device)
    legacy._resolve_camera_defaults(args, meta)
    if args.load_history_from_dir is not None:
        history_prefix = str(args.load_history_prefix or args.prefix)
        sim_data = legacy._load_saved_history(Path(args.load_history_from_dir), prefix=history_prefix, args=args, meta=meta)
        legacy._materialize_loaded_history(args, sim_data)
    else:
        sim_data = simulate(model, ir_obj, meta, args, n_obj, device)
    out_mp4 = legacy.render_video(model, sim_data, meta, args, device)
    out_gif = legacy.make_gif(args, out_mp4)
    summary = build_summary(model, sim_data, meta, args, out_mp4)
    physics_validation = build_physics_validation(model, sim_data, meta, args, summary)
    physics_validation_path = legacy.save_physics_validation_json(args, physics_validation)
    summary["physics_validation_path"] = str(physics_validation_path)
    if out_gif is not None:
        summary["output_gif"] = str(out_gif)
    summary_path = legacy.save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
