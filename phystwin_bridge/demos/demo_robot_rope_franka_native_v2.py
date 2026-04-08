#!/usr/bin/env python3
"""Native-style Franka tabletop rope demo v2.

This bridge-side demo keeps the rope on the existing PhysTwin -> Newton import
path, but rewrites the robot control surface to follow the native Newton robot
example structure more closely:

- build a minimal scene with native Franka + native table + bridged rope
- drive Cartesian end-effector waypoints through native Newton IK
- write only `control.joint_target_pos` / `control.joint_target_vel`
- keep solved `state_out.body_q` as the only post-step body truth

The first milestone intentionally omits support-box and visible-tool complexity.
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
from rollout_storage import RolloutStorage

newton = legacy.newton
ik = legacy.ik
newton_import_ir = legacy.newton_import_ir
path_defaults = legacy.path_defaults

TABLETOP_PUSH_TASK = legacy.TABLETOP_PUSH_TASK
FRANKA_INIT_Q = legacy.FRANKA_INIT_Q.copy()


def _default_rope_ir() -> Path:
    return legacy._default_rope_ir()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native-style Franka tabletop rope demo v2.")
    p.add_argument("--ir", type=Path, default=_default_rope_ir())
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_rope_tabletop_hero")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--task", choices=[TABLETOP_PUSH_TASK], default=TABLETOP_PUSH_TASK)
    p.add_argument("--blocking-stage", choices=["rope_integrated", "rigid_only"], default="rope_integrated")
    p.add_argument("--tabletop-control-mode", choices=["joint_target_drive"], default="joint_target_drive")

    p.add_argument("--render-mode", choices=["debug", "presentation"], default="presentation")
    p.add_argument("--camera-profile", choices=["hero", "validation"], default="hero")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=5.0e-5)
    p.add_argument("--substeps", type=int, default=667)
    p.add_argument("--history-storage", choices=["memory", "memmap"], default="memmap")
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--solver-type", choices=["semiimplicit", "mujoco"], default="mujoco")
    p.add_argument("--enable-gravcomp", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--object-mass", type=float, default=1.0)
    p.add_argument("--auto-set-weight", type=float, default=3.0)
    p.add_argument("--mass-spring-scale", type=float, default=None)
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument("--particle-radius-scale", type=float, default=0.1)
    p.add_argument("--particle-contacts", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--particle-contact-kernel", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument("--drag-ignore-gravity-axis", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pre-release-settle-damping-scale", type=float, default=1.0)

    p.add_argument("--ik-iters", type=int, default=24)
    p.add_argument("--ik-target-blend", type=float, default=1.0)
    p.add_argument("--joint-target-ke", type=float, default=100.0)
    p.add_argument("--joint-target-kd", type=float, default=10.0)
    p.add_argument("--finger-target-ke", type=float, default=20.0)
    p.add_argument("--finger-target-kd", type=float, default=2.0)
    p.add_argument("--solver-joint-attach-ke", type=float, default=50.0)
    p.add_argument("--solver-joint-attach-kd", type=float, default=5.0)
    p.add_argument("--default-body-armature", type=float, default=0.01)
    p.add_argument("--default-joint-armature", type=float, default=0.01)
    p.add_argument("--ignore-urdf-inertial-definitions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gripper-open", type=float, default=0.04)
    p.add_argument("--anchor-count-per-end", type=int, default=2)
    p.add_argument("--ee-contact-radius", type=float, default=0.04)

    p.add_argument("--tabletop-initial-pose", choices=["tabletop_curve", "tabletop_shallow_curve"], default="tabletop_curve")
    p.add_argument("--tabletop-rope-height", type=float, default=0.156)
    p.add_argument("--tabletop-table-top-z", type=float, default=0.200)
    p.add_argument("--tabletop-table-hx", type=float, default=0.42)
    p.add_argument("--tabletop-table-hy", type=float, default=0.24)
    p.add_argument("--tabletop-table-hz", type=float, default=0.020)
    p.add_argument("--tabletop-robot-base-offset", nargs=3, type=float, default=(-0.56, -0.22, 0.10))
    p.add_argument("--tabletop-ee-offset-z", type=float, default=None)
    p.add_argument("--tabletop-ee-yaw-deg", type=float, default=-18.0)

    p.add_argument("--tabletop-preroll-settle-seconds", type=float, default=3.0)
    p.add_argument("--tabletop-preroll-damping-scale", type=float, default=2.5)
    p.add_argument("--tabletop-reset-robot-after-preroll", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--tabletop-settle-seconds", type=float, default=0.20)
    p.add_argument("--tabletop-pre-seconds", type=float, default=0.50)
    p.add_argument("--tabletop-approach-seconds", type=float, default=1.10)
    p.add_argument("--tabletop-contact-seconds", type=float, default=0.35)
    p.add_argument("--tabletop-push-seconds", type=float, default=2.10)
    p.add_argument("--tabletop-retract-seconds", type=float, default=1.20)

    p.add_argument("--tabletop-park-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    p.add_argument("--tabletop-pre-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    p.add_argument("--tabletop-approach-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    p.add_argument("--tabletop-contact-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    p.add_argument("--tabletop-push-end-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    p.add_argument("--tabletop-retract-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))

    p.add_argument("--tabletop-park-clearance-z", type=float, default=0.06)
    p.add_argument("--tabletop-pre-clearance-z", type=float, default=0.00)
    p.add_argument("--tabletop-approach-clearance-z", type=float, default=0.02)
    p.add_argument("--tabletop-contact-clearance-z", type=float, default=0.00)
    p.add_argument("--tabletop-push-clearance-z", type=float, default=0.00)
    p.add_argument("--tabletop-retract-clearance-z", type=float, default=0.06)

    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gif-width", type=int, default=960)
    p.add_argument("--gif-fps", type=float, default=15.0)
    p.add_argument("--gif-max-colors", type=int, default=256)
    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
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
    return p.parse_args()


def _enable_franka_gravity_comp(builder: newton.ModelBuilder) -> None:
    gravcomp_attr = builder.custom_attributes.get("mujoco:jnt_actgravcomp")
    gravcomp_body = builder.custom_attributes.get("mujoco:gravcomp")
    if gravcomp_attr is None or gravcomp_body is None:
        return
    if gravcomp_attr.values is None:
        gravcomp_attr.values = {}
    for dof_idx in range(7):
        gravcomp_attr.values[dof_idx] = True
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body_idx, body_label in enumerate(builder.body_label):
        label = str(body_label)
        if any(
            label.endswith(suffix)
            for suffix in (
                "/fr3_link1",
                "/fr3_link2",
                "/fr3_link3",
                "/fr3_link4",
                "/fr3_link5",
                "/fr3_link6",
                "/fr3_link7",
                "/fr3_link8",
                "/fr3_hand",
                "/fr3_hand_tcp",
                "/fr3_leftfinger",
                "/fr3_rightfinger",
            )
        ):
            gravcomp_body.values[body_idx] = 1.0


def _total_task_duration(args: argparse.Namespace) -> float:
    return float(
        args.tabletop_settle_seconds
        + args.tabletop_pre_seconds
        + args.tabletop_approach_seconds
        + args.tabletop_contact_seconds
        + args.tabletop_push_seconds
        + args.tabletop_retract_seconds
    )


def _resolve_runtime_defaults(args: argparse.Namespace) -> None:
    if args.frames is None:
        frame_dt = float(args.sim_dt) * max(1, int(args.substeps))
        total_duration = _total_task_duration(args)
        args.frames = max(120, int(np.ceil(total_duration / max(frame_dt, 1.0e-8))) + 1)
    if args.overlay_label is None:
        args.overlay_label = str(args.render_mode) == "debug"


def _target_quat(args: argparse.Namespace) -> np.ndarray:
    down = legacy._quat_from_axis_angle_np(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.pi)
    yaw = legacy._quat_from_axis_angle_np(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32), np.deg2rad(float(args.tabletop_ee_yaw_deg))
    )
    return legacy._quat_multiply(down, yaw).astype(np.float32, copy=False)


def _phase_pose(
    rope_center: np.ndarray,
    rope_top_z: float,
    offset_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    clearance_z: float,
) -> np.ndarray:
    offset = np.asarray(offset_xyz, dtype=np.float32)
    pos = rope_center + offset
    pos = pos.astype(np.float32, copy=True)
    pos[2] = float(rope_top_z + clearance_z)
    return pos


def _build_rough_cartesian_phases(
    rope_center: np.ndarray,
    rope_top_z: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    quat = _target_quat(args)
    park = _phase_pose(rope_center, rope_top_z, args.tabletop_park_offset, float(args.tabletop_park_clearance_z))
    pre = _phase_pose(rope_center, rope_top_z, args.tabletop_pre_offset, float(args.tabletop_pre_clearance_z))
    approach = _phase_pose(rope_center, rope_top_z, args.tabletop_approach_offset, float(args.tabletop_approach_clearance_z))
    contact = _phase_pose(rope_center, rope_top_z, args.tabletop_contact_offset, float(args.tabletop_contact_clearance_z))
    push = _phase_pose(rope_center, rope_top_z, args.tabletop_push_end_offset, float(args.tabletop_push_clearance_z))
    retract = _phase_pose(rope_center, rope_top_z, args.tabletop_retract_offset, float(args.tabletop_retract_clearance_z))
    phases = [
        {
            "name": "settle",
            "duration": float(args.tabletop_settle_seconds),
            "start": pre.copy(),
            "end": pre.copy(),
            "quat": quat.copy(),
        },
        {
            "name": "pre",
            "duration": float(args.tabletop_pre_seconds),
            "start": pre.copy(),
            "end": pre.copy(),
            "quat": quat.copy(),
        },
        {
            "name": "approach",
            "duration": float(args.tabletop_approach_seconds),
            "start": pre.copy(),
            "end": approach.copy(),
            "quat": quat.copy(),
        },
        {
            "name": "contact",
            "duration": float(args.tabletop_contact_seconds),
            "start": approach.copy(),
            "end": contact.copy(),
            "quat": quat.copy(),
        },
        {
            "name": "push",
            "duration": float(args.tabletop_push_seconds),
            "start": contact.copy(),
            "end": push.copy(),
            "quat": quat.copy(),
        },
        {
            "name": "retract",
            "duration": float(args.tabletop_retract_seconds),
            "start": push.copy(),
            "end": retract.copy(),
            "quat": quat.copy(),
        },
    ]
    return park, phases


def _ee_world_from_joint_q(
    model: newton.Model,
    joint_q_row: np.ndarray,
    ee_body_index: int,
    ee_offset_local: np.ndarray,
) -> np.ndarray:
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q_row, dtype=np.float32))
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy().astype(np.float32)
    return legacy._ee_world_position(body_q[int(ee_body_index)], np.asarray(ee_offset_local, dtype=np.float32))


def _actualize_task_phases(
    model: newton.Model,
    ee_body_index: int,
    ee_offset_local: np.ndarray,
    args: argparse.Namespace,
    device: str,
    rough_park: np.ndarray,
    rough_phases: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    target_quat = np.asarray(rough_phases[0]["quat"], dtype=np.float32)
    rough_pre = np.asarray(rough_phases[0]["start"], dtype=np.float32)
    q_pre = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        FRANKA_INIT_Q,
        rough_pre,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    pre_world = _ee_world_from_joint_q(model, q_pre, ee_body_index, ee_offset_local)

    rough_approach = np.asarray(rough_phases[2]["end"], dtype=np.float32)
    q_approach = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_pre,
        rough_approach,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    approach_world = _ee_world_from_joint_q(model, q_approach, ee_body_index, ee_offset_local)

    rough_contact = np.asarray(rough_phases[3]["end"], dtype=np.float32)
    q_contact = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_approach,
        rough_contact,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    contact_world = _ee_world_from_joint_q(model, q_contact, ee_body_index, ee_offset_local)

    rough_push = np.asarray(rough_phases[4]["end"], dtype=np.float32)
    q_push = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_contact,
        rough_push,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    push_world = _ee_world_from_joint_q(model, q_push, ee_body_index, ee_offset_local)

    rough_retract = np.asarray(rough_phases[5]["end"], dtype=np.float32)
    q_retract = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_push,
        rough_retract,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    retract_world = _ee_world_from_joint_q(model, q_retract, ee_body_index, ee_offset_local)

    park_world_target = pre_world + (np.asarray(rough_park, dtype=np.float32) - rough_pre)
    q_park = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_pre,
        park_world_target,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    park_world = _ee_world_from_joint_q(model, q_park, ee_body_index, ee_offset_local)

    actual_phases = [
        {
            "name": "settle",
            "duration": float(args.tabletop_settle_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": pre_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "pre",
            "duration": float(args.tabletop_pre_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": pre_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "approach",
            "duration": float(args.tabletop_approach_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": approach_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "contact",
            "duration": float(args.tabletop_contact_seconds),
            "start": approach_world.astype(np.float32, copy=True),
            "end": contact_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "push",
            "duration": float(args.tabletop_push_seconds),
            "start": contact_world.astype(np.float32, copy=True),
            "end": push_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "retract",
            "duration": float(args.tabletop_retract_seconds),
            "start": push_world.astype(np.float32, copy=True),
            "end": retract_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
    ]
    return park_world.astype(np.float32), actual_phases, q_park.astype(np.float32), q_pre.astype(np.float32)


def _seeded_task_phases(
    model: newton.Model,
    args: argparse.Namespace,
    left_finger_index: int,
    right_finger_index: int,
    ee_body_index: int,
    ee_offset_local: np.ndarray,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    def _seed_pos(joint_q: np.ndarray) -> np.ndarray:
        return legacy._fk_gripper_center_from_joint_q(
            model,
            np.asarray(joint_q, dtype=np.float32),
            left_finger_idx=int(left_finger_index),
            right_finger_idx=int(right_finger_index),
        ).astype(np.float32, copy=False)

    target_quat = _target_quat(args)
    q_pre = np.asarray(legacy.TABLETOP_BLOCKING_Q_HIGH_PRE, dtype=np.float32).copy()
    q_pre[7:9] = float(args.gripper_open)

    pre_world = _seed_pos(q_pre)
    approach_world = _seed_pos(legacy.TABLETOP_FRANKA_Q_PRE)
    contact_world = _seed_pos(legacy.TABLETOP_BLOCKING_Q_PUSH_START)
    push_world = _seed_pos(legacy.TABLETOP_FRANKA_Q_PUSH_END)

    pre_world = pre_world + np.asarray(args.tabletop_pre_offset, dtype=np.float32)
    pre_world[2] += float(args.tabletop_pre_clearance_z)
    approach_world = approach_world + np.asarray(args.tabletop_approach_offset, dtype=np.float32)
    approach_world[2] += float(args.tabletop_approach_clearance_z)
    contact_world = contact_world + np.asarray(args.tabletop_contact_offset, dtype=np.float32)
    contact_world[2] += float(args.tabletop_contact_clearance_z)
    push_world = push_world + np.asarray(args.tabletop_push_end_offset, dtype=np.float32)
    push_world[2] += float(args.tabletop_push_clearance_z)

    retract_world = pre_world + np.asarray(args.tabletop_retract_offset, dtype=np.float32)
    retract_world[2] += float(args.tabletop_retract_clearance_z)
    park_world_target = pre_world + np.asarray(args.tabletop_park_offset, dtype=np.float32)
    park_world_target[2] += float(args.tabletop_park_clearance_z)

    q_park = _solve_cartesian_ik_target(
        model,
        ee_body_index,
        ee_offset_local,
        q_pre,
        park_world_target,
        target_quat,
        int(args.ik_iters),
        float(args.gripper_open),
        device,
    )
    park_world = _ee_world_from_joint_q(model, q_park, ee_body_index, ee_offset_local)

    phases = [
        {
            "name": "settle",
            "duration": float(args.tabletop_settle_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": pre_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "pre",
            "duration": float(args.tabletop_pre_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": pre_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "approach",
            "duration": float(args.tabletop_approach_seconds),
            "start": pre_world.astype(np.float32, copy=True),
            "end": approach_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "contact",
            "duration": float(args.tabletop_contact_seconds),
            "start": approach_world.astype(np.float32, copy=True),
            "end": contact_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "push",
            "duration": float(args.tabletop_push_seconds),
            "start": contact_world.astype(np.float32, copy=True),
            "end": push_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
        {
            "name": "retract",
            "duration": float(args.tabletop_retract_seconds),
            "start": push_world.astype(np.float32, copy=True),
            "end": retract_world.astype(np.float32, copy=True),
            "quat": target_quat.copy(),
        },
    ]
    return park_world.astype(np.float32), phases, q_park.astype(np.float32), q_pre.astype(np.float32)


def _task_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    phases = meta["task_phases"]
    elapsed = 0.0
    for phase in phases:
        duration = max(float(phase["duration"]), 1.0e-8)
        end_t = elapsed + duration
        if t <= end_t:
            if np.allclose(phase["start"], phase["end"]):
                alpha = 0.0
            else:
                alpha = np.clip((t - elapsed) / duration, 0.0, 1.0)
            pos = (1.0 - alpha) * np.asarray(phase["start"], dtype=np.float32) + alpha * np.asarray(phase["end"], dtype=np.float32)
            quat = np.asarray(phase["quat"], dtype=np.float32)
            return str(phase["name"]), pos.astype(np.float32, copy=False), quat
        elapsed = end_t
    last = phases[-1]
    return str(last["name"]), np.asarray(last["end"], dtype=np.float32), np.asarray(last["quat"], dtype=np.float32)


def _joint_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray]:
    phase_name, _, _ = _task_phase_state(t, meta)
    return phase_name, np.asarray(meta["joint_q_init"], dtype=np.float32)


def _solve_cartesian_ik_target(
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
    q_seed = np.asarray(q_seed, dtype=np.float32).copy()
    joint_q_ik = wp.array(q_seed.reshape(1, -1), dtype=wp.float32, device=device)
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
    solver.step(joint_q_ik, joint_q_ik, iterations=int(ik_iters))
    q_target = joint_q_ik.numpy().reshape(-1).astype(np.float32)
    q_target[7:9] = float(gripper_open)
    return q_target


def _measured_link7_gripper_center_local_offset(
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


class NativeV2SceneBuilder:
    def __init__(self, args: argparse.Namespace, device: str):
        self.args = args
        self.device = device

    def build(self) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
        args = self.args
        device = self.device
        rigid_only_stage = str(args.blocking_stage) == "rigid_only"

        raw_ir = legacy.load_ir(args.ir)
        legacy._validate_scaling_args(args)
        legacy._maybe_autoset_mass_spring_scale(args, raw_ir)
        ir_obj = legacy._copy_object_only_ir(raw_ir, args)

        ir_n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
        n_obj = 0 if rigid_only_stage else int(ir_n_obj)
        x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
        edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
        collision_radius_arr = np.asarray(
            ir_obj.get(
                "collision_radius",
                ir_obj.get("contact_collision_dist", np.full((ir_n_obj,), 0.026, dtype=np.float32)),
            ),
            dtype=np.float32,
        ).reshape(-1)
        particle_radius_ref = float(collision_radius_arr[0]) if collision_radius_arr.size else 0.026

        endpoint_indices = legacy._rope_endpoints(edges, ir_n_obj, x0)
        endpoint_mid = 0.5 * (x0[int(endpoint_indices[0])] + x0[int(endpoint_indices[1])])
        shift = np.asarray(
            [
                -float(endpoint_mid[0]),
                -float(endpoint_mid[1]),
                float(args.tabletop_rope_height) - float(endpoint_mid[2]),
            ],
            dtype=np.float32,
        )
        shifted_q = x0 + shift
        shifted_q[:ir_n_obj] = legacy._reshape_rope_for_tabletop(
            shifted_q[:ir_n_obj],
            table_top_z=float(args.tabletop_table_top_z),
            particle_radius=particle_radius_ref,
            pose_mode=str(args.tabletop_initial_pose),
        )
        rope_center = shifted_q[:ir_n_obj].mean(axis=0).astype(np.float32, copy=False)
        anchor_indices = legacy._anchor_particle_indices(
            shifted_q,
            endpoint_indices=endpoint_indices,
            count_per_end=int(args.anchor_count_per_end),
        )
        anchor_positions = shifted_q[anchor_indices].astype(np.float32, copy=False)

        spring_ke_scale, spring_kd_scale = legacy._effective_spring_scales(ir_obj, args)
        particle_contacts, particle_contact_kernel = legacy._resolve_particle_contact_settings(ir_obj, args)
        cfg = newton_import_ir.SimConfig(
            ir_path=args.ir.resolve(),
            out_dir=args.out_dir.resolve(),
            output_prefix=args.prefix,
            spring_ke_scale=float(spring_ke_scale),
            spring_kd_scale=float(spring_kd_scale),
            angular_damping=0.05,
            friction_smoothing=1.0,
            enable_tri_contact=False,
            disable_particle_contact_kernel=not bool(particle_contact_kernel),
            shape_contacts=True,
            strict_physics_checks=False,
            apply_drag=bool(args.apply_drag),
            drag_damping_scale=float(args.drag_damping_scale),
            gravity=-float(args.gravity_mag),
            gravity_from_reverse_z=False,
            up_axis="Z",
            particle_contacts=bool(particle_contacts),
            device=device,
            add_ground_plane=False,
        )

        checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)
        builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
        if str(self.args.solver_type) == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_body_armature = float(args.default_body_armature)
        builder.default_joint_cfg.armature = float(args.default_joint_armature)

        if not rigid_only_stage:
            _, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=bool(particle_contacts))
            newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
            builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]
            for idx in anchor_indices.tolist():
                builder.particle_flags[idx] = int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)
                builder.particle_mass[idx] = 0.0

        robot_base_pos = rope_center + np.asarray(args.tabletop_robot_base_offset, dtype=np.float32)
        franka_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        builder.add_urdf(
            franka_asset,
            xform=wp.transform(wp.vec3(*robot_base_pos.tolist()), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
            ignore_inertial_definitions=bool(args.ignore_urdf_inertial_definitions),
        )

        builder.joint_q[:9] = FRANKA_INIT_Q.tolist()
        builder.joint_target_pos[:9] = FRANKA_INIT_Q.tolist()
        builder.joint_target_ke[:7] = [float(args.joint_target_ke)] * 7
        builder.joint_target_kd[:7] = [float(args.joint_target_kd)] * 7
        builder.joint_target_ke[7:9] = [float(args.finger_target_ke)] * 2
        builder.joint_target_kd[7:9] = [float(args.finger_target_kd)] * 2
        builder.joint_target_pos[7:9] = [float(args.gripper_open)] * 2
        builder.joint_armature[:7] = [float(args.default_joint_armature)] * 7
        builder.joint_armature[7:9] = [0.15, 0.15]
        builder.joint_effort_limit[:7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
        builder.joint_effort_limit[7:9] = [20.0, 20.0]
        if str(args.solver_type) == "mujoco" and bool(args.enable_gravcomp):
            _enable_franka_gravity_comp(builder)

        ee_body_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_link7")
        left_finger_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_leftfinger")
        right_finger_index = legacy._find_index_by_suffix(builder.body_label, "/fr3_rightfinger")

        table_center = np.asarray(
            [
                float(rope_center[0]),
                float(rope_center[1]),
                float(args.tabletop_table_top_z - args.tabletop_table_hz),
            ],
            dtype=np.float32,
        )
        table_scale = np.asarray(
            [float(args.tabletop_table_hx), float(args.tabletop_table_hy), float(args.tabletop_table_hz)],
            dtype=np.float32,
        )
        table_cfg = builder.default_shape_cfg.copy()
        newton_import_ir._configure_ground_contact_material(table_cfg, ir_obj, cfg, checks, context="tabletop_table_box")
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*table_center.tolist()), wp.quat_identity()),
            hx=float(table_scale[0]),
            hy=float(table_scale[1]),
            hz=float(table_scale[2]),
            cfg=table_cfg,
            label="tabletop_table_box",
        )

        model = builder.finalize(device=device)
        _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
        if not bool(particle_contact_kernel):
            model.particle_grid = None
        _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
        model.set_gravity(gravity_vec)

        measured_offset = _measured_link7_gripper_center_local_offset(
            model,
            int(ee_body_index),
            int(left_finger_index),
            int(right_finger_index),
        )
        if args.tabletop_ee_offset_z is None:
            ee_offset_local = measured_offset.astype(np.float32, copy=True)
        else:
            ee_offset_local = np.asarray([0.0, 0.0, float(args.tabletop_ee_offset_z)], dtype=np.float32)
        rope_top_z = float(args.tabletop_table_top_z) + 2.0 * float(particle_radius_ref) + 0.004
        park_pose, task_phases, q_park, q_pre = _seeded_task_phases(
            model,
            args,
            int(left_finger_index),
            int(right_finger_index),
            int(ee_body_index),
            ee_offset_local,
            device,
        )
        target_quat = np.asarray(task_phases[0]["quat"], dtype=np.float32)

        init_state = model.state()
        init_state.joint_q.assign(q_pre.astype(np.float32))
        init_state.joint_qd.zero_()
        newton.eval_fk(model, init_state.joint_q, init_state.joint_qd, init_state)
        init_body_q = init_state.body_q.numpy().astype(np.float32)

        floor_scale = np.asarray([1.45, 1.10, 0.012], dtype=np.float32)
        floor_center = np.asarray([0.0, 0.0, -float(floor_scale[2])], dtype=np.float32)
        robot_base_scale = np.asarray([0.17, 0.17, max(0.18, float(robot_base_pos[2]))], dtype=np.float32)
        robot_base_center = np.asarray(
            [float(robot_base_pos[0]), float(robot_base_pos[1]), float(robot_base_scale[2])],
            dtype=np.float32,
        )
        meta: dict[str, Any] = {
            "task": TABLETOP_PUSH_TASK,
            "blocking_stage": str(args.blocking_stage),
            "tabletop_control_mode": "joint_target_drive",
            "solver_type": str(args.solver_type),
            "enable_gravcomp": bool(args.enable_gravcomp),
            "particle_contacts": bool(particle_contacts),
            "particle_contact_kernel": bool(particle_contact_kernel),
            "joint_q_init": q_pre.astype(np.float32),
            "joint_q_park": q_park.astype(np.float32),
            "ee_body_index": int(ee_body_index),
            "ee_offset_local": ee_offset_local.astype(np.float32),
            "measured_ee_offset_local": measured_offset.astype(np.float32),
            "left_finger_index": int(left_finger_index),
            "right_finger_index": int(right_finger_index),
            "rope_center": rope_center.astype(np.float32),
            "table_center": table_center.astype(np.float32),
            "table_top_z": float(args.tabletop_table_top_z),
            "table_top_z_m": float(args.tabletop_table_top_z),
            "tabletop_target_z": {
                "park": float(park_pose[2]),
                "pre": float(task_phases[0]["start"][2]),
                "approach": float(task_phases[2]["end"][2]),
                "contact": float(task_phases[3]["end"][2]),
                "push": float(task_phases[4]["end"][2]),
                "retract": float(task_phases[5]["end"][2]),
            },
            "tabletop_rope_top_z": float(rope_top_z),
            "anchor_indices": np.asarray(anchor_indices, dtype=np.int32),
            "anchor_positions": np.asarray(anchor_positions, dtype=np.float32),
            "support_patch_indices": np.asarray(anchor_indices, dtype=np.int32),
            "support_patch_center_m": np.mean(anchor_positions, axis=0).astype(np.float32, copy=False),
            "visible_support_center_m": np.mean(anchor_positions, axis=0).astype(np.float32, copy=False),
            "render_edges": np.asarray(edges[:n_obj if rigid_only_stage else edges.shape[0]], dtype=np.int32),
            "stage_center": table_center.astype(np.float32),
            "stage_scale": table_scale.astype(np.float32),
            "anchor_bar_center": table_center.astype(np.float32),
            "anchor_bar_scale": table_scale.astype(np.float32),
            "support_point": table_center.astype(np.float32),
            "floor_z": 0.0,
            "floor_center": floor_center.astype(np.float32),
            "floor_scale": floor_scale.astype(np.float32),
            "robot_base_center": robot_base_center.astype(np.float32),
            "robot_base_scale": robot_base_scale.astype(np.float32),
            "support_box_enabled": False,
            "support_box_physical": False,
            "support_box_shape_index": None,
            "support_box_center": robot_base_center.astype(np.float32),
            "support_box_scale": robot_base_scale.astype(np.float32),
            "support_box_default_center": robot_base_center.astype(np.float32),
            "support_box_default_scale": robot_base_scale.astype(np.float32),
            "support_box_geometry_source": "disabled_in_v2",
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
            "ee_target_start": np.asarray(task_phases[0]["start"], dtype=np.float32),
            "ee_target_quat": target_quat.astype(np.float32),
            "task_phases": task_phases,
            "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:ir_n_obj].sum()) if not rigid_only_stage else 0.0,
            "camera_presets": {},
        }
        meta["camera_presets"] = legacy._camera_presets(meta)

        return model, ir_obj, meta, n_obj


class NativeStyleController:
    def __init__(self, model: newton.Model, meta: dict[str, Any], args: argparse.Namespace, device: str):
        self.model = model
        self.meta = meta
        self.args = args
        self.device = device
        self.ee_body_index = int(meta["ee_body_index"])
        self.ee_offset_local = np.asarray(meta["ee_offset_local"], dtype=np.float32)
        self.joint_q_ik = wp.array(np.asarray(meta["joint_q_init"], dtype=np.float32).reshape(1, -1), dtype=wp.float32, device=device)
        self.prev_target_q = np.asarray(meta["joint_q_init"], dtype=np.float32).copy()
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_body_index,
            link_offset=wp.vec3(*self.ee_offset_local.tolist()),
            target_positions=wp.array([wp.vec3(*np.asarray(meta["ee_target_start"], dtype=np.float32).tolist())], dtype=wp.vec3, device=device),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*np.asarray(meta["ee_target_quat"], dtype=np.float32).tolist())], dtype=wp.vec4, device=device),
        )
        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def reset_to_joint_target(self, joint_q: np.ndarray) -> None:
        joint_q = np.asarray(joint_q, dtype=np.float32)
        self.prev_target_q = joint_q.copy()
        self.joint_q_ik.assign(joint_q.reshape(1, -1))

    def command_for_time(self, sim_t: float) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phase_name, target_pos, target_quat = _task_phase_state(sim_t, self.meta)
        self.pos_obj.set_target_position(0, wp.vec3(*target_pos.tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_quat.tolist()))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=int(self.args.ik_iters))
        joint_target_q = self.joint_q_ik.numpy().reshape(-1).astype(np.float32)
        blend = float(np.clip(float(self.args.ik_target_blend), 0.0, 1.0))
        if blend < 1.0:
            joint_target_q = self.prev_target_q + blend * (joint_target_q - self.prev_target_q)
        joint_target_q[7:9] = float(self.args.gripper_open)
        joint_target_qd = (joint_target_q - self.prev_target_q) / max(float(self.args.sim_dt), 1.0e-12)
        self.prev_target_q = joint_target_q.copy()
        return phase_name, target_pos, target_quat, joint_target_q.astype(np.float32), joint_target_qd.astype(np.float32)


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
    return NativeV2SceneBuilder(args, device).build()


def _set_control_targets(
    control: Any,
    joint_target_pos_buf: wp.array | None,
    joint_target_vel_buf: wp.array | None,
    joint_target_q: np.ndarray,
    joint_target_qd: np.ndarray,
) -> None:
    if control is not None and getattr(control, "joint_target_pos", None) is not None:
        joint_target_pos_buf.assign(np.asarray(joint_target_q, dtype=np.float32))
        wp.copy(dest=control.joint_target_pos, src=joint_target_pos_buf)
    if control is not None and getattr(control, "joint_target_vel", None) is not None:
        joint_target_vel_buf.assign(np.asarray(joint_target_qd, dtype=np.float32))
        wp.copy(dest=control.joint_target_vel, src=joint_target_vel_buf)
    if control is not None and getattr(control, "joint_f", None) is not None:
        control.joint_f.zero_()


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    rigid_only_stage = str(args.blocking_stage) == "rigid_only"
    spring_ke_scale, spring_kd_scale = legacy._effective_spring_scales(ir_obj, args)
    particle_contacts, particle_contact_kernel = legacy._resolve_particle_contact_settings(ir_obj, args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(spring_ke_scale),
        spring_kd_scale=float(spring_kd_scale),
        angular_damping=0.05,
        friction_smoothing=1.0,
        enable_tri_contact=False,
        disable_particle_contact_kernel=not bool(particle_contact_kernel),
        shape_contacts=True,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=bool(particle_contacts),
        device=device,
        add_ground_plane=False,
    )

    if str(args.solver_type) == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model,
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=8000,
            njmax=16000,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        )
    else:
        solver = newton.solvers.SolverSemiImplicit(
            model,
            angular_damping=cfg.angular_damping,
            friction_smoothing=cfg.friction_smoothing,
            joint_attach_ke=float(args.solver_joint_attach_ke),
            joint_attach_kd=float(args.solver_joint_attach_kd),
            enable_tri_contact=cfg.enable_tri_contact,
        )
    collision_pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_target_pos_buf = wp.zeros(control.joint_target_pos.shape, dtype=wp.float32, device=device)
    joint_target_vel_buf = (
        wp.zeros(control.joint_target_vel.shape, dtype=wp.float32, device=device)
        if getattr(control, "joint_target_vel", None) is not None
        else None
    )

    controller = NativeStyleController(model, meta, args, device)
    state_0 = model.state()
    state_1 = model.state()

    joint_q_pre = np.asarray(meta["joint_q_init"], dtype=np.float32)
    joint_q_park = np.asarray(meta["joint_q_park"], dtype=np.float32)
    for state in (state_0, state_1):
        state.joint_q.assign(joint_q_park)
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    controller.reset_to_joint_target(joint_q_park)
    _set_control_targets(control, joint_target_pos_buf, joint_target_vel_buf, joint_q_park, np.zeros_like(joint_q_park))

    sim_dt = float(args.sim_dt)
    substeps = max(1, int(args.substeps))
    frame_dt = sim_dt * float(substeps)
    n_frames = max(2, int(args.frames))
    history = RolloutStorage(args.out_dir, args.prefix, mode=str(args.history_storage))

    particle_q0 = (
        state_0.particle_q.numpy().astype(np.float32)
        if state_0.particle_q is not None
        else np.zeros((0, 3), dtype=np.float32)
    )
    body_q0 = state_0.body_q.numpy().astype(np.float32)
    body_qd0 = state_0.body_qd.numpy().astype(np.float32)
    particle_q_all = history.allocate("particle_q_all", (n_frames, particle_q0.shape[0], 3), np.float32)
    particle_q_object = history.allocate("particle_q_object", (n_frames, n_obj, 3), np.float32)
    body_q = history.allocate("body_q", (n_frames, body_q0.shape[0], body_q0.shape[1]), np.float32)
    body_vel = history.allocate("body_vel", (n_frames, body_qd0.shape[0], 3), np.float32)
    ee_target_pos = history.allocate("ee_target_pos", (n_frames, 3), np.float32)

    particle_radius = (
        model.particle_radius.numpy().astype(np.float32)[:n_obj]
        if getattr(model, "particle_radius", None) is not None and n_obj > 0
        else np.zeros((0,), dtype=np.float32)
    )

    preroll_frames = 0
    if (not rigid_only_stage) and float(args.tabletop_preroll_settle_seconds) > 0.0:
        preroll_frames = max(1, int(np.ceil(float(args.tabletop_preroll_settle_seconds) / max(frame_dt, 1.0e-12))))
        _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
        gravity_axis = None
        gravity_norm = float(np.linalg.norm(gravity_vec))
        if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
            gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)
        preroll_drag = 0.0
        if "drag_damping" in ir_obj:
            preroll_drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.tabletop_preroll_damping_scale)
        for _ in range(preroll_frames):
            for _sub in range(substeps):
                state_0.clear_forces()
                state_1.clear_forces()
                _set_control_targets(
                    control,
                    joint_target_pos_buf,
                    joint_target_vel_buf,
                    joint_q_park,
                    np.zeros_like(joint_q_park, dtype=np.float32),
                )
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                if str(args.solver_type) == "semiimplicit":
                    newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)
                state_0, state_1 = state_1, state_0
                if preroll_drag > 0.0 and n_obj > 0 and state_0.particle_q is not None and state_0.particle_qd is not None:
                    if gravity_axis is not None:
                        wp.launch(
                            legacy._apply_drag_correction_ignore_axis,
                            dim=n_obj,
                            inputs=[
                                state_0.particle_q,
                                state_0.particle_qd,
                                n_obj,
                                sim_dt,
                                preroll_drag,
                                wp.vec3(*gravity_axis.tolist()),
                            ],
                            device=device,
                        )
                    else:
                        wp.launch(
                            newton_import_ir._apply_drag_correction,
                            dim=n_obj,
                            inputs=[state_0.particle_q, state_0.particle_qd, n_obj, sim_dt, preroll_drag],
                            device=device,
                        )

    if bool(args.tabletop_reset_robot_after_preroll):
        for state in (state_0, state_1):
            state.joint_q.assign(joint_q_pre)
            state.joint_qd.zero_()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        controller.reset_to_joint_target(joint_q_pre)
        _set_control_targets(control, joint_target_pos_buf, joint_target_vel_buf, joint_q_pre, np.zeros_like(joint_q_pre))

    phase_name_series: list[str] = []
    t0 = time.perf_counter()
    for frame in range(n_frames):
        body_q_frame = state_0.body_q.numpy().astype(np.float32)
        body_q[frame] = body_q_frame
        body_vel[frame] = state_0.body_qd.numpy().astype(np.float32)[:, :3]
        if state_0.particle_q is not None:
            particle_q_all[frame] = state_0.particle_q.numpy().astype(np.float32)
            particle_q_object[frame] = particle_q_all[frame][:n_obj]
        else:
            particle_q_all[frame] = np.zeros((0, 3), dtype=np.float32)
            particle_q_object[frame] = np.zeros((n_obj, 3), dtype=np.float32)

        phase_name, target_pos, _ = _task_phase_state(float(frame) * frame_dt, meta)
        phase_name_series.append(phase_name)
        ee_target_pos[frame] = target_pos.astype(np.float32, copy=False)

        _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
        gravity_axis = None
        gravity_norm = float(np.linalg.norm(gravity_vec))
        if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
            gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)
        active_drag = 0.0
        if args.apply_drag and "drag_damping" in ir_obj:
            active_drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)

        for sub in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()
            sim_t = (float(frame) * float(substeps) + float(sub)) * sim_dt
            _, target_pos_sub, _, joint_target_q, joint_target_qd = controller.command_for_time(sim_t)
            _set_control_targets(control, joint_target_pos_buf, joint_target_vel_buf, joint_target_q, joint_target_qd)
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            if str(args.solver_type) == "semiimplicit":
                newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)
            state_0, state_1 = state_1, state_0
            if active_drag > 0.0 and n_obj > 0 and state_0.particle_q is not None and state_0.particle_qd is not None:
                if gravity_axis is not None:
                    wp.launch(
                        legacy._apply_drag_correction_ignore_axis,
                        dim=n_obj,
                        inputs=[
                            state_0.particle_q,
                            state_0.particle_qd,
                            n_obj,
                            sim_dt,
                            active_drag,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        newton_import_ir._apply_drag_correction,
                        dim=n_obj,
                        inputs=[state_0.particle_q, state_0.particle_qd, n_obj, sim_dt, active_drag],
                        device=device,
                    )
            ee_target_pos[frame] = target_pos_sub.astype(np.float32, copy=False)

    wall_time = float(time.perf_counter() - t0)
    sim_data = {
        "particle_q_all": particle_q_all,
        "particle_q_object": particle_q_object,
        "body_q": body_q,
        "body_vel": body_vel,
        "ee_target_pos": ee_target_pos,
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": wall_time,
        "robot_motion_mode": "native_cartesian_ik_joint_target_pos",
        "replay_source": None,
        "preroll_frame_count": int(preroll_frames),
        "history_storage_mode": history.mode,
        "history_storage_files": history.files,
        "phase_name_series": phase_name_series,
    }
    return sim_data


def _camera_presets(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return legacy._camera_presets(meta)


def _resolve_camera_defaults(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    presets = meta.get("camera_presets") or _camera_presets(meta)
    profile = str(getattr(args, "camera_profile", "hero"))
    if profile not in presets:
        profile = "hero"
    chosen = presets[profile]
    if args.camera_pos is None:
        args.camera_pos = tuple(float(v) for v in chosen["pos"])
    else:
        args.camera_pos = tuple(float(v) for v in args.camera_pos)
    if args.camera_pitch is None:
        args.camera_pitch = float(chosen["pitch"])
    if args.camera_yaw is None:
        args.camera_yaw = float(chosen["yaw"])
    if args.camera_fov is None:
        args.camera_fov = float(chosen["fov"])
    args.camera_profile = profile


def _compute_rope_contact_metrics(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    particle_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    ee_target_pos = np.asarray(sim_data["ee_target_pos"], dtype=np.float32)
    frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    frames = int(body_q.shape[0])
    finger_entries = legacy._finger_box_entries(model, meta)
    particle_radius = (
        model.particle_radius.numpy().astype(np.float32)[: particle_q.shape[1]]
        if getattr(model, "particle_radius", None) is not None and particle_q.shape[1] > 0
        else np.zeros((0,), dtype=np.float32)
    )

    actual_gripper = 0.5 * (body_q[:, int(meta["left_finger_index"]), :3] + body_q[:, int(meta["right_finger_index"]), :3])
    ee_error = np.linalg.norm(actual_gripper - ee_target_pos, axis=1)

    finger_clearance = np.full((frames,), np.inf, dtype=np.float32)
    finger_peak_name = np.full((frames,), "", dtype=object)
    for frame_idx in range(frames):
        if particle_q.shape[1] == 0:
            continue
        clearance, peak_name, _ = legacy._min_finger_box_clearance(
            particle_q[frame_idx],
            particle_radius,
            body_q[frame_idx],
            finger_entries,
        )
        finger_clearance[frame_idx] = float(clearance)
        finger_peak_name[frame_idx] = str(peak_name)
    finger_contact_mask = np.asarray(np.isfinite(finger_clearance) & (finger_clearance <= 0.0), dtype=bool)
    finger_contact_frames = np.flatnonzero(finger_contact_mask)
    first_rope_contact_frame = None if finger_contact_frames.size == 0 else int(finger_contact_frames[0])
    last_rope_contact_frame = None if finger_contact_frames.size == 0 else int(finger_contact_frames[-1])
    rope_com = particle_q.mean(axis=1) if particle_q.shape[1] > 0 else np.zeros((frames, 3), dtype=np.float32)
    rope_com_disp = float(np.linalg.norm(rope_com[-1] - rope_com[0])) if rope_com.shape[0] > 0 else 0.0
    first_contact_phase = None if first_rope_contact_frame is None else str(sim_data["phase_name_series"][first_rope_contact_frame])
    return {
        "gripper_center_tracking_error_mean_m": float(np.mean(ee_error)),
        "gripper_center_tracking_error_max_m": float(np.max(ee_error)),
        "rope_com_displacement_m": rope_com_disp,
        "actual_finger_box_contact_started": bool(first_rope_contact_frame is not None),
        "actual_finger_box_first_contact_frame": first_rope_contact_frame,
        "actual_finger_box_first_contact_time_s": (
            None if first_rope_contact_frame is None else float(first_rope_contact_frame * frame_dt)
        ),
        "actual_finger_box_last_contact_frame": last_rope_contact_frame,
        "actual_finger_box_contact_duration_s": float(np.count_nonzero(finger_contact_mask) * frame_dt),
        "actual_finger_box_clearance_min_m": (
            None if not np.any(np.isfinite(finger_clearance)) else float(np.min(finger_clearance[np.isfinite(finger_clearance)]))
        ),
        "actual_finger_box_peak_source": (
            None if first_rope_contact_frame is None else str(finger_peak_name[int(np.argmin(finger_clearance))])
        ),
        "first_contact_phase": first_contact_phase,
        "first_contact_time_s": (
            None if first_rope_contact_frame is None else float(first_rope_contact_frame * frame_dt)
        ),
    }


def build_summary(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    out_mp4: Path,
) -> dict[str, Any]:
    frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    rope_metrics = _compute_rope_contact_metrics(model, sim_data, meta, args)
    summary = {
        "ir_path": str(args.ir.resolve()),
        "output_mp4": str(out_mp4),
        "frames": int(sim_data["body_q"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(frame_dt),
        "task": TABLETOP_PUSH_TASK,
        "task_duration_s": float(_total_task_duration(args)),
        "wall_time_sec": float(sim_data["wall_time"]),
        "blocking_stage": str(args.blocking_stage),
        "render_mode": str(args.render_mode),
        "camera_profile": str(args.camera_profile),
        "recommended_hero_camera": meta["camera_presets"]["hero"],
        "recommended_validation_camera": meta["camera_presets"]["validation"],
        "history_storage_mode": str(sim_data["history_storage_mode"]),
        "history_storage_files": sim_data["history_storage_files"],
        "robot_motion_mode": str(sim_data["robot_motion_mode"]),
        "robot_geometry": "native_franka",
        "tabletop_control_mode": "joint_target_drive",
        "tabletop_initial_pose": str(args.tabletop_initial_pose),
        "tabletop_settle_seconds": float(args.tabletop_settle_seconds),
        "tabletop_pre_seconds": float(args.tabletop_pre_seconds),
        "tabletop_approach_seconds": float(args.tabletop_approach_seconds),
        "tabletop_contact_seconds": float(args.tabletop_contact_seconds),
        "tabletop_push_seconds": float(args.tabletop_push_seconds),
        "tabletop_retract_seconds": float(args.tabletop_retract_seconds),
        "tabletop_preroll_settle_seconds": float(args.tabletop_preroll_settle_seconds),
        "tabletop_reset_robot_after_preroll": bool(args.tabletop_reset_robot_after_preroll),
        "tabletop_robot_base_offset": [float(v) for v in np.asarray(args.tabletop_robot_base_offset, dtype=np.float32)],
        "tabletop_table_top_z": float(args.tabletop_table_top_z),
        "tabletop_support_box_mode": "none",
        "support_box_enabled": False,
        "support_box_physical": False,
        "visible_tool_enabled": False,
        "visible_tool_mode": "none",
        "particle_contacts": bool(meta["particle_contacts"]),
        "particle_contact_kernel": bool(meta["particle_contact_kernel"]),
        "particle_radius_scale": float(args.particle_radius_scale),
        "particle_radius_vis_scale": (None if args.particle_radius_vis_scale is None else float(args.particle_radius_vis_scale)),
        "particle_radius_vis_min": (None if args.particle_radius_vis_min is None else float(args.particle_radius_vis_min)),
        "gravity_mag_m_s2": float(args.gravity_mag),
        "apply_drag": bool(args.apply_drag),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "solver_type": str(meta.get("solver_type", args.solver_type)),
        "enable_gravcomp": bool(meta.get("enable_gravcomp", args.enable_gravcomp)),
        "solver_joint_attach_ke": float(args.solver_joint_attach_ke),
        "solver_joint_attach_kd": float(args.solver_joint_attach_kd),
        "joint_target_ke": float(args.joint_target_ke),
        "joint_target_kd": float(args.joint_target_kd),
        "finger_target_ke": float(args.finger_target_ke),
        "finger_target_kd": float(args.finger_target_kd),
        "proof_surface": "actual_multi_box_finger_colliders",
        "visible_rope_present": bool(sim_data["particle_q_object"].shape[1] > 0),
        "same_history_render_chain": bool(args.load_history_from_dir is not None or True),
    }
    summary.update(rope_metrics)
    return summary


def build_physics_validation(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": TABLETOP_PUSH_TASK,
        "blocking_stage": str(args.blocking_stage),
        "rope_present": bool(sim_data["particle_q_object"].shape[1] > 0),
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
    _resolve_camera_defaults(args, meta)

    if args.load_history_from_dir is not None:
        history_prefix = str(args.load_history_prefix or args.prefix)
        sim_data = legacy._load_saved_history(Path(args.load_history_from_dir), prefix=history_prefix, args=args, meta=meta)
        legacy._materialize_loaded_history(args, sim_data)
    else:
        sim_data = simulate(model, ir_obj, meta, args, n_obj, device)

    out_mp4 = args.out_dir / f"{args.prefix}.mp4"
    out_gif = None
    if not bool(args.skip_render):
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
