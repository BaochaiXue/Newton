#!/usr/bin/env python3
"""Native Newton Franka Panda manipulates a PhysTwin rope.

This demo keeps the deformable side on the PhysTwin -> Newton bridge and uses
a native Newton robotics asset:

- load one rope from PhysTwin IR
- keep it object-only and pin the rope endpoints
- add the native Franka Panda URDF through ``ModelBuilder.add_urdf()``
- drive the end-effector through a task preset using Newton IK
- solve the combined robot + rope scene with ``SolverSemiImplicit``

The goal is a taskful native robot-asset baseline that is easier to defend in a
meeting than a pure contact probe.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from bridge_shared import (
    BRIDGE_ROOT,
    CORE_DIR,
    apply_viewer_shape_colors,
    camera_position,
    compute_visual_particle_radii,
    load_core_module,
    ground_grid,
    overlay_text_lines_rgb,
    temporary_particle_radius_override,
)
from rollout_storage import RolloutStorage

WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
NEWTON_PY_ROOT = WORKSPACE_ROOT / "Newton" / "newton"
for _path in (CORE_DIR, NEWTON_PY_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

if not hasattr(wp, "quat_twist"):
    @wp.func
    def _quat_twist_compat(axis: wp.vec3, q: wp.quat) -> wp.quat:
        ax = wp.normalize(axis)
        imag = wp.vec3(q[0], q[1], q[2])
        proj = ax * wp.dot(imag, ax)
        return wp.normalize(wp.quat(proj[0], proj[1], proj[2], q[3]))

    wp.quat_twist = _quat_twist_compat

if not hasattr(wp, "quat_to_euler"):
    @wp.func
    def _quat_to_euler_compat(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = wp.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = wp.asin(wp.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = wp.atan2(siny_cosp, cosy_cosp)
        return wp.vec3(roll, pitch, yaw)

    wp.quat_to_euler = _quat_to_euler_compat

if not hasattr(wp, "transform_twist"):
    @wp.func
    def _transform_twist_compat(t: wp.transform, x: wp.spatial_vector) -> wp.spatial_vector:
        p = wp.transform_get_translation(t)
        v = wp.spatial_top(x)
        w = wp.spatial_bottom(x)
        w_out = wp.transform_vector(t, w)
        v_out = wp.transform_vector(t, v) + wp.cross(p, w_out)
        return wp.spatial_vector(v_out, w_out)

    wp.transform_twist = _transform_twist_compat

if not hasattr(wp, "transform_wrench"):
    @wp.func
    def _transform_wrench_compat(t: wp.transform, x: wp.spatial_vector) -> wp.spatial_vector:
        p = wp.transform_get_translation(t)
        f = wp.spatial_top(x)
        tau = wp.spatial_bottom(x)
        f_out = wp.transform_vector(t, f)
        tau_out = wp.transform_vector(t, tau) + wp.cross(p, f_out)
        return wp.spatial_vector(f_out, tau_out)

    wp.transform_wrench = _transform_wrench_compat

import newton  # noqa: E402

_bridge_bootstrap_stub = types.ModuleType("bridge_bootstrap")
_bridge_bootstrap_stub.newton = newton
_bridge_bootstrap_stub.newton_import_ir = None
_bridge_bootstrap_stub.path_defaults = None
def _ensure_bridge_runtime_paths() -> None:
    return None

_bridge_bootstrap_stub.ensure_bridge_runtime_paths = _ensure_bridge_runtime_paths
sys.modules["bridge_bootstrap"] = _bridge_bootstrap_stub

path_defaults = load_core_module("phystwin_bridge_path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = load_core_module("phystwin_bridge_newton_import_ir", CORE_DIR / "newton_import_ir.py")
_bridge_bootstrap_stub.newton_import_ir = newton_import_ir
_bridge_bootstrap_stub.path_defaults = path_defaults

import newton.ik as ik  # noqa: E402
import newton.utils  # noqa: E402

from bridge_deformable_common import (  # noqa: E402
    _apply_drag_correction_ignore_axis,
    _copy_object_only_ir,
    _effective_spring_scales,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    load_ir,
)
from rope_demo_common import (  # noqa: E402
    anchor_particle_indices as _anchor_particle_indices,
    quat_conjugate as _quat_conjugate,
    quat_multiply as _quat_multiply,
    resolve_particle_contact_settings as _resolve_particle_contact_settings,
    rope_endpoints as _rope_endpoints,
)

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
MID_SEGMENT_WINDOW_SIZE = 24
DROP_RELEASE_TASK = "drop_release_baseline"
TABLETOP_PUSH_TASK = "tabletop_push_hero"

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
        description="Native Newton Franka Panda manipulates a hanging PhysTwin rope."
    )
    p.add_argument("--ir", type=Path, default=_default_rope_ir(), help="Path to rope PhysTwin IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_push_rope_franka")
    p.add_argument("--device", default=path_defaults.default_device())
    p.add_argument(
        "--task",
        choices=["lift_release", "push_probe", "drop_release_baseline", TABLETOP_PUSH_TASK],
        default="lift_release",
        help=(
            "Task preset. `lift_release` is the meeting-ready default; "
            "`push_probe` keeps the older contact-probe behavior; "
            "`drop_release_baseline` is the stage-0 gravity drop sanity case; "
            f"`{TABLETOP_PUSH_TASK}` is the slow tabletop push hero demo."
        ),
    )
    p.add_argument(
        "--render-mode",
        choices=["debug", "presentation"],
        default="presentation",
        help="Render preset for internal debugging or meeting presentation.",
    )
    p.add_argument(
        "--camera-profile",
        choices=["hero", "validation"],
        default="hero",
        help="Camera preset to use when explicit camera parameters are not provided.",
    )
    p.add_argument(
        "--robot-motion-mode",
        choices=["ik", "replay"],
        default="ik",
        help=(
            "Robot motion source. `ik` uses the native Newton IK objective path. "
            "`replay` replays a native-robot body trajectory from a prior validated run "
            "while keeping the rope in the semi-implicit simulation."
        ),
    )
    p.add_argument(
        "--ik-target-blend",
        type=float,
        default=1.0,
        help=(
            "Blend factor applied to raw IK targets before commanding the robot. "
            "Lower values smooth phase transitions and reduce teleports."
        ),
    )
    p.add_argument(
        "--replay-source",
        type=Path,
        default=None,
        help=(
            "Directory containing `robot_push_rope_franka_body_q.npy` and related files for "
            "`--robot-motion-mode replay`."
        ),
    )

    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=1.0e-4)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument(
        "--history-storage",
        choices=["memory", "memmap"],
        default="memmap",
        help="Storage backend for rollout histories.",
    )
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
        "--particle-contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable rope particle self-contact. If omitted, follow the IR self_collision flag.",
    )
    p.add_argument(
        "--particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the Newton particle contact kernel when particle contacts are active.",
    )
    p.add_argument(
        "--drag-ignore-gravity-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply drag only orthogonal to gravity so free-fall acceleration is preserved.",
    )

    p.add_argument("--anchor-height", type=float, default=0.72)
    p.add_argument("--anchor-count-per-end", type=int, default=1)
    p.add_argument(
        "--anchor-mass-mode",
        choices=["preserve", "zero"],
        default="preserve",
        help=(
            "How to treat rope endpoint particle masses after marking them inactive. "
            "preserve keeps the original IR mass field; zero reproduces the older hard-anchor rewrite."
        ),
    )

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
    p.add_argument(
        "--lift-pre-seconds",
        type=float,
        default=0.40,
        help="Pre-approach hold time for `lift_release` [s].",
    )
    p.add_argument(
        "--lift-approach-seconds",
        type=float,
        default=0.80,
        help="Approach phase time for `lift_release` [s].",
    )
    p.add_argument(
        "--lift-seconds",
        type=float,
        default=0.75,
        help="Lift/pull phase time for `lift_release` [s].",
    )
    p.add_argument(
        "--lift-hold-seconds",
        type=float,
        default=0.45,
        help="Hold phase time for `lift_release` [s].",
    )
    p.add_argument(
        "--lift-release-seconds",
        type=float,
        default=0.80,
        help="Release/retract phase time for `lift_release` [s].",
    )
    p.add_argument(
        "--drop-approach-seconds",
        type=float,
        default=0.45,
        help="Approach-to-support phase time for `drop_release_baseline` [s].",
    )
    p.add_argument(
        "--drop-support-seconds",
        type=float,
        default=0.30,
        help="Static support phase before release for `drop_release_baseline` [s].",
    )
    p.add_argument(
        "--drop-release-seconds",
        type=float,
        default=0.12,
        help="Finger-open release phase for `drop_release_baseline` [s].",
    )
    p.add_argument(
        "--drop-freefall-seconds",
        type=float,
        default=1.60,
        help="Free-fall and settling phase after release for `drop_release_baseline` [s].",
    )
    p.add_argument(
        "--gripper-hold",
        type=float,
        default=0.014,
        help="Finger target position while the rope is supported.",
    )
    p.add_argument(
        "--drop-support-patch-count",
        type=int,
        default=8,
        help="Number of particles constrained in the visible support patch for `drop_release_baseline`.",
    )
    p.add_argument(
        "--settle-window-seconds",
        type=float,
        default=0.08,
        help="Required pre-release settle window length [s].",
    )
    p.add_argument(
        "--pre-release-particle-speed-mean-max",
        type=float,
        default=0.05,
        help="Maximum allowed rope particle mean speed over the settle window [m/s].",
    )
    p.add_argument(
        "--pre-release-com-horizontal-speed-max",
        type=float,
        default=0.03,
        help="Maximum allowed rope COM horizontal speed over the settle window [m/s].",
    )
    p.add_argument(
        "--pre-release-support-speed-mean-max",
        type=float,
        default=0.02,
        help="Maximum allowed support-patch mean speed over the settle window [m/s].",
    )
    p.add_argument(
        "--post-release-kick-window-seconds",
        type=float,
        default=0.08,
        help="Post-release window used for horizontal-kick diagnostics [s].",
    )
    p.add_argument(
        "--drop-preroll-settle-seconds",
        type=float,
        default=2.0,
        help="Maximum hidden settle time before the recorded support/release sequence starts [s].",
    )
    p.add_argument(
        "--tabletop-settle-seconds",
        type=float,
        default=0.9,
        help="Initial stillness window before the tabletop push starts [s].",
    )
    p.add_argument(
        "--tabletop-approach-seconds",
        type=float,
        default=1.2,
        help="Approach-to-contact phase for the tabletop push hero [s].",
    )
    p.add_argument(
        "--tabletop-push-seconds",
        type=float,
        default=2.6,
        help="Slow lateral push phase for the tabletop push hero [s].",
    )
    p.add_argument(
        "--tabletop-hold-seconds",
        type=float,
        default=0.8,
        help="Short hold phase after the tabletop push reaches the target [s].",
    )
    p.add_argument(
        "--tabletop-retract-seconds",
        type=float,
        default=1.1,
        help="Retract-and-settle phase after the tabletop push [s].",
    )
    p.add_argument(
        "--tabletop-rope-height",
        type=float,
        default=0.135,
        help="Target rope center height used for the tabletop push hero [m].",
    )
    p.add_argument(
        "--tabletop-table-top-z",
        type=float,
        default=0.120,
        help="World-space tabletop surface height [m].",
    )
    p.add_argument(
        "--tabletop-table-hx",
        type=float,
        default=0.42,
        help="Table half-extent in X for the tabletop push hero [m].",
    )
    p.add_argument(
        "--tabletop-table-hy",
        type=float,
        default=0.24,
        help="Table half-extent in Y for the tabletop push hero [m].",
    )
    p.add_argument(
        "--tabletop-table-hz",
        type=float,
        default=0.020,
        help="Table thickness half-extent in Z for the tabletop push hero [m].",
    )
    p.add_argument(
        "--tabletop-robot-base-offset",
        type=float,
        nargs=3,
        default=(0.56, -0.34, 0.10),
        metavar=("X", "Y", "Z"),
        help="Robot base offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--tabletop-push-start-offset",
        type=float,
        nargs=3,
        default=(-0.28, -0.12, 0.11),
        metavar=("X", "Y", "Z"),
        help="Approach start offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--tabletop-push-contact-offset",
        type=float,
        nargs=3,
        default=(-0.10, -0.06, 0.02),
        metavar=("X", "Y", "Z"),
        help="Contact-ready offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--tabletop-push-end-offset",
        type=float,
        nargs=3,
        default=(0.18, -0.06, 0.02),
        metavar=("X", "Y", "Z"),
        help="Push end offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--tabletop-retract-offset",
        type=float,
        nargs=3,
        default=(0.24, -0.20, 0.18),
        metavar=("X", "Y", "Z"),
        help="Retract offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--pre-release-settle-damping-scale",
        type=float,
        default=8.0,
        help="Extra pre-release orthogonal-to-gravity damping scale used only while the support patch is still constrained.",
    )

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=None)
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
        default=None,
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=None)
    p.add_argument("--camera-yaw", type=float, default=None)
    p.add_argument("--camera-fov", type=float, default=None)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.004)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--label-font-size", type=int, default=24)
    p.add_argument("--rope-line-width", type=float, default=0.02)
    p.add_argument("--spring-stride", type=int, default=20)
    return p.parse_args()


def _total_task_duration(args: argparse.Namespace) -> float:
    if str(args.task) == "lift_release":
        return float(
            float(args.lift_pre_seconds)
            + float(args.lift_approach_seconds)
            + float(args.lift_seconds)
            + float(args.lift_hold_seconds)
            + float(args.lift_release_seconds)
        )
    if str(args.task) == DROP_RELEASE_TASK:
        return float(
            float(args.drop_approach_seconds)
            + float(args.drop_support_seconds)
            + float(args.drop_release_seconds)
            + float(args.drop_freefall_seconds)
        )
    if str(args.task) == TABLETOP_PUSH_TASK:
        return float(
            float(args.tabletop_settle_seconds)
            + float(args.tabletop_approach_seconds)
            + float(args.tabletop_push_seconds)
            + float(args.tabletop_hold_seconds)
            + float(args.tabletop_retract_seconds)
        )
    return float(
        float(args.settle_seconds)
        + float(args.push_seconds)
        + float(args.hold_seconds)
        + max(float(args.push_seconds), 0.06)
    )


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


def _quat_relative(q_curr: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    dq = _quat_multiply(np.asarray(q_curr, dtype=np.float32), _quat_conjugate(np.asarray(q_prev, dtype=np.float32)))
    if float(dq[3]) < 0.0:
        dq = -dq
    return dq.astype(np.float32, copy=False)


def _quat_delta_to_angular_velocity(q_curr: np.ndarray, q_prev: np.ndarray, dt: float) -> np.ndarray:
    dq = _quat_relative(q_curr, q_prev)
    vec = np.asarray(dq[:3], dtype=np.float32)
    w = float(np.clip(dq[3], -1.0, 1.0))
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1.0e-8 or dt <= 0.0:
        return np.zeros((3,), dtype=np.float32)
    angle = 2.0 * np.arctan2(vec_norm, w)
    axis = vec / vec_norm
    return (axis * (angle / dt)).astype(np.float32, copy=False)


def _resample_indices(src_frames: int, dst_frames: int) -> np.ndarray:
    if src_frames <= 1 or dst_frames <= 1:
        return np.zeros((dst_frames,), dtype=np.int32)
    return np.clip(
        np.rint(np.linspace(0.0, float(src_frames - 1), int(dst_frames), endpoint=True)).astype(np.int32),
        0,
        src_frames - 1,
    )


def _load_replay_trajectory(source_dir: Path, n_frames: int, frame_dt: float) -> dict[str, np.ndarray]:
    source_dir = Path(source_dir).resolve()
    body_q_path = source_dir / "robot_push_rope_franka_body_q.npy"
    target_path = source_dir / "robot_push_rope_franka_ee_target_pos.npy"
    if not body_q_path.exists():
        raise FileNotFoundError(f"Replay source missing body trajectory: {body_q_path}")

    body_q_src = np.asarray(np.load(body_q_path), dtype=np.float32)
    if body_q_src.ndim != 3 or body_q_src.shape[-1] != 7:
        raise ValueError(f"Unexpected replay body_q shape: {body_q_src.shape}")
    sample_idx = _resample_indices(int(body_q_src.shape[0]), int(n_frames))
    body_q = body_q_src[sample_idx].astype(np.float32, copy=False)

    body_qd = np.zeros((int(n_frames), body_q.shape[1], 6), dtype=np.float32)
    if int(n_frames) > 1:
        for frame in range(1, int(n_frames)):
            prev = body_q[frame - 1]
            curr = body_q[frame]
            body_qd[frame, :, :3] = (curr[:, :3] - prev[:, :3]) / float(frame_dt)
            for body_idx in range(body_q.shape[1]):
                body_qd[frame, body_idx, 3:] = _quat_delta_to_angular_velocity(
                    curr[body_idx, 3:7],
                    prev[body_idx, 3:7],
                    float(frame_dt),
                )
        body_qd[0] = body_qd[1]

    if target_path.exists():
        ee_target_src = np.asarray(np.load(target_path), dtype=np.float32)
        ee_target = ee_target_src[_resample_indices(int(ee_target_src.shape[0]), int(n_frames))].astype(np.float32, copy=False)
    else:
        ee_target = np.zeros((int(n_frames), 3), dtype=np.float32)

    return {
        "source_dir": str(source_dir),
        "body_q": body_q,
        "body_qd": body_qd,
        "ee_target_pos": ee_target,
    }


def _gripper_contact_proxies_world(
    body_q: np.ndarray,
    left_finger_idx: int,
    right_finger_idx: int,
) -> dict[str, np.ndarray]:
    body_q = np.asarray(body_q, dtype=np.float32)
    left = body_q[int(left_finger_idx), :3]
    right = body_q[int(right_finger_idx), :3]
    center = 0.5 * (left + right)
    return {
        "gripper_center": np.asarray(center, dtype=np.float32),
        "left_finger": np.asarray(left, dtype=np.float32),
        "right_finger": np.asarray(right, dtype=np.float32),
    }


def _gripper_contact_proxy_radii(contact_radius: float) -> dict[str, float]:
    contact_radius = float(contact_radius)
    effective_radius = max(0.012, 0.60 * contact_radius)
    return {
        "gripper_center": effective_radius,
        "left_finger": effective_radius,
        "right_finger": effective_radius,
        "finger_span": effective_radius,
    }


def _point_segment_min_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1.0e-12:
        return float(np.min(np.linalg.norm(points - start[None, :], axis=1)))
    t = np.clip(((points - start[None, :]) @ segment) / denom, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * segment[None, :]
    return float(np.min(np.linalg.norm(points - proj, axis=1)))


def _min_gripper_proxy_clearance(
    particle_q: np.ndarray,
    particle_radius: np.ndarray,
    proxy_positions: dict[str, np.ndarray],
    contact_radius: float,
) -> tuple[float, str, dict[str, float]]:
    proxy_radii = _gripper_contact_proxy_radii(contact_radius)
    per_proxy_min: dict[str, float] = {}
    best_name = "gripper_center"
    best_clearance = float("inf")
    for name, proxy_pos in proxy_positions.items():
        d = np.linalg.norm(particle_q - proxy_pos[None, :], axis=1) - (particle_radius + proxy_radii.get(name, float(contact_radius)))
        min_d = float(np.min(d))
        per_proxy_min[name] = min_d
        if min_d < best_clearance:
            best_clearance = min_d
            best_name = name
    left = proxy_positions.get("left_finger")
    right = proxy_positions.get("right_finger")
    if left is not None and right is not None:
        span_min_d = _point_segment_min_distance(particle_q, left, right) - (
            float(np.min(particle_radius)) + proxy_radii["finger_span"]
        )
        per_proxy_min["finger_span"] = float(span_min_d)
        if span_min_d < best_clearance:
            best_clearance = float(span_min_d)
            best_name = "finger_span"
    return best_clearance, best_name, per_proxy_min


def _fit_quadratic_acceleration(times: np.ndarray, values: np.ndarray) -> float | None:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size < 3 or values.size < 3:
        return None
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        return None
    try:
        coeff = np.polyfit(times, values, deg=2)
    except Exception:
        return None
    return float(2.0 * coeff[0])


def _phase_start_time(phases: list[dict[str, Any]], phase_name: str) -> float | None:
    elapsed = 0.0
    for phase in phases:
        if str(phase.get("name")) == str(phase_name):
            return float(elapsed)
        elapsed += float(phase.get("duration", 0.0))
    return None


def _activate_particle_indices(model: newton.Model, particle_indices: np.ndarray, *, device: str) -> None:
    particle_indices = np.asarray(particle_indices, dtype=np.int32).reshape(-1)
    if particle_indices.size == 0:
        return
    active_mask = int(newton.ParticleFlags.ACTIVE)
    flags = model.particle_flags.numpy().astype(np.int32, copy=True)
    flags[particle_indices] = flags[particle_indices] | active_mask
    model.particle_flags = wp.array(flags, dtype=wp.int32, device=device)


def _select_drop_support_patch_indices(points: np.ndarray, *, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected point shape for support selection: {pts.shape}")
    count = max(1, int(count))
    candidate_count = min(int(pts.shape[0]), max(24, 4 * count))
    top_idx = np.argsort(-pts[:, 2])[:candidate_count].astype(np.int32)
    top_pts = pts[top_idx]
    seed_count = min(int(top_pts.shape[0]), max(4, count))
    seed_center = np.mean(top_pts[:seed_count], axis=0).astype(np.float32, copy=False)
    d = np.linalg.norm(top_pts - seed_center[None, :], axis=1)
    chosen = top_idx[np.argsort(d)[: min(count, top_idx.shape[0])]]
    return np.sort(chosen.astype(np.int32, copy=False))


def _settle_window_metrics_from_series(
    particle_speed_mean_series: np.ndarray,
    rope_com_horizontal_speed_series: np.ndarray,
    support_patch_speed_mean_series: np.ndarray,
    *,
    frame_idx: int,
    window_frames: int,
    particle_speed_threshold_m_s: float,
    com_horizontal_speed_threshold_m_s: float,
    support_patch_speed_threshold_m_s: float,
    frame_dt: float,
) -> tuple[dict[str, Any], bool]:
    start = max(0, int(frame_idx) - int(window_frames) + 1)
    metrics = {
        "window_start_frame": int(start),
        "window_end_frame": int(frame_idx),
        "window_duration_s": float((int(frame_idx) - start + 1) * frame_dt),
        "particle_speed_mean_max_m_s": float(np.max(particle_speed_mean_series[start : int(frame_idx) + 1])),
        "com_horizontal_speed_max_m_s": float(np.max(rope_com_horizontal_speed_series[start : int(frame_idx) + 1])),
        "support_patch_speed_mean_max_m_s": float(np.max(support_patch_speed_mean_series[start : int(frame_idx) + 1])),
        "particle_speed_mean_threshold_m_s": float(particle_speed_threshold_m_s),
        "com_horizontal_speed_threshold_m_s": float(com_horizontal_speed_threshold_m_s),
        "support_patch_speed_threshold_m_s": float(support_patch_speed_threshold_m_s),
    }
    settle_ok = (
        (int(frame_idx) - start + 1) >= int(window_frames)
        and metrics["particle_speed_mean_max_m_s"] <= float(particle_speed_threshold_m_s)
        and metrics["com_horizontal_speed_max_m_s"] <= float(com_horizontal_speed_threshold_m_s)
        and metrics["support_patch_speed_mean_max_m_s"] <= float(support_patch_speed_threshold_m_s)
    )
    return metrics, bool(settle_ok)


def _camera_presets(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rope_center = np.asarray(meta["rope_center"], dtype=np.float32)
    stage_center = np.asarray(meta["stage_center"], dtype=np.float32)
    stage_scale = np.asarray(meta["stage_scale"], dtype=np.float32)
    anchor_bar_center = np.asarray(meta["anchor_bar_center"], dtype=np.float32)
    task = str(meta.get("task", ""))
    floor_z = float(meta.get("floor_z", 0.0))
    support_point = np.asarray(meta.get("support_point", anchor_bar_center), dtype=np.float32)
    table_center = np.asarray(meta.get("table_center", stage_center), dtype=np.float32)

    if task == DROP_RELEASE_TASK:
        hero_target = np.asarray(
            [
                float(0.62 * rope_center[0] + 0.38 * support_point[0]),
                float(0.62 * rope_center[1] + 0.38 * support_point[1] - 0.02),
                float(max(support_point[2] + 0.05, floor_z + 0.20)),
            ],
            dtype=np.float32,
        )
        hero_pos = camera_position(hero_target, yaw_deg=-36.0, pitch_deg=-18.0, distance=1.62)

        validation_target = np.asarray(
            [
                float(support_point[0]),
                float(support_point[1]),
                float(max(floor_z + 0.16, support_point[2] - 0.02)),
            ],
            dtype=np.float32,
        )
        validation_pos = camera_position(validation_target, yaw_deg=-42.0, pitch_deg=-12.0, distance=1.95)

        return {
            "hero": {
                "pos": hero_pos.astype(np.float32, copy=False).tolist(),
                "pitch": -18.0,
                "yaw": -36.0,
                "fov": 35.0,
                "target": hero_target.astype(np.float32, copy=False).tolist(),
            },
            "validation": {
                "pos": validation_pos.astype(np.float32, copy=False).tolist(),
                "pitch": -12.0,
                "yaw": -42.0,
                "fov": 46.0,
                "target": validation_target.astype(np.float32, copy=False).tolist(),
            },
        }

    if task == TABLETOP_PUSH_TASK:
        push_focus = np.asarray(meta.get("tabletop_push_focus", rope_center), dtype=np.float32)
        hero_target = np.asarray(
            [
                float(0.42 * table_center[0] + 0.58 * push_focus[0] - 0.08),
                float(0.42 * table_center[1] + 0.58 * push_focus[1] - 0.16),
                float(max(table_center[2] + 0.20, rope_center[2] + 0.08)),
            ],
            dtype=np.float32,
        )
        hero_pos = camera_position(hero_target, yaw_deg=-28.0, pitch_deg=-22.0, distance=1.70)

        validation_target = np.asarray(
            [
                float(push_focus[0]),
                float(push_focus[1]),
                float(max(table_center[2] + 0.14, rope_center[2] + 0.04)),
            ],
            dtype=np.float32,
        )
        validation_pos = camera_position(validation_target, yaw_deg=142.0, pitch_deg=-20.0, distance=1.85)

        return {
            "hero": {
                "pos": hero_pos.astype(np.float32, copy=False).tolist(),
                "pitch": -22.0,
                "yaw": -28.0,
                "fov": 34.0,
                "target": hero_target.astype(np.float32, copy=False).tolist(),
            },
            "validation": {
                "pos": validation_pos.astype(np.float32, copy=False).tolist(),
                "pitch": -20.0,
                "yaw": 142.0,
                "fov": 46.0,
                "target": validation_target.astype(np.float32, copy=False).tolist(),
            },
        }

    hero_target = np.asarray(
        [
            float(0.65 * rope_center[0] + 0.35 * stage_center[0]),
            float(0.65 * rope_center[1] + 0.35 * stage_center[1] - 0.03),
            float(max(rope_center[2] + 0.10, stage_center[2] + 0.12, anchor_bar_center[2] - 0.02)),
        ],
        dtype=np.float32,
    )
    hero_pos = camera_position(hero_target, yaw_deg=-34.0, pitch_deg=-17.0, distance=1.48)

    validation_target = np.asarray(
        [
            float(stage_center[0] + 0.03),
            float(stage_center[1] - 0.01),
            float(max(stage_center[2] + 0.08, anchor_bar_center[2] - 0.03)),
        ],
        dtype=np.float32,
    )
    validation_distance = 1.92 + 0.20 * float(stage_scale[0])
    validation_pos = camera_position(validation_target, yaw_deg=-39.0, pitch_deg=-10.0, distance=validation_distance)

    return {
        "hero": {
            "pos": hero_pos.astype(np.float32, copy=False).tolist(),
            "pitch": -17.0,
            "yaw": -34.0,
            "fov": 36.0,
            "target": hero_target.astype(np.float32, copy=False).tolist(),
        },
        "validation": {
            "pos": validation_pos.astype(np.float32, copy=False).tolist(),
            "pitch": -10.0,
            "yaw": -39.0,
            "fov": 48.0,
            "target": validation_target.astype(np.float32, copy=False).tolist(),
        },
    }


def _resolve_runtime_defaults(args: argparse.Namespace) -> None:
    if args.frames is None:
        frame_dt = float(args.sim_dt) * max(1, int(args.substeps))
        total_duration = _total_task_duration(args)
        args.frames = max(120, int(np.ceil(total_duration / max(frame_dt, 1.0e-8))) + 1)

    if args.slowdown is None:
        args.slowdown = 1.0

    if args.overlay_label is None:
        args.overlay_label = str(args.render_mode) == "debug"


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


def _mid_segment_indices(points: np.ndarray, rope_center: np.ndarray, count: int = MID_SEGMENT_WINDOW_SIZE) -> np.ndarray:
    d = np.linalg.norm(np.asarray(points, dtype=np.float32) - rope_center[None, :], axis=1)
    return np.argsort(d)[: max(1, int(count))].astype(np.int32)


def _task_phase_definitions(rope_center: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    rope_center = np.asarray(rope_center, dtype=np.float32)
    target_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if str(args.task) == "lift_release":
        pre = rope_center + np.asarray([-0.24, 0.00, 0.12], dtype=np.float32)
        approach = rope_center + np.asarray([0.02, 0.00, -0.018], dtype=np.float32)
        lift = rope_center + np.asarray([0.15, -0.08, 0.14], dtype=np.float32)
        release = rope_center + np.asarray([0.24, -0.18, 0.18], dtype=np.float32)
        return [
            {
                "name": "pre_approach",
                "duration": float(args.lift_pre_seconds),
                "start": pre,
                "end": pre,
                "quat": target_quat,
            },
            {
                "name": "approach_under",
                "duration": float(args.lift_approach_seconds),
                "start": pre,
                "end": approach,
                "quat": target_quat,
            },
            {
                "name": "lift",
                "duration": float(args.lift_seconds),
                "start": approach,
                "end": lift,
                "quat": target_quat,
            },
            {
                "name": "hold",
                "duration": float(args.lift_hold_seconds),
                "start": lift,
                "end": lift,
                "quat": target_quat,
            },
            {
                "name": "release_retract",
                "duration": float(args.lift_release_seconds),
                "start": lift,
                "end": release,
                "quat": target_quat,
            },
        ]

    if str(args.task) == DROP_RELEASE_TASK:
        approach = rope_center + np.asarray([-0.18, 0.00, 0.14], dtype=np.float32)
        support = rope_center + np.asarray([-0.04, 0.00, 0.08], dtype=np.float32)
        return [
            {
                "name": "approach_support",
                "duration": float(args.drop_approach_seconds),
                "start": approach,
                "end": support,
                "quat": target_quat,
            },
            {
                "name": "support_hold",
                "duration": float(args.drop_support_seconds),
                "start": support,
                "end": support,
                "quat": target_quat,
            },
            {
                "name": "release",
                "duration": float(args.drop_release_seconds),
                "start": support,
                "end": support,
                "quat": target_quat,
            },
            {
                "name": "free_fall",
                "duration": float(args.drop_freefall_seconds),
                "start": support,
                "end": support,
                "quat": target_quat,
            },
        ]

    if str(args.task) == TABLETOP_PUSH_TASK:
        settle = rope_center + np.asarray([-0.22, -0.16, 0.14], dtype=np.float32)
        approach = settle + np.asarray([0.08, 0.04, -0.03], dtype=np.float32)
        contact_ready = rope_center + np.asarray(np.asarray(args.tabletop_push_contact_offset, dtype=np.float32), dtype=np.float32)
        push_end = rope_center + np.asarray(np.asarray(args.tabletop_push_end_offset, dtype=np.float32), dtype=np.float32)
        retract = rope_center + np.asarray(np.asarray(args.tabletop_retract_offset, dtype=np.float32), dtype=np.float32)
        return [
            {
                "name": "settle",
                "duration": float(args.tabletop_settle_seconds),
                "start": settle,
                "end": settle,
                "quat": target_quat,
            },
            {
                "name": "approach",
                "duration": float(args.tabletop_approach_seconds),
                "start": settle,
                "end": approach,
                "quat": target_quat,
            },
            {
                "name": "push",
                "duration": float(args.tabletop_push_seconds),
                "start": contact_ready,
                "end": push_end,
                "quat": target_quat,
            },
            {
                "name": "hold",
                "duration": float(args.tabletop_hold_seconds),
                "start": push_end,
                "end": push_end,
                "quat": target_quat,
            },
            {
                "name": "retract",
                "duration": float(args.tabletop_retract_seconds),
                "start": push_end,
                "end": retract,
                "quat": target_quat,
            },
        ]

    start = rope_center + np.asarray([float(args.ee_start_x_offset), 0.0, float(args.ee_z_offset)], dtype=np.float32)
    end = rope_center + np.asarray([float(args.ee_end_x_offset), 0.0, float(args.ee_z_offset)], dtype=np.float32)
    retract = start.copy()
    return [
        {"name": "pre_approach", "duration": float(args.settle_seconds), "start": start, "end": start, "quat": target_quat},
        {"name": "push", "duration": float(args.push_seconds), "start": start, "end": end, "quat": target_quat},
        {"name": "hold", "duration": float(args.hold_seconds), "start": end, "end": end, "quat": target_quat},
        {
            "name": "release_retract",
            "duration": max(float(args.push_seconds), 0.06),
            "start": end,
            "end": retract,
            "quat": target_quat,
        },
    ]


def _task_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    phases = meta["task_phases"]
    elapsed = 0.0
    for phase in phases:
        duration = max(float(phase["duration"]), 1.0e-8)
        end_t = elapsed + duration
        if t <= end_t:
            alpha = 0.0 if phase["name"] == "pre_approach" or np.allclose(phase["start"], phase["end"]) else np.clip((t - elapsed) / duration, 0.0, 1.0)
            pos = (1.0 - alpha) * np.asarray(phase["start"], dtype=np.float32) + alpha * np.asarray(phase["end"], dtype=np.float32)
            quat = np.asarray(phase["quat"], dtype=np.float32)
            return str(phase["name"]), pos.astype(np.float32, copy=False), quat
        elapsed = end_t

    last = phases[-1]
    return str(last["name"]), np.asarray(last["end"], dtype=np.float32), np.asarray(last["quat"], dtype=np.float32)


def _find_index_by_suffix(labels: list[str], suffix: str) -> int:
    for idx, label in enumerate(labels):
        if label.endswith(suffix):
            return idx
    raise KeyError(f"Could not find label suffix {suffix!r} in {labels}")


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
    raw_ir = load_ir(args.ir)
    drop_release_task = str(args.task) == DROP_RELEASE_TASK
    _validate_scaling_args(args)
    _maybe_autoset_mass_spring_scale(args, raw_ir)
    ir_obj = _copy_object_only_ir(raw_ir, args)

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)

    endpoint_indices = _rope_endpoints(edges, n_obj, x0)
    endpoint_mid = 0.5 * (x0[int(endpoint_indices[0])] + x0[int(endpoint_indices[1])])
    support_patch_indices = np.empty((0,), dtype=np.int32)
    support_patch_center = endpoint_mid.astype(np.float32, copy=False)
    shift = np.array(
        [
            -float(endpoint_mid[0]),
            -float(endpoint_mid[1]),
            float(args.anchor_height) - float(endpoint_mid[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    rope_center = shifted_q.mean(axis=0).astype(np.float32, copy=False)
    if drop_release_task:
        anchor_indices = _anchor_particle_indices(
            shifted_q, endpoint_indices=endpoint_indices, count_per_end=int(args.anchor_count_per_end)
        )
        anchor_positions = shifted_q[anchor_indices].astype(np.float32, copy=False)
    else:
        anchor_indices = _anchor_particle_indices(
            shifted_q, endpoint_indices=endpoint_indices, count_per_end=int(args.anchor_count_per_end)
        )
        anchor_positions = shifted_q[anchor_indices].astype(np.float32, copy=False)

    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    particle_contacts, particle_contact_kernel = _resolve_particle_contact_settings(ir_obj, args)
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
    _, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=bool(particle_contacts))
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]
    for idx in anchor_indices.tolist():
        builder.particle_flags[idx] = int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)
        if (not drop_release_task) and str(args.anchor_mass_mode) == "zero":
            builder.particle_mass[idx] = 0.0

    support_patch_center_shifted = np.mean(anchor_positions, axis=0).astype(np.float32, copy=False) if anchor_positions.size else rope_center
    robot_reference_point = support_patch_center_shifted if drop_release_task else rope_center
    robot_base_pos = robot_reference_point + np.asarray(args.robot_base_offset, dtype=np.float32)
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
    builder.joint_target_pos[7:9] = [float(args.gripper_hold if drop_release_task else args.gripper_open)] * 2
    builder.joint_armature[:7] = [0.3] * 4 + [0.11] * 3
    builder.joint_armature[7:9] = [0.15] * 2
    builder.joint_effort_limit[:7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    builder.joint_effort_limit[7:9] = [20.0, 20.0]

    ee_body_index = _find_index_by_suffix(builder.body_label, "/fr3_link7")
    left_finger_index = _find_index_by_suffix(builder.body_label, "/fr3_leftfinger")
    right_finger_index = _find_index_by_suffix(builder.body_label, "/fr3_rightfinger")
    ee_offset_local = np.asarray([0.0, 0.0, 0.22], dtype=np.float32)
    floor_z = 0.0
    if drop_release_task:
        support_anchor_center = support_patch_center_shifted.astype(np.float32, copy=False)
        support_point = np.asarray(
            [
                float(support_anchor_center[0]),
                float(support_anchor_center[1]),
                float(support_anchor_center[2] + 0.015),
            ],
            dtype=np.float32,
        )
        anchor_bar_center = np.asarray(
            [
                float(support_anchor_center[0]),
                float(support_anchor_center[1]),
                float(max(support_anchor_center[2] + 0.025, floor_z + 0.18)),
            ],
            dtype=np.float32,
        )
        anchor_bar_scale = np.asarray(
            [
                max(0.012, 0.5 * float(np.max(anchor_positions[:, 0]) - np.min(anchor_positions[:, 0])) + 0.015),
                max(0.028, 0.5 * float(np.max(anchor_positions[:, 1]) - np.min(anchor_positions[:, 1])) + 0.020),
                0.010,
            ],
            dtype=np.float32,
        )
    else:
        anchor_bar_center = np.asarray(
            [
                float(np.mean(anchor_positions, axis=0)[0]),
                float(np.mean(anchor_positions, axis=0)[1]),
                float(np.max(anchor_positions[:, 2]) + 0.04),
            ],
            dtype=np.float32,
        )
        anchor_bar_scale = np.asarray(
            [
                0.028,
                max(0.11, 0.5 * float(np.max(anchor_positions[:, 1]) - np.min(anchor_positions[:, 1])) + 0.05),
                0.020,
            ],
            dtype=np.float32,
        )
        support_point = anchor_bar_center
    task_phases = _task_phase_definitions(rope_center, args)
    if drop_release_task:
        task_phases[0]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[1]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[1]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[2]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[2]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[3]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[3]["end"] = support_point.astype(np.float32, copy=False)
    ee_target_end = np.asarray(task_phases[-1]["end"], dtype=np.float32)
    ee_target_quat = np.asarray(task_phases[0]["quat"], dtype=np.float32)
    mid_segment_indices = _mid_segment_indices(shifted_q[:n_obj], rope_center)
    stage_center = np.asarray(
        [float(rope_center[0]), float(rope_center[1]), float(floor_z + (0.16 if drop_release_task else 0.18))],
        dtype=np.float32,
    )
    stage_scale = np.asarray([0.28, 0.18, 0.05], dtype=np.float32)
    robot_base_scale = np.asarray([0.17, 0.17, max(0.18, float(robot_base_pos[2] - floor_z))], dtype=np.float32)
    robot_base_center = np.asarray(
        [
            float(robot_base_pos[0]),
            float(robot_base_pos[1]),
            float(floor_z + robot_base_scale[2]),
        ],
        dtype=np.float32,
    )
    if drop_release_task:
        anchor_left_center = np.zeros((3,), dtype=np.float32)
        anchor_right_center = np.zeros((3,), dtype=np.float32)
        floor_scale = np.asarray([1.45, 1.10, 0.012], dtype=np.float32)
        floor_center = np.asarray([0.0, 0.0, float(floor_z - floor_scale[2])], dtype=np.float32)
        anchor_post_scale = np.asarray([0.024, 0.024, 0.024], dtype=np.float32)
    else:
        anchor_post_height = max(0.18, float(anchor_bar_center[2] - floor_z))
        anchor_post_scale = np.asarray([0.024, 0.024, anchor_post_height], dtype=np.float32)
        anchor_left_center = np.asarray(
            [
                float(anchor_positions[0, 0]),
                float(anchor_positions[0, 1]),
                float(floor_z + anchor_post_height),
            ],
            dtype=np.float32,
        )
        anchor_right_center = np.asarray(
            [
                float(anchor_positions[-1, 0]),
                float(anchor_positions[-1, 1]),
                float(floor_z + anchor_post_height),
            ],
            dtype=np.float32,
        )
        floor_center = np.asarray([0.0, 0.0, float(floor_z - 0.01)], dtype=np.float32)
        floor_scale = np.asarray([1.45, 1.10, 0.012], dtype=np.float32)
    camera_presets = _camera_presets(
        {
            "rope_center": rope_center,
            "stage_center": stage_center,
            "stage_scale": stage_scale,
            "anchor_bar_center": anchor_bar_center,
            "task": str(args.task),
            "support_point": support_point,
            "floor_z": float(floor_z),
        }
    )

    if drop_release_task:
        ground_cfg = builder.default_shape_cfg.copy()
        newton_import_ir._configure_ground_contact_material(
            ground_cfg,
            ir_obj,
            cfg,
            checks,
            context="ground_floor_box",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*floor_center.tolist()), wp.quat_identity()),
            hx=float(floor_scale[0]),
            hy=float(floor_scale[1]),
            hz=float(floor_scale[2]),
            cfg=ground_cfg,
            label="ground_floor_box",
        )
    model = builder.finalize(device=device)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    if not bool(particle_contact_kernel):
        model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    init_state = model.state()
    init_state.joint_q.assign(FRANKA_INIT_Q.astype(np.float32))
    init_state.joint_qd.zero_()
    newton.eval_fk(model, init_state.joint_q, init_state.joint_qd, init_state)
    init_body_q = init_state.body_q.numpy().astype(np.float32)
    ee_target_start = _ee_world_position(init_body_q[int(ee_body_index)], ee_offset_local)
    task_phases[0]["start"] = ee_target_start.astype(np.float32, copy=False)
    if not drop_release_task:
        task_phases[0]["end"] = ee_target_start.astype(np.float32, copy=False)
        task_phases[1]["start"] = ee_target_start.astype(np.float32, copy=False)

    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "robot_geometry": "native_franka",
        "robot_motion_mode": str(args.robot_motion_mode),
        "robot_base_pos": robot_base_pos.astype(np.float32),
        "ee_body_index": int(ee_body_index),
        "left_finger_index": int(left_finger_index),
        "right_finger_index": int(right_finger_index),
        "ee_offset_local": ee_offset_local.astype(np.float32),
        "ee_target_quat": ee_target_quat.astype(np.float32),
        "ee_target_start": ee_target_start.astype(np.float32),
        "ee_target_end": ee_target_end.astype(np.float32),
        "task": str(args.task),
        "task_phases": task_phases,
        "mid_segment_indices": mid_segment_indices.astype(np.int32),
        "anchor_bar_center": anchor_bar_center.astype(np.float32),
        "anchor_bar_scale": anchor_bar_scale.astype(np.float32),
        "anchor_left_center": anchor_left_center.astype(np.float32),
        "anchor_right_center": anchor_right_center.astype(np.float32),
        "anchor_post_scale": anchor_post_scale.astype(np.float32),
        "stage_center": stage_center.astype(np.float32),
        "stage_scale": stage_scale.astype(np.float32),
        "robot_base_center": robot_base_center.astype(np.float32),
        "robot_base_scale": robot_base_scale.astype(np.float32),
        "floor_center": floor_center.astype(np.float32),
        "floor_scale": floor_scale.astype(np.float32),
        "floor_z": float(floor_z),
        "camera_presets": camera_presets,
        "joint_q_init": FRANKA_INIT_Q.astype(np.float32),
        "endpoint_indices": endpoint_indices.astype(np.int32),
        "anchor_indices": anchor_indices.astype(np.int32),
        "anchor_positions": shifted_q[anchor_indices].astype(np.float32),
        "support_patch_indices": anchor_indices.astype(np.int32),
        "support_patch_center_m": support_patch_center_shifted.astype(np.float32),
        "visible_support_center_m": support_patch_center_shifted.astype(np.float32),
        "rope_center": rope_center.astype(np.float32),
        "render_edges": render_edges,
        "particle_contacts": bool(particle_contacts),
        "particle_contact_kernel": bool(particle_contact_kernel),
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "release_phase_start_s": _phase_start_time(task_phases, "release" if drop_release_task else "release_retract"),
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
    drop_release_task = str(args.task) == DROP_RELEASE_TASK
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    particle_contacts, particle_contact_kernel = _resolve_particle_contact_settings(ir_obj, args)
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
        add_ground_plane=False,
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=bool(particle_contacts),
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
    settle_drag = 0.0
    if drop_release_task and "drag_damping" in ir_obj:
        settle_drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.pre_release_settle_damping_scale)
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
    frame_dt = sim_dt * float(substeps)
    release_triggered = False
    anchor_indices = np.asarray(meta.get("anchor_indices", np.empty((0,), dtype=np.int32)), dtype=np.int32)
    settle_window_frames = max(1, int(np.ceil(float(args.settle_window_seconds) / max(frame_dt, 1.0e-12))))
    release_frame_actual = None
    release_time_actual = None
    settle_gate_pass = None
    settle_gate_frame = None
    settle_gate_metrics: dict[str, Any] | None = None
    replay = None
    if str(args.robot_motion_mode) == "replay":
        if args.replay_source is None:
            raise ValueError("--replay-source is required when --robot-motion-mode replay")
        replay = _load_replay_trajectory(Path(args.replay_source), n_frames=n_frames, frame_dt=frame_dt)

    store = RolloutStorage(args.out_dir, args.prefix, mode=str(args.history_storage))
    particle_q0 = state_in.particle_q.numpy().astype(np.float32)
    body_q0 = state_in.body_q.numpy().astype(np.float32)
    body_qd0 = state_in.body_qd.numpy().astype(np.float32)
    particle_q_all = store.allocate("particle_q_all", (n_frames, particle_q0.shape[0], 3), np.float32)
    particle_q_object = store.allocate("particle_q_object", (n_frames, n_obj, 3), np.float32)
    body_q = store.allocate("body_q", (n_frames, body_q0.shape[0], body_q0.shape[1]), np.float32)
    body_vel = store.allocate("body_vel", (n_frames, body_qd0.shape[0], 3), np.float32)
    ee_target_pos = store.allocate("ee_target_pos", (n_frames, 3), np.float32)
    particle_speed_mean_series = np.zeros((n_frames,), dtype=np.float32)
    support_patch_speed_mean_series = np.zeros((n_frames,), dtype=np.float32)
    rope_com_horizontal_speed_series = np.zeros((n_frames,), dtype=np.float32)
    rope_com_vertical_speed_series = np.zeros((n_frames,), dtype=np.float32)
    rope_spring_energy_proxy_series = np.zeros((n_frames,), dtype=np.float32)
    support_patch_center_series = np.zeros((n_frames, 3), dtype=np.float32)
    prev_q_obj = None
    prev_rope_com = None
    spring_edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
    spring_rest = np.asarray(ir_obj.get("spring_rest_length", ir_obj.get("spring_rest_lengths", np.zeros((spring_edges.shape[0],), dtype=np.float32))), dtype=np.float32).reshape(-1)
    if spring_rest.shape[0] != spring_edges.shape[0]:
        spring_rest = np.zeros((spring_edges.shape[0],), dtype=np.float32)
    preroll_settle_pass = None
    preroll_settle_time_s = None
    preroll_settle_metrics = None
    preroll_frame_count = 0

    if drop_release_task and float(args.drop_preroll_settle_seconds) > 0.0:
        preroll_max_frames = max(1, int(np.ceil(float(args.drop_preroll_settle_seconds) / max(frame_dt, 1.0e-12))))
        preroll_particle_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        preroll_support_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        preroll_com_h_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        prev_q_obj_preroll = None
        prev_rope_com_preroll = None
        for preroll_frame in range(preroll_max_frames):
            q_obj_preroll = state_in.particle_q.numpy().astype(np.float32)[:n_obj]
            rope_com_preroll = np.mean(q_obj_preroll, axis=0).astype(np.float32, copy=False)
            if prev_q_obj_preroll is not None:
                qd_preroll = (q_obj_preroll - prev_q_obj_preroll) / max(frame_dt, 1.0e-12)
                preroll_particle_speed[preroll_frame] = float(np.mean(np.linalg.norm(qd_preroll, axis=1)))
                if anchor_indices.size:
                    preroll_support_speed[preroll_frame] = float(np.mean(np.linalg.norm(qd_preroll[anchor_indices], axis=1)))
                com_vel_preroll = (rope_com_preroll - prev_rope_com_preroll) / max(frame_dt, 1.0e-12)
                preroll_com_h_speed[preroll_frame] = float(np.linalg.norm(com_vel_preroll[:2]))
            prev_q_obj_preroll = q_obj_preroll.copy()
            prev_rope_com_preroll = rope_com_preroll.copy()

            preroll_settle_metrics, preroll_ok = _settle_window_metrics_from_series(
                preroll_particle_speed,
                preroll_com_h_speed,
                preroll_support_speed,
                frame_idx=preroll_frame,
                window_frames=settle_window_frames,
                particle_speed_threshold_m_s=float(args.pre_release_particle_speed_mean_max),
                com_horizontal_speed_threshold_m_s=float(args.pre_release_com_horizontal_speed_max),
                support_patch_speed_threshold_m_s=float(args.pre_release_support_speed_mean_max),
                frame_dt=float(frame_dt),
            )
            preroll_frame_count = int(preroll_frame + 1)
            if preroll_ok:
                preroll_settle_pass = True
                preroll_settle_time_s = float(preroll_frame * frame_dt)
                break

            for _ in range(substeps):
                state_in.clear_forces()
                if contacts is not None:
                    model.collide(state_in, contacts)
                solver.step(state_in, state_out, None, contacts, sim_dt)
                state_out.joint_q.assign(prev_joint_q)
                state_out.joint_qd.zero_()
                newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
                state_in, state_out = state_out, state_in
                if settle_drag > 0.0:
                    if gravity_axis is not None:
                        wp.launch(
                            _apply_drag_correction_ignore_axis,
                            dim=n_obj,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
                                n_obj,
                                sim_dt,
                                settle_drag,
                                wp.vec3(*gravity_axis.tolist()),
                            ],
                            device=device,
                        )
                    else:
                        wp.launch(
                            newton_import_ir._apply_drag_correction,
                            dim=n_obj,
                            inputs=[state_in.particle_q, state_in.particle_qd, n_obj, sim_dt, settle_drag],
                            device=device,
                        )
        if preroll_settle_pass is None:
            preroll_settle_pass = False
            preroll_settle_time_s = float(preroll_frame_count * frame_dt)

    t0 = time.perf_counter()
    for frame in range(n_frames):
        if replay is not None:
            state_in.body_q.assign(replay["body_q"][frame])
            state_in.body_qd.assign(replay["body_qd"][frame])
        q = state_in.particle_q.numpy().astype(np.float32)
        particle_q_all[frame] = q
        particle_q_object[frame] = q[:n_obj]
        q_obj = particle_q_object[frame]
        rope_com_frame = np.mean(q_obj, axis=0).astype(np.float32, copy=False)
        support_patch_center_series[frame] = np.mean(q_obj[anchor_indices], axis=0).astype(np.float32, copy=False)
        if prev_q_obj is not None:
            qd_frame = (q_obj - prev_q_obj) / max(frame_dt, 1.0e-12)
            particle_speed_mean_series[frame] = float(np.mean(np.linalg.norm(qd_frame, axis=1)))
            if anchor_indices.size:
                support_patch_speed_mean_series[frame] = float(np.mean(np.linalg.norm(qd_frame[anchor_indices], axis=1)))
            com_vel = (rope_com_frame - prev_rope_com) / max(frame_dt, 1.0e-12)
            rope_com_horizontal_speed_series[frame] = float(np.linalg.norm(com_vel[:2]))
            rope_com_vertical_speed_series[frame] = float(com_vel[2])
        if spring_edges.size:
            edge_vec = q_obj[spring_edges[:, 0]] - q_obj[spring_edges[:, 1]]
            edge_len = np.linalg.norm(edge_vec, axis=1)
            stretch = edge_len - spring_rest
            rope_spring_energy_proxy_series[frame] = float(np.mean(stretch * stretch))
        prev_q_obj = q_obj.copy()
        prev_rope_com = rope_com_frame.copy()
        body_q[frame] = state_in.body_q.numpy().astype(np.float32)
        body_vel[frame] = state_in.body_qd.numpy().astype(np.float32)[:, :3]
        if replay is not None:
            ee_target_pos[frame] = replay["ee_target_pos"][frame]
        else:
            _, target_pos_frame, _ = _task_phase_state(float(frame) * frame_dt, meta)
            ee_target_pos[frame] = target_pos_frame

        for sub in range(substeps):
            state_in.clear_forces()
            sim_t = (float(frame) * float(substeps) + float(sub)) * sim_dt
            phase_name, _, _ = _task_phase_state(sim_t, meta)
            if drop_release_task and (not release_triggered) and phase_name in {"release", "free_fall"}:
                settle_gate_metrics, settle_ok = _settle_window_metrics_from_series(
                    particle_speed_mean_series,
                    rope_com_horizontal_speed_series,
                    support_patch_speed_mean_series,
                    frame_idx=frame,
                    window_frames=settle_window_frames,
                    particle_speed_threshold_m_s=float(args.pre_release_particle_speed_mean_max),
                    com_horizontal_speed_threshold_m_s=float(args.pre_release_com_horizontal_speed_max),
                    support_patch_speed_threshold_m_s=float(args.pre_release_support_speed_mean_max),
                    frame_dt=float(frame_dt),
                )
                if settle_ok:
                    _activate_particle_indices(model, anchor_indices, device=device)
                    if anchor_indices.size and state_in.particle_qd is not None:
                        qd_np = state_in.particle_qd.numpy().astype(np.float32, copy=True)
                        qd_np[anchor_indices] = 0.0
                        state_in.particle_qd.assign(qd_np)
                    if anchor_indices.size and state_out.particle_qd is not None:
                        qd_np = state_out.particle_qd.numpy().astype(np.float32, copy=True)
                        qd_np[anchor_indices] = 0.0
                        state_out.particle_qd.assign(qd_np)
                    release_triggered = True
                    release_frame_actual = int(frame)
                    release_time_actual = float(frame * frame_dt)
                    settle_gate_pass = True
                    settle_gate_frame = int(frame)
                else:
                    if settle_gate_pass is None:
                        settle_gate_pass = False
                        settle_gate_frame = int(frame)
            if replay is not None:
                body_q_frame = replay["body_q"][frame]
                body_qd_frame = replay["body_qd"][frame]
                state_in.body_q.assign(body_q_frame)
                state_in.body_qd.assign(body_qd_frame)
            else:
                _, target_pos, target_quat = _task_phase_state(sim_t, meta)
                pos_obj.set_target_position(0, wp.vec3(*target_pos.tolist()))
                rot_obj.set_target_rotation(0, wp.vec4(*target_quat.tolist()))
                ik_solver.step(ik_joint_q, ik_joint_q, iterations=int(args.ik_iters))

                joint_target_np = ik_joint_q.numpy().reshape(-1).astype(np.float32)
                blend = float(np.clip(float(args.ik_target_blend), 0.0, 1.0))
                if blend < 1.0:
                    joint_target_np = prev_joint_q + blend * (joint_target_np - prev_joint_q)
                if drop_release_task and (not release_triggered):
                    joint_target_np[7:9] = float(args.gripper_hold)
                else:
                    joint_target_np[7:9] = float(args.gripper_open)
                joint_target_qd = (joint_target_np - prev_joint_q) / sim_dt
                prev_joint_q = joint_target_np.copy()

                state_in.joint_q.assign(joint_target_np)
                state_in.joint_qd.assign(joint_target_qd)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            if contacts is not None:
                model.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, sim_dt)
            if replay is not None:
                state_out.body_q.assign(body_q_frame)
                state_out.body_qd.assign(body_qd_frame)
            else:
                state_out.joint_q.assign(joint_target_np)
                state_out.joint_qd.assign(joint_target_qd)
                newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
            state_in, state_out = state_out, state_in

            active_drag = 0.0
            if drop_release_task and not release_triggered:
                active_drag = settle_drag
            elif drag > 0.0:
                if str(args.task) == "lift_release" and phase_name in {"approach_under", "lift", "hold"}:
                    active_drag = 0.0
                else:
                    active_drag = drag
            if active_drag > 0.0:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis,
                        dim=n_obj,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
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
                        inputs=[state_in.particle_q, state_in.particle_qd, n_obj, sim_dt, active_drag],
                        device=device,
                    )

        if (frame + 1) % 50 == 0 or frame == n_frames - 1:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": particle_q_all,
        "particle_q_object": particle_q_object,
        "body_q": body_q,
        "body_vel": body_vel,
        "ee_target_pos": ee_target_pos,
        "particle_speed_mean_series": particle_speed_mean_series,
        "support_patch_speed_mean_series": support_patch_speed_mean_series,
        "rope_com_horizontal_speed_series": rope_com_horizontal_speed_series,
        "rope_com_vertical_speed_series": rope_com_vertical_speed_series,
        "rope_spring_energy_proxy_series": rope_spring_energy_proxy_series,
        "support_patch_center_series": support_patch_center_series,
        "release_frame_actual": release_frame_actual,
        "release_time_actual": release_time_actual,
        "settle_gate_pass": settle_gate_pass,
        "settle_gate_frame": settle_gate_frame,
        "settle_gate_metrics": settle_gate_metrics,
        "preroll_settle_pass": preroll_settle_pass,
        "preroll_settle_time_s": preroll_settle_time_s,
        "preroll_settle_metrics": preroll_settle_metrics,
        "preroll_frame_count": preroll_frame_count,
        "robot_motion_mode": str(args.robot_motion_mode),
        "replay_source": (None if replay is None else str(replay["source_dir"])),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "pre_release_settle_damping_scale": float(args.pre_release_settle_damping_scale),
        "wall_time": float(wall_time),
        **store.summary_dict(),
    }


def render_video(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> Path:
    drop_release_task = str(args.task) == DROP_RELEASE_TASK
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
        try:
            apply_viewer_shape_colors(viewer, model)
        except Exception:
            pass
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
        anchor_xforms_wp = (
            wp.array(
                [wp.transform(wp.vec3(*row.tolist()), wp.quat_identity()) for row in anchor_positions],
                dtype=wp.transform,
                device=device,
            )
            if anchor_positions.size
            else None
        )
        anchor_colors_wp = (
            wp.array([wp.vec3(0.34, 0.90, 0.52) for _ in range(anchor_positions.shape[0])], dtype=wp.vec3, device=device)
            if anchor_positions.size
            else None
        )
        left_finger_idx = int(meta["left_finger_index"])
        right_finger_idx = int(meta["right_finger_index"])
        stage_center = np.asarray(meta["stage_center"], dtype=np.float32)
        stage_scale = np.asarray(meta["stage_scale"], dtype=np.float32)
        robot_base_center = np.asarray(meta["robot_base_center"], dtype=np.float32)
        robot_base_scale = np.asarray(meta["robot_base_scale"], dtype=np.float32)
        floor_center = np.asarray(meta["floor_center"], dtype=np.float32)
        floor_scale = np.asarray(meta["floor_scale"], dtype=np.float32)
        ground_grid_z = float(meta.get("floor_z", 0.0))
        ground_grid_size = float(max(1.10, 1.0 + 0.25 * abs(float(stage_center[0]))))
        ground_grid_steps = 10
        ground_starts_np, ground_ends_np, ground_colors_np = ground_grid(
            size=ground_grid_size,
            steps=ground_grid_steps,
            z=ground_grid_z,
            color=(0.66, 0.57, 0.40),
        )
        ground_starts_wp = wp.array(ground_starts_np, dtype=wp.vec3, device=device)
        ground_ends_wp = wp.array(ground_ends_np, dtype=wp.vec3, device=device)
        ground_colors_wp = wp.array(ground_colors_np, dtype=wp.vec3, device=device)
        release_time_s = None
        elapsed = 0.0
        for phase in meta["task_phases"]:
            phase_name = str(phase["name"])
            if phase_name in {"release", "release_retract"}:
                release_time_s = float(elapsed)
                break
            elapsed += max(float(phase["duration"]), 1.0e-8)

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
                phase_name, _, _ = _task_phase_state(sim_t, meta)
                viewer.begin_frame(sim_t)
                viewer.log_state(state)
                gripper_proxies = _gripper_contact_proxies_world(
                    sim_data["body_q"][sim_idx], left_finger_idx, right_finger_idx
                )
                gripper_center = gripper_proxies["gripper_center"].astype(np.float32, copy=False)
                if str(args.render_mode) == "debug":
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
                    if anchor_xforms_wp is not None and anchor_colors_wp is not None:
                        viewer.log_shapes(
                            "/demo/anchors",
                            newton.GeoType.SPHERE,
                            0.018,
                            anchor_xforms_wp,
                            anchor_colors_wp,
                        )
                if str(args.render_mode) == "presentation":
                    if drop_release_task:
                        viewer.log_shapes(
                            "/demo/floor",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in floor_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*floor_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.30, 0.28, 0.24)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/robot_pedestal",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in robot_base_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*robot_base_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.45, 0.39, 0.31)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/anchor_bar",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in np.asarray(meta["anchor_bar_scale"], dtype=np.float32).tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*np.asarray(meta["anchor_bar_center"], dtype=np.float32).tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.52, 0.44, 0.34)], dtype=wp.vec3, device=device),
                        )
                        if anchor_xforms_wp is not None and anchor_colors_wp is not None:
                            viewer.log_shapes(
                                "/demo/anchors",
                                newton.GeoType.SPHERE,
                                0.015,
                                anchor_xforms_wp,
                                anchor_colors_wp,
                            )
                    else:
                        viewer.log_shapes(
                            "/demo/floor",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in floor_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*floor_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.30, 0.28, 0.24)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/robot_pedestal",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in robot_base_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*robot_base_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.45, 0.39, 0.31)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/anchor_posts",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in np.asarray(meta["anchor_post_scale"], dtype=np.float32).tolist()),
                            wp.array(
                                [
                                    wp.transform(wp.vec3(*np.asarray(meta["anchor_left_center"], dtype=np.float32).tolist()), wp.quat_identity()),
                                    wp.transform(wp.vec3(*np.asarray(meta["anchor_right_center"], dtype=np.float32).tolist()), wp.quat_identity()),
                                ],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.48, 0.41, 0.32), wp.vec3(0.48, 0.41, 0.32)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/anchor_bar",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in np.asarray(meta["anchor_bar_scale"], dtype=np.float32).tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*np.asarray(meta["anchor_bar_center"], dtype=np.float32).tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.52, 0.44, 0.34)], dtype=wp.vec3, device=device),
                        )
                        viewer.log_shapes(
                            "/demo/stage_reference",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in stage_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*stage_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.31, 0.28, 0.22)], dtype=wp.vec3, device=device),
                        )
                    viewer.log_lines(
                        "/demo/ground_grid",
                        ground_starts_wp,
                        ground_ends_wp,
                        ground_colors_wp,
                        width=1.0,
                        hidden=False,
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
                    min_clearance, min_proxy_name, _ = _min_gripper_proxy_clearance(
                        q_obj,
                        particle_radius,
                        gripper_proxies,
                        float(args.ee_contact_radius),
                    )
                    contact_state = "ON" if min_clearance <= 0.0 else "OFF"
                    tracking_line = f"tracking err: {tracking_err:.3f} m | gripper speed: {speed:.3f} m/s"
                    if release_time_s is not None:
                        tracking_line += f" | t_release: {float(release_time_s):.3f}s"
                    frame = overlay_text_lines_rgb(
                        frame,
                        [
                            f"task: {args.task} | phase: {phase_name}",
                            "Native Newton Franka Panda + rope drop baseline | yellow=target, cyan=gripper center",
                            tracking_line,
                            f"contact: {contact_state} via {min_proxy_name} | approx gripper clearance: {1000.0 * min_clearance:.1f} mm",
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
    drop_release_task = str(args.task) == DROP_RELEASE_TASK
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
    left_finger_pos = body_q[:, left_finger_idx, :3].astype(np.float32, copy=False)
    right_finger_pos = body_q[:, right_finger_idx, :3].astype(np.float32, copy=False)
    tracking_error = np.linalg.norm(gripper_center - target_pos, axis=1)
    gripper_speed = np.linalg.norm(np.diff(gripper_center, axis=0), axis=1) / float(sim_data["sim_dt"] * sim_data["substeps"])
    if gripper_speed.size == 0:
        gripper_speed = np.zeros((1,), dtype=np.float32)
    gripper_path_length = float(np.sum(np.linalg.norm(np.diff(gripper_center, axis=0), axis=1)))

    rope_com = particle_q.mean(axis=1)
    rope_min_z = np.min(particle_q[:, :, 2], axis=1)
    ground_z = float(meta.get("floor_z", 0.0))
    rope_surface_clearance_to_ground = np.min(
        particle_q[:, :, 2] - particle_radius[None, :], axis=1
    ) - ground_z
    rope_com_disp = float(np.linalg.norm(rope_com[-1] - rope_com[0]))
    mid_segment_indices = np.asarray(meta["mid_segment_indices"], dtype=np.int32)
    mid_segment_z = particle_q[:, mid_segment_indices, 2].mean(axis=1)
    frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    particle_speed_mean_series = np.asarray(sim_data.get("particle_speed_mean_series"), dtype=np.float32)
    support_patch_speed_mean_series = np.asarray(sim_data.get("support_patch_speed_mean_series"), dtype=np.float32)
    rope_com_horizontal_speed_series = np.asarray(sim_data.get("rope_com_horizontal_speed_series"), dtype=np.float32)
    rope_com_vertical_speed_series = np.asarray(sim_data.get("rope_com_vertical_speed_series"), dtype=np.float32)
    rope_spring_energy_proxy_series = np.asarray(sim_data.get("rope_spring_energy_proxy_series"), dtype=np.float32)
    phase_names = [str(_task_phase_state(float(i) * frame_dt, meta)[0]) for i in range(particle_q.shape[0])]
    pre_frames = [i for i, name in enumerate(phase_names) if name == "pre_approach"]
    baseline_z = float(np.mean(mid_segment_z[pre_frames])) if pre_frames else float(mid_segment_z[0])
    rope_mid_peak_z_delta = float(np.max(mid_segment_z) - baseline_z)
    rope_mid_final_z_delta = float(mid_segment_z[-1] - baseline_z)
    rope_mid_peak_frame = int(np.argmax(mid_segment_z))

    contact_frames = []
    min_clearance = []
    min_clearance_source = []
    gripper_center_clearance = []
    left_finger_clearance = []
    right_finger_clearance = []
    finger_span_clearance = []
    for frame_idx in range(particle_q.shape[0]):
        min_d, min_source, per_proxy_min = _min_gripper_proxy_clearance(
            particle_q[frame_idx],
            particle_radius,
            {
                "gripper_center": gripper_center[frame_idx],
                "left_finger": left_finger_pos[frame_idx],
                "right_finger": right_finger_pos[frame_idx],
            },
            float(args.ee_contact_radius),
        )
        min_clearance.append(min_d)
        min_clearance_source.append(min_source)
        gripper_center_clearance.append(float(per_proxy_min["gripper_center"]))
        left_finger_clearance.append(float(per_proxy_min["left_finger"]))
        right_finger_clearance.append(float(per_proxy_min["right_finger"]))
        finger_span_clearance.append(float(per_proxy_min.get("finger_span", min_d)))
        if min_d <= 0.0:
            contact_frames.append(frame_idx)

    support_contact_frames = list(contact_frames)
    support_contact_mask = np.asarray([m <= 0.0 for m in min_clearance], dtype=bool)
    support_contact_duration_s = float(np.count_nonzero(support_contact_mask) * frame_dt)
    support_first_contact_frame = int(support_contact_frames[0]) if support_contact_frames else None
    support_last_contact_frame = int(support_contact_frames[-1]) if support_contact_frames else None
    support_contact_peak_frame = (
        int(np.argmin(np.asarray(min_clearance, dtype=np.float32))) if min_clearance else None
    )

    ground_contact_mask = np.asarray(rope_surface_clearance_to_ground <= 0.0, dtype=bool)
    ground_contact_frames = np.flatnonzero(ground_contact_mask).tolist()
    contact_mask = ground_contact_mask if drop_release_task else support_contact_mask
    contact_frames = ground_contact_frames if drop_release_task else support_contact_frames
    primary_clearance = rope_surface_clearance_to_ground if drop_release_task else np.asarray(min_clearance, dtype=np.float32)

    first_contact_frame = int(contact_frames[0]) if contact_frames else None
    last_contact_frame = int(contact_frames[-1]) if contact_frames else None
    contact_peak_frame = int(np.argmin(primary_clearance)) if primary_clearance.size else None
    release_candidates = [i for i, name in enumerate(phase_names) if name in {"release", "release_retract"}]
    release_frame = None
    if release_candidates:
        release_start = int(release_candidates[0])
        if drop_release_task:
            release_frame = release_start
        else:
            for idx in range(release_start, len(contact_mask)):
                window = contact_mask[idx : min(len(contact_mask), idx + 3)]
                if window.size and not np.any(window):
                    release_frame = int(idx)
                    break
    actual_release_frame = sim_data.get("release_frame_actual")
    if isinstance(actual_release_frame, (int, np.integer)):
        release_frame = int(actual_release_frame)
    elif drop_release_task:
        release_frame = None

    pre_contact_mask = np.asarray([name == "pre_approach" for name in phase_names], dtype=bool)
    release_mask = np.asarray([name in {"release", "release_retract"} for name in phase_names], dtype=bool)
    contact_duration_s = float(np.count_nonzero(contact_mask) * frame_dt)
    first_contact_phase = None if first_contact_frame is None else str(phase_names[first_contact_frame])

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        if values.size == 0 or not np.any(mask):
            return None
        return float(np.mean(values[mask]))

    def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
        if values.size == 0 or not np.any(mask):
            return None
        return float(np.max(values[mask]))

    pre_tracking_mean = _masked_mean(tracking_error, pre_contact_mask)
    contact_tracking_mean = _masked_mean(tracking_error, contact_mask)
    release_tracking_mean = _masked_mean(tracking_error, release_mask)
    pre_speed_mean = _masked_mean(gripper_speed, pre_contact_mask[: gripper_speed.shape[0]])
    contact_speed_mean = _masked_mean(gripper_speed, contact_mask[: gripper_speed.shape[0]])
    release_speed_mean = _masked_mean(gripper_speed, release_mask[: gripper_speed.shape[0]])

    unique_phase_sequence: list[str] = []
    for name in phase_names:
        if not unique_phase_sequence or unique_phase_sequence[-1] != name:
            unique_phase_sequence.append(name)

    support_source_counts: dict[str, int] = {}
    for name, active in zip(min_clearance_source, support_contact_mask.tolist(), strict=False):
        if not active:
            continue
        support_source_counts[name] = int(support_source_counts.get(name, 0) + 1)

    min_clearance_np = np.asarray(min_clearance, dtype=np.float32)
    gripper_center_clearance_np = np.asarray(gripper_center_clearance, dtype=np.float32)
    left_finger_clearance_np = np.asarray(left_finger_clearance, dtype=np.float32)
    right_finger_clearance_np = np.asarray(right_finger_clearance, dtype=np.float32)
    finger_span_clearance_np = np.asarray(finger_span_clearance, dtype=np.float32)
    contact_peak_source = (
        None if contact_peak_frame is None else str(min_clearance_source[int(contact_peak_frame)])
    )
    if drop_release_task:
        contact_peak_source = "ground_plane"

    if first_contact_frame is not None and first_contact_frame > 0 and release_frame is not None:
        fit_start = int(release_frame)
        fit_stop = int(first_contact_frame)
        fit_times = np.arange(fit_start, fit_stop, dtype=np.float64) * frame_dt
        fit_values = rope_com[fit_start:fit_stop, 2]
    else:
        fit_times = np.asarray([], dtype=np.float64)
        fit_values = np.asarray([], dtype=np.float64)
    early_fall_accel_estimate = _fit_quadratic_acceleration(fit_times, fit_values)
    impact_speed_estimate = None
    if first_contact_frame is not None and first_contact_frame > 0:
        impact_speed_estimate = float(
            max(0.0, (rope_com[first_contact_frame - 1, 2] - rope_com[first_contact_frame, 2]) / max(frame_dt, 1.0e-12))
        )

    settle_gate_metrics = sim_data.get("settle_gate_metrics") or {}
    settle_gate_pass = sim_data.get("settle_gate_pass")
    settle_gate_frame = sim_data.get("settle_gate_frame")
    release_time_s = (None if release_frame is None else float(release_frame * frame_dt))
    post_release_kick_window_frames = max(1, int(np.ceil(float(args.post_release_kick_window_seconds) / max(frame_dt, 1.0e-12))))
    post_release_horizontal_kick = None
    early_fall_horizontal_velocity = None
    pre_release_spring_energy_proxy = None
    post_release_spring_energy_proxy = None
    if release_frame is not None and rope_com_horizontal_speed_series.size:
        kick_slice = rope_com_horizontal_speed_series[
            release_frame : min(len(rope_com_horizontal_speed_series), release_frame + post_release_kick_window_frames)
        ]
        if kick_slice.size:
            post_release_horizontal_kick = float(np.mean(kick_slice))
            early_fall_horizontal_velocity = float(kick_slice[0])
    if release_frame is not None and rope_spring_energy_proxy_series.size:
        pre_slice = rope_spring_energy_proxy_series[max(0, release_frame - post_release_kick_window_frames) : release_frame]
        post_slice = rope_spring_energy_proxy_series[
            release_frame : min(len(rope_spring_energy_proxy_series), release_frame + post_release_kick_window_frames)
        ]
        if pre_slice.size:
            pre_release_spring_energy_proxy = float(np.mean(pre_slice))
        if post_slice.size:
            post_release_spring_energy_proxy = float(np.mean(post_slice))

    return {
        "ir_path": str(args.ir.resolve()),
        "output_mp4": str(out_mp4),
        "frames": int(particle_q.shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(frame_dt),
        "video_slowdown": float(args.slowdown),
        "task_duration_s": float(_total_task_duration(args)),
        "wall_time_sec": float(sim_data["wall_time"]),
        "rope_total_mass": float(meta["total_object_mass"]),
        "anchor_count": int(meta["anchor_indices"].shape[0]),
        "anchor_constraint_mode": (
            "inactive_fixed_particles_until_release" if drop_release_task else "inactive_fixed_particles"
        ),
        "anchor_mass_mode": str(args.anchor_mass_mode),
        "task": str(args.task),
        "render_mode": str(args.render_mode),
        "camera_profile": str(args.camera_profile),
        "recommended_hero_camera": meta["camera_presets"]["hero"],
        "recommended_validation_camera": meta["camera_presets"]["validation"],
        "particle_contacts": bool(meta["particle_contacts"]),
        "particle_contact_kernel": bool(meta["particle_contact_kernel"]),
        "robot_motion_mode": str(sim_data.get("robot_motion_mode", meta.get("robot_motion_mode", "ik"))),
        "replay_source": sim_data.get("replay_source"),
        "history_storage_mode": str(sim_data["history_storage_mode"]),
        "history_storage_files": sim_data["history_storage_files"],
        "apply_drag": bool(args.apply_drag),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "gravity_mag_m_s2": float(args.gravity_mag),
        "pre_release_settle_damping_scale": float(sim_data.get("pre_release_settle_damping_scale", args.pre_release_settle_damping_scale)),
        "robot_geometry": "native_franka",
        "ee_body_index": int(meta["ee_body_index"]),
        "ik_target_blend": float(args.ik_target_blend),
        "gripper_center_tracking_error_mean_m": float(np.mean(tracking_error)),
        "gripper_center_tracking_error_max_m": float(np.max(tracking_error)),
        "gripper_center_tracking_error_pre_contact_mean_m": pre_tracking_mean,
        "gripper_center_tracking_error_during_contact_mean_m": contact_tracking_mean,
        "gripper_center_tracking_error_release_mean_m": release_tracking_mean,
        "gripper_center_speed_max_m_s": float(np.max(gripper_speed)),
        "gripper_center_speed_pre_contact_mean_m_s": pre_speed_mean,
        "gripper_center_speed_during_contact_mean_m_s": contact_speed_mean,
        "gripper_center_speed_release_mean_m_s": release_speed_mean,
        "gripper_center_path_length_m": gripper_path_length,
        "ground_z_m": float(ground_z),
        "rope_min_z_m": float(np.min(rope_min_z)),
        "rope_surface_clearance_min_m": float(np.min(rope_surface_clearance_to_ground)),
        "support_contact_active_frames": int(len(support_contact_frames)),
        "support_contact_duration_s": float(support_contact_duration_s),
        "support_first_contact_frame": support_first_contact_frame,
        "support_last_contact_frame": support_last_contact_frame,
        "support_contact_peak_frame": support_contact_peak_frame,
        "support_contact_proxy_mode": "min(gripper_center,left_finger,right_finger,finger_span)",
        "support_contact_proxy_counts": support_source_counts,
        "support_patch_indices": np.asarray(meta.get("support_patch_indices", meta["anchor_indices"]), dtype=np.int32).astype(int).tolist(),
        "support_patch_center_m": np.asarray(meta.get("support_patch_center_m", meta["anchor_positions"].mean(axis=0)), dtype=np.float32).astype(float).tolist(),
        "visible_support_center_m": np.asarray(meta.get("visible_support_center_m", meta["anchor_positions"].mean(axis=0)), dtype=np.float32).astype(float).tolist(),
        "contact_proxy_mode": (
            "rope_surface_vs_ground_plane" if drop_release_task else "min(gripper_center,left_finger,right_finger,finger_span)"
        ),
        "contact_proxy_radius_m": (
            0.0 if drop_release_task else float(_gripper_contact_proxy_radii(float(args.ee_contact_radius))["gripper_center"])
        ),
        "contact_proxy_radii_m": (
            {"ground_plane": 0.0}
            if drop_release_task
            else _gripper_contact_proxy_radii(float(args.ee_contact_radius))
        ),
        "rope_com_displacement_m": rope_com_disp,
        "rope_com_z_min_m": float(np.min(rope_com[:, 2])),
        "rope_mid_segment_peak_z_delta_m": rope_mid_peak_z_delta,
        "rope_mid_segment_final_z_delta_m": rope_mid_final_z_delta,
        "rope_mid_segment_peak_frame": rope_mid_peak_frame,
        "contact_started": bool(first_contact_frame is not None),
        "first_contact_frame": first_contact_frame,
        "first_contact_time_s": (None if first_contact_frame is None else float(first_contact_frame * frame_dt)),
        "first_ground_contact_frame": first_contact_frame,
        "first_ground_contact_time_s": (None if first_contact_frame is None else float(first_contact_frame * frame_dt)),
        "last_contact_frame": last_contact_frame,
        "last_contact_time_s": (None if last_contact_frame is None else float(last_contact_frame * frame_dt)),
        "first_contact_phase": first_contact_phase,
        "contact_peak_frame": contact_peak_frame,
        "release_frame": release_frame,
        "release_time_s": release_time_s,
        "t_release_s": release_time_s,
        "impact_speed_before_first_contact_m_s": impact_speed_estimate,
        "early_fall_acceleration_estimate_m_s2": early_fall_accel_estimate,
        "early_fall_com_horizontal_velocity_m_s": early_fall_horizontal_velocity,
        "post_release_horizontal_kick_window_s": float(args.post_release_kick_window_seconds),
        "post_release_horizontal_kick_m_s": post_release_horizontal_kick,
        "contact_duration_s": contact_duration_s,
        "contact_active_frames": int(len(contact_frames)),
        "contact_phase_count": int(np.count_nonzero(contact_mask)),
        "task_phase_sequence": unique_phase_sequence,
        "rope_com_horizontal_speed_mean_pre_release_m_s": (
            None if release_frame is None or rope_com_horizontal_speed_series.size == 0 else float(np.mean(rope_com_horizontal_speed_series[:release_frame]))
        ),
        "rope_particle_speed_mean_pre_release_m_s": (
            None if release_frame is None or particle_speed_mean_series.size == 0 else float(np.mean(particle_speed_mean_series[:release_frame]))
        ),
        "support_patch_speed_mean_pre_release_m_s": (
            None if release_frame is None or support_patch_speed_mean_series.size == 0 else float(np.mean(support_patch_speed_mean_series[:release_frame]))
        ),
        "pre_release_spring_energy_proxy": pre_release_spring_energy_proxy,
        "post_release_spring_energy_proxy": post_release_spring_energy_proxy,
        "settle_window_seconds": float(args.settle_window_seconds),
        "settle_gate_pass": settle_gate_pass,
        "settle_gate_frame": settle_gate_frame,
        "settle_gate_metrics": settle_gate_metrics,
        "preroll_settle_pass": sim_data.get("preroll_settle_pass"),
        "preroll_settle_time_s": sim_data.get("preroll_settle_time_s"),
        "preroll_settle_metrics": sim_data.get("preroll_settle_metrics"),
        "preroll_frame_count": sim_data.get("preroll_frame_count"),
        "min_clearance_min_m": float(np.min(primary_clearance)),
        "min_clearance_final_m": float(primary_clearance[-1]),
        "gripper_center_clearance_min_m": float(np.min(gripper_center_clearance_np)),
        "left_finger_clearance_min_m": float(np.min(left_finger_clearance_np)),
        "right_finger_clearance_min_m": float(np.min(right_finger_clearance_np)),
        "finger_span_clearance_min_m": float(np.min(finger_span_clearance_np)),
        "contact_proxy_counts": (
            {"ground_plane": int(np.count_nonzero(contact_mask))}
            if drop_release_task
            else support_source_counts
        ),
        "contact_peak_proxy": contact_peak_source,
        "drag_phase_gating": (
            "pre_approach + release_retract"
            if str(args.task) == "lift_release"
            else "always_on"
        ),
    }


def build_physics_validation(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> dict[str, Any]:
    particle_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    particle_radius = model.particle_radius.numpy().astype(np.float32)[: particle_q.shape[1]]
    rope_com = particle_q.mean(axis=1)
    rope_min_z = np.min(particle_q[:, :, 2], axis=1)
    ground_z = float(meta.get("floor_z", 0.0))
    ground_surface_clearance = np.min(particle_q[:, :, 2] - particle_radius[None, :], axis=1) - ground_z
    ground_contact_mask = np.asarray(ground_surface_clearance <= 0.0, dtype=bool)
    ground_contact_frames = np.flatnonzero(ground_contact_mask).astype(np.int32)
    release_frame = summary.get("release_frame")
    frame_dt = float(summary.get("frame_dt") or (float(sim_data["sim_dt"]) * float(sim_data["substeps"])))

    if ground_contact_frames.size:
        first_ground_contact_frame = int(ground_contact_frames[0])
    else:
        first_ground_contact_frame = None

    if first_ground_contact_frame is not None:
        first_ground_contact_time_s = float(first_ground_contact_frame * frame_dt)
    else:
        first_ground_contact_time_s = None

    impact_speed_estimate = summary.get("impact_speed_before_first_contact_m_s")
    if impact_speed_estimate is None and first_ground_contact_frame is not None and first_ground_contact_frame > 0:
        impact_speed_estimate = float(
            max(
                0.0,
                (
                    rope_com[first_ground_contact_frame - 1, 2]
                    - rope_com[first_ground_contact_frame, 2]
                )
                / max(frame_dt, 1.0e-12),
            )
        )

    if first_ground_contact_frame is not None and release_frame is not None and first_ground_contact_frame > release_frame + 2:
        fit_start = int(release_frame)
        fit_stop = int(first_ground_contact_frame)
        fit_times = np.arange(fit_start, fit_stop, dtype=np.float64) * frame_dt
        fit_values = rope_com[fit_start:fit_stop, 2]
    else:
        fit_times = np.asarray([], dtype=np.float64)
        fit_values = np.asarray([], dtype=np.float64)

    early_fall_accel_estimate = _fit_quadratic_acceleration(fit_times, fit_values)
    penetration_mask = np.asarray(rope_min_z < ground_z - 1.0e-5, dtype=bool)

    return {
        "task": str(args.task),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "ground_z_m": float(ground_z),
        "release_frame": summary.get("release_frame"),
        "release_time_s": summary.get("release_time_s"),
        "first_ground_contact_frame": first_ground_contact_frame,
        "first_ground_contact_time_s": first_ground_contact_time_s,
        "impact_speed_before_first_contact_m_s": impact_speed_estimate,
        "rope_com_z_m": rope_com[:, 2].astype(float).tolist(),
        "rope_min_z_m": rope_min_z.astype(float).tolist(),
        "ground_surface_clearance_m": ground_surface_clearance.astype(float).tolist(),
        "any_rope_point_below_ground_threshold": bool(np.any(penetration_mask)),
        "penetration_frame_count": int(np.count_nonzero(penetration_mask)),
        "penetration_min_z_m": float(np.min(rope_min_z)),
        "early_fall_acceleration_estimate_m_s2": early_fall_accel_estimate,
        "gravity_mag_target_m_s2": float(args.gravity_mag),
        "gravity_like_fraction": (
            None
            if early_fall_accel_estimate is None
            else float(abs(abs(early_fall_accel_estimate) - float(args.gravity_mag)) / max(float(args.gravity_mag), 1.0e-12))
        ),
        "ground_contact_frames": ground_contact_frames.astype(int).tolist(),
        "summary_first_contact_frame": summary.get("first_contact_frame"),
        "summary_contact_duration_s": summary.get("contact_duration_s"),
        "summary_contact_active_frames": summary.get("contact_active_frames"),
        "support_contact_active_frames": summary.get("support_contact_active_frames"),
        "support_contact_duration_s": summary.get("support_contact_duration_s"),
        "summary_contact_proxy_mode": summary.get("contact_proxy_mode"),
        "support_contact_proxy_mode": summary.get("support_contact_proxy_mode"),
        "summary_contact_peak_proxy": summary.get("contact_peak_proxy"),
        "support_contact_proxy_counts": summary.get("support_contact_proxy_counts"),
    }


def save_physics_validation_json(args: argparse.Namespace, payload: dict[str, Any]) -> Path:
    run_root = args.out_dir.parent if args.out_dir.name == "work" else args.out_dir
    out_path = run_root / "physics_validation.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


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
    _resolve_runtime_defaults(args)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.device)

    model, ir_obj, meta, n_obj = build_model(args, device)
    _resolve_camera_defaults(args, meta)
    sim_data = simulate(model, ir_obj, meta, args, n_obj, device)
    out_mp4 = render_video(model, sim_data, meta, args, device)
    out_gif = make_gif(args, out_mp4)
    summary = build_summary(model, sim_data, meta, args, out_mp4)
    physics_validation = build_physics_validation(model, sim_data, meta, args, summary)
    physics_validation_path = save_physics_validation_json(args, physics_validation)
    summary["physics_validation_path"] = str(physics_validation_path)
    if out_gif is not None:
        summary["output_gif"] = str(out_gif)
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
