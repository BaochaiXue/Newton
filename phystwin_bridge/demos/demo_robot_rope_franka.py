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

TABLETOP_FRANKA_Q_PRE = np.asarray(
    [-0.068, 0.617, 0.308, -2.188, -0.574, 2.691, 0.287, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_FRANKA_Q_PUSH_START = np.asarray(
    [0.28778, 0.86028, 0.08928, -1.95776, 0.07106, 2.5288, 0.6783, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_FRANKA_Q_PUSH_END = np.asarray(
    [-0.105, 0.980, 0.332, -1.779, 0.577, 2.578, 1.363, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_BLOCKING_Q_PUSH_START = np.asarray(
    [0.2509, 0.7853, 0.1489, -1.9619, -0.0649, 2.5607, 0.6005, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_BLOCKING_Q_PUSH_END = np.asarray(
    [0.1299, 0.8765, 0.2067, -1.8453, 0.2722, 2.5554, 0.9594, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_BLOCKING_Q_HIGH_PRE = np.asarray(
    [0.0050, 0.2010, 0.0950, -2.3140, -0.1720, 2.4820, 0.6360, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_BLOCKING_Q_HIGH_APPROACH = np.asarray(
    [-0.0072, 0.7012, 0.2289, -2.0749, -0.3194, 2.6258, 0.4438, 0.04, 0.04],
    dtype=np.float32,
)
TABLETOP_BLOCKING_Q_UPRIGHT_PRE = FRANKA_INIT_Q.copy()
TABLETOP_BLOCKING_Q_UPRIGHT_APPROACH = np.asarray(
    [0.1236, 0.4046, 0.0763, -2.1651, -0.0325, 2.4765, 0.6925, 0.04, 0.04],
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
        "--tabletop-control-mode",
        choices=["ik", "joint_trajectory", "joint_target_drive"],
        default="joint_trajectory",
        help="Controller used only for tabletop_push_hero. `joint_trajectory` is the current promoted path, `joint_target_drive` sends the same desired path through articulation targets so rigid contact can create tracking error, and `ik` remains available for bounded tabletop contact experiments.",
    )
    p.add_argument(
        "--blocking-stage",
        choices=["rope_integrated", "rigid_only"],
        default="rope_integrated",
        help=(
            "Physical-blocking staging for tabletop_push_hero. `rigid_only` removes the rope and keeps only direct-finger vs table blocking "
            "under the same controller truth. `rope_integrated` keeps the rope in the scene."
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
    p.add_argument(
        "--load-history-from-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing previously saved rollout arrays. When set, "
            "skip simulation and render/validate from the loaded history instead."
        ),
    )
    p.add_argument(
        "--load-history-prefix",
        default=None,
        help=(
            "Prefix used when loading saved rollout arrays from --load-history-from-dir. "
            "If omitted, fall back to the current --prefix."
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
    p.add_argument(
        "--particle-radius-scale",
        type=float,
        default=0.1,
        help="Scale factor applied to the rope's physical collision/contact radius fields before building the Newton model.",
    )
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
    p.add_argument(
        "--ee-contact-radius",
        type=float,
        default=0.055,
        help=(
            "Diagnostic-only proxy radius for gripper-center / left-finger / right-finger / finger-span "
            "clearance overlays and offline reports. This does NOT change the real finger collider geometry "
            "and is NOT a valid final contact-proof surface."
        ),
    )
    p.add_argument("--ik-iters", type=int, default=24)
    p.add_argument("--joint-target-ke", type=float, default=400.0)
    p.add_argument("--joint-target-kd", type=float, default=40.0)
    p.add_argument("--finger-target-ke", type=float, default=100.0)
    p.add_argument("--finger-target-kd", type=float, default=10.0)
    p.add_argument(
        "--solver-joint-attach-ke",
        type=float,
        default=1.0e4,
        help=(
            "SemiImplicit articulation attachment stiffness. Lower values can stabilize physically actuated "
            "articulation tracking in joint_target_drive/blocking experiments."
        ),
    )
    p.add_argument(
        "--solver-joint-attach-kd",
        type=float,
        default=1.0e2,
        help=(
            "SemiImplicit articulation attachment damping paired with --solver-joint-attach-ke for "
            "joint_target_drive/blocking experiments."
        ),
    )
    p.add_argument(
        "--default-body-armature",
        type=float,
        default=0.01,
        help="Default body armature used for stable SemiImplicit joint_target_drive/blocking runs.",
    )
    p.add_argument(
        "--default-joint-armature",
        type=float,
        default=0.01,
        help="Default joint armature used for stable SemiImplicit joint_target_drive/blocking runs.",
    )
    p.add_argument(
        "--ignore-urdf-inertial-definitions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore URDF inertial definitions when using joint_target_drive so SemiImplicit derives a more stable bridge-layer Franka build.",
    )
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
        default=0.8,
        help="Initial stillness window before the tabletop push starts [s].",
    )
    p.add_argument(
        "--tabletop-approach-seconds",
        type=float,
        default=1.4,
        help="Approach-to-contact phase for the tabletop push hero [s].",
    )
    p.add_argument(
        "--tabletop-push-seconds",
        type=float,
        default=2.4,
        help="Slow lateral push phase for the tabletop push hero [s].",
    )
    p.add_argument(
        "--tabletop-hold-seconds",
        type=float,
        default=0.4,
        help="Short hold phase after the tabletop push reaches the target [s].",
    )
    p.add_argument(
        "--tabletop-retract-seconds",
        type=float,
        default=1.0,
        help="Retract-and-settle phase after the tabletop push [s].",
    )
    p.add_argument(
        "--tabletop-rope-height",
        type=float,
        default=0.070,
        help="Target rope center height used for the tabletop push hero [m].",
    )
    p.add_argument(
        "--tabletop-table-top-z",
        type=float,
        default=0.000,
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
        default=(-0.56, -0.34, 0.10),
        metavar=("X", "Y", "Z"),
        help="Robot base offset relative to the rope/table center for the tabletop hero [m].",
    )
    p.add_argument(
        "--tabletop-push-start-offset",
        type=float,
        nargs=3,
        default=(-0.18, -0.02, 0.0),
        metavar=("X", "Y", "Z"),
        help="Approach start offset relative to the rope/table center for the tabletop hero [m]. XY are used directly; Z is ignored when explicit tabletop clearances are enabled.",
    )
    p.add_argument(
        "--tabletop-push-contact-offset",
        type=float,
        nargs=3,
        default=(-0.06, -0.02, 0.0),
        metavar=("X", "Y", "Z"),
        help="Contact-ready offset relative to the rope/table center for the tabletop hero [m]. XY are used directly; Z is ignored when explicit tabletop clearances are enabled.",
    )
    p.add_argument(
        "--tabletop-push-end-offset",
        type=float,
        nargs=3,
        default=(0.14, -0.02, 0.0),
        metavar=("X", "Y", "Z"),
        help="Push end offset relative to the rope/table center for the tabletop hero [m]. XY are used directly; Z is ignored when explicit tabletop clearances are enabled.",
    )
    p.add_argument(
        "--tabletop-retract-offset",
        type=float,
        nargs=3,
        default=(0.18, -0.16, 0.0),
        metavar=("X", "Y", "Z"),
        help="Retract offset relative to the rope/table center for the tabletop hero [m]. XY are used directly; Z is ignored when explicit tabletop clearances are enabled.",
    )
    p.add_argument(
        "--tabletop-approach-clearance-z",
        type=float,
        default=0.08,
        help="World-space clearance above the tabletop rope-top reference used for the settle/approach phases [m].",
    )
    p.add_argument(
        "--tabletop-contact-clearance-z",
        type=float,
        default=0.012,
        help="World-space clearance above the tabletop rope-top reference used for contact-ready [m].",
    )
    p.add_argument(
        "--tabletop-push-clearance-z",
        type=float,
        default=0.008,
        help="World-space clearance above the tabletop rope-top reference used during the lateral push [m].",
    )
    p.add_argument(
        "--tabletop-retract-clearance-z",
        type=float,
        default=0.10,
        help="World-space clearance above the tabletop rope-top reference used for retract [m].",
    )
    p.add_argument(
        "--tabletop-ee-offset-z",
        type=float,
        default=0.165,
        help="Local +Z offset on /fr3_link7 used to target the tabletop gripper-center contact point [m].",
    )
    p.add_argument(
        "--tabletop-initial-pose",
        choices=["ir_shifted", "tabletop_curve", "tabletop_shallow_curve"],
        default="tabletop_curve",
        help="Initial rope pose for the tabletop hero: preserve the shifted IR shape or apply a hand-shaped tabletop curve.",
    )
    p.add_argument(
        "--tabletop-preroll-settle-seconds",
        type=float,
        default=2.0,
        help="Hidden preroll time used to settle the rope on the tabletop before the recorded hero clip [s].",
    )
    p.add_argument(
        "--tabletop-preroll-damping-scale",
        type=float,
        default=6.0,
        help="Extra preroll damping scale used only while settling the tabletop hero before the recorded clip.",
    )
    p.add_argument(
        "--tabletop-reset-robot-after-preroll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After hidden tabletop preroll settle, restore the robot to the nominal initial joint pose "
            "before the visible clip begins. This keeps rope settle hidden without letting low-gain "
            "joint_target_drive preroll sag pre-load the robot into the table."
        ),
    )
    p.add_argument(
        "--visible-tool-mode",
        choices=["none", "short_rod"],
        default="none",
        help="Optional visible rigid tool attached to the robot. `short_rod` adds a small capsule tool that is both the visible mesh and the physical contactor.",
    )
    p.add_argument(
        "--visible-tool-body",
        choices=["right_finger", "left_finger", "link7"],
        default="right_finger",
        help="Robot body the visible rigid tool is attached to.",
    )
    p.add_argument(
        "--visible-tool-radius",
        type=float,
        default=0.0055,
        help="Radius of the visible rigid tool capsule [m].",
    )
    p.add_argument(
        "--visible-tool-half-height",
        type=float,
        default=0.0050,
        help="Half-height of the visible rigid tool capsule cylinder section [m].",
    )
    p.add_argument(
        "--visible-tool-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0076, 0.0535),
        metavar=("X", "Y", "Z"),
        help="Local offset of the visible rigid tool center in the attachment body frame [m].",
    )
    p.add_argument(
        "--tabletop-joint-reference-family",
        choices=["accepted", "blocking_lowprofile", "blocking_highclearance", "blocking_upright"],
        default="accepted",
        help=(
            "Joint-space reference family for tabletop joint-space phases. "
            "`accepted` preserves the readable tabletop baseline waypoints. "
            "`blocking_lowprofile` uses a shallower blocking-specific family that keeps more wrist/hand clearance. "
            "`blocking_highclearance` raises the pre/approach target substantially so joint_target_drive does not fall into settle-phase table contact immediately. "
            "`blocking_upright` uses a much more tucked FRANKA_INIT_Q-like pre-pose for gravity stability."
        ),
    )
    p.add_argument(
        "--visible-tool-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Local axis along which the visible rigid tool capsule extends.",
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
    p.add_argument(
        "--particle-radius-vis-scale",
        type=float,
        default=None,
        help="Optional render-only particle radius scale. If omitted, render the true physical particle radius.",
    )
    p.add_argument(
        "--particle-radius-vis-min",
        type=float,
        default=None,
        help="Optional render-only particle radius cap. If omitted, no radius cap is applied.",
    )
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--tabletop-support-box-mode",
        choices=["none", "render_only", "physical"],
        default="render_only",
        help=(
            "Support/backstop box mode for tabletop scenes. "
            "`render_only` preserves the old visual-only pedestal story, while "
            "`physical` adds the same box as a native Newton static collider."
        ),
    )
    p.add_argument(
        "--tabletop-support-box-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-space offset applied to the tabletop support/backstop box center [m].",
    )
    p.add_argument(
        "--tabletop-support-box-scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("HX", "HY", "HZ"),
        help="Optional half-extents for the tabletop support/backstop box. If omitted, use the blocking-task backstop slab defaults on joint_target_drive runs and the historical pedestal dimensions otherwise.",
    )
    p.add_argument(
        "--tabletop-hero-hide-pedestal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide the support/backstop box in the tabletop hero presentation camera. Debug and validation remain unchanged.",
    )
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


def _quat_inverse_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    vec_quat = np.asarray([v[0], v[1], v[2], 0.0], dtype=np.float32)
    rotated = _quat_multiply(_quat_multiply(_quat_conjugate(q), vec_quat), q)
    return rotated[:3]


def _quat_from_axis_angle_np(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    s = float(np.sin(half))
    c = float(np.cos(half))
    return np.asarray([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def _visible_tool_local_quat(axis_name: str) -> np.ndarray:
    axis_name = str(axis_name).lower()
    if axis_name == "z":
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if axis_name == "x":
        return _quat_from_axis_angle_np(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), 0.5 * np.pi)
    if axis_name == "y":
        return _quat_from_axis_angle_np(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), -0.5 * np.pi)
    raise ValueError(f"Unsupported visible-tool axis: {axis_name}")


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


def _fk_gripper_center_from_joint_q(
    model: newton.Model,
    joint_q_row: np.ndarray,
    *,
    left_finger_idx: int,
    right_finger_idx: int,
) -> np.ndarray:
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q_row, dtype=np.float32))
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy().astype(np.float32)
    return _gripper_center_world_position(body_q, left_finger_idx, right_finger_idx)


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


def _combine_world_transform(body_q_row: np.ndarray, local_tf_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_q_row = np.asarray(body_q_row, dtype=np.float32)
    local_tf_row = np.asarray(local_tf_row, dtype=np.float32)
    body_pos = body_q_row[:3]
    body_quat = body_q_row[3:7]
    local_pos = local_tf_row[:3]
    local_quat = local_tf_row[3:7]
    world_pos = body_pos + _quat_rotate_vector(body_quat, local_pos)
    world_quat = _quat_multiply(body_quat, local_quat)
    return world_pos.astype(np.float32, copy=False), world_quat.astype(np.float32, copy=False)


def _signed_distance_points_to_box(points: np.ndarray, center: np.ndarray, quat: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    half_extents = np.asarray(half_extents, dtype=np.float32)
    rel = points - center[None, :]
    local = np.stack([_quat_inverse_rotate_vector(quat, row) for row in rel], axis=0)
    q = np.abs(local) - half_extents[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return (outside + inside).astype(np.float32, copy=False)


def _finger_box_entries(model: newton.Model, meta: dict[str, Any]) -> list[dict[str, Any]]:
    shape_body = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
    shape_type = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else np.asarray(model.shape_type)
    shape_scale = model.shape_scale.numpy() if hasattr(model.shape_scale, "numpy") else np.asarray(model.shape_scale)
    shape_transform = model.shape_transform.numpy() if hasattr(model.shape_transform, "numpy") else np.asarray(model.shape_transform)
    left_idx = int(meta["left_finger_index"])
    right_idx = int(meta["right_finger_index"])
    entries: list[dict[str, Any]] = []
    for shape_idx in range(int(model.shape_count)):
        body_idx = int(shape_body[shape_idx])
        if body_idx not in {left_idx, right_idx}:
            continue
        if int(shape_type[shape_idx]) != int(newton.GeoType.BOX):
            continue
        side = "left" if body_idx == left_idx else "right"
        entries.append(
            {
                "shape_index": int(shape_idx),
                "body_index": int(body_idx),
                "side": side,
                "name": f"{side}_box_{len([e for e in entries if e['side'] == side])}",
                "half_extents": np.asarray(shape_scale[shape_idx], dtype=np.float32),
                "local_transform": np.asarray(shape_transform[shape_idx], dtype=np.float32),
            }
        )
    side_groups = {"left": [], "right": []}
    for entry in entries:
        side_groups[entry["side"]].append(entry)
    for side, group in side_groups.items():
        if not group:
            continue
        tip_entry = max(group, key=lambda item: float(item["local_transform"][2]))
        tip_entry["name"] = f"{side}_tip_box"
        tip_entry["is_tip"] = True
    return entries


def _min_finger_box_clearance(
    particle_q: np.ndarray,
    particle_radius: np.ndarray,
    body_q_row: np.ndarray,
    finger_box_entries: list[dict[str, Any]],
) -> tuple[float, str, dict[str, float]]:
    best_name = "none"
    best_clearance = float("inf")
    per_name: dict[str, float] = {}
    left_best = float("inf")
    right_best = float("inf")
    for entry in finger_box_entries:
        center_w, quat_w = _combine_world_transform(
            body_q_row[int(entry["body_index"])],
            np.asarray(entry["local_transform"], dtype=np.float32),
        )
        sdf = _signed_distance_points_to_box(
            particle_q,
            center_w,
            quat_w,
            np.asarray(entry["half_extents"], dtype=np.float32),
        )
        clearance = float(np.min(sdf - particle_radius))
        per_name[str(entry["name"])] = clearance
        if str(entry["side"]) == "left":
            left_best = min(left_best, clearance)
        else:
            right_best = min(right_best, clearance)
        if clearance < best_clearance:
            best_clearance = clearance
            best_name = str(entry["name"])
    if left_best < float("inf"):
        per_name["left_any_box"] = float(left_best)
    if right_best < float("inf"):
        per_name["right_any_box"] = float(right_best)
    per_name["any_finger_box"] = float(best_clearance)
    return float(best_clearance), best_name, per_name


def _capsule_segment_endpoints(center: np.ndarray, quat: np.ndarray, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    axis_offset = _quat_rotate_vector(quat, np.asarray([0.0, 0.0, float(half_height)], dtype=np.float32))
    return (
        (center - axis_offset).astype(np.float32, copy=False),
        (center + axis_offset).astype(np.float32, copy=False),
    )


def _visible_tool_entry(model: newton.Model, meta: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(meta.get("visible_tool_enabled", False)):
        return None
    shape_idx = meta.get("visible_tool_shape_index")
    if shape_idx is None:
        return None
    shape_idx = int(shape_idx)
    shape_scale = model.shape_scale.numpy() if hasattr(model.shape_scale, "numpy") else np.asarray(model.shape_scale)
    shape_transform = model.shape_transform.numpy() if hasattr(model.shape_transform, "numpy") else np.asarray(model.shape_transform)
    shape_body = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
    shape_label = getattr(model, "shape_label", None)
    shape_label_value = None
    if isinstance(shape_label, list) and 0 <= shape_idx < len(shape_label):
        shape_label_value = str(shape_label[shape_idx])
    return {
        "shape_index": shape_idx,
        "shape_label": shape_label_value,
        "name": str(meta.get("visible_tool_label", "visible_tool_capsule")),
        "body_index": int(shape_body[shape_idx]),
        "body_label": str(meta.get("visible_tool_body_label")),
        "radius": float(shape_scale[shape_idx][0]),
        "half_height": float(shape_scale[shape_idx][1]),
        "local_transform": np.asarray(shape_transform[shape_idx], dtype=np.float32),
        "axis": str(meta.get("visible_tool_axis", "z")),
    }


def _tool_world_transform(body_q_frame: np.ndarray, tool_entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return _combine_world_transform(
        body_q_frame[int(tool_entry["body_index"])],
        np.asarray(tool_entry["local_transform"], dtype=np.float32),
    )


def _min_capsule_clearance(
    particle_q: np.ndarray,
    particle_radius: np.ndarray,
    body_q_frame: np.ndarray,
    tool_entry: dict[str, Any] | None,
) -> tuple[float | None, str | None]:
    if tool_entry is None:
        return None, None
    center_w, quat_w = _tool_world_transform(body_q_frame, tool_entry)
    seg_a, seg_b = _capsule_segment_endpoints(
        center_w,
        quat_w,
        float(tool_entry["half_height"]),
    )
    segment = seg_b - seg_a
    denom = float(np.dot(segment, segment))
    if denom <= 1.0e-12:
        d = np.linalg.norm(particle_q - seg_a[None, :], axis=1) - (
            particle_radius + float(tool_entry["radius"])
        )
    else:
        t = np.clip(((particle_q - seg_a[None, :]) @ segment) / denom, 0.0, 1.0)
        proj = seg_a[None, :] + t[:, None] * segment[None, :]
        d = np.linalg.norm(particle_q - proj, axis=1) - (
            particle_radius + float(tool_entry["radius"])
        )
    return float(np.min(d)), str(tool_entry.get("shape_label") or tool_entry.get("name") or "visible_tool_capsule")


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
        if bool(meta.get("visible_tool_enabled", False)):
            hero_target = np.asarray(
                [
                    float(push_focus[0] - 0.004),
                    float(push_focus[1] - 0.006),
                    float(max(table_center[2] + 0.075, rope_center[2] + 0.020)),
                ],
                dtype=np.float32,
            )
            hero_pos = camera_position(hero_target, yaw_deg=162.0, pitch_deg=-15.0, distance=1.34)

            validation_target = np.asarray(
                [
                    float(push_focus[0] - 0.002),
                    float(push_focus[1] - 0.004),
                    float(max(table_center[2] + 0.082, rope_center[2] + 0.025)),
                ],
                dtype=np.float32,
            )
            validation_pos = camera_position(validation_target, yaw_deg=166.0, pitch_deg=-17.0, distance=1.54)
            hero_pitch = -15.0
            hero_yaw = 162.0
            hero_fov = 34.0
            validation_pitch = -17.0
            validation_yaw = 166.0
            validation_fov = 36.0
        else:
            hero_target = np.asarray(
                [
                    float(push_focus[0] + 0.004),
                    float(push_focus[1] + 0.012),
                    float(max(table_center[2] + 0.11, rope_center[2] + 0.045)),
                ],
                dtype=np.float32,
            )
            hero_pos = camera_position(hero_target, yaw_deg=148.0, pitch_deg=-22.0, distance=1.42)

            validation_target = np.asarray(
                [
                    float(push_focus[0]),
                    float(push_focus[1]),
                    float(max(table_center[2] + 0.11, rope_center[2] + 0.04)),
                ],
                dtype=np.float32,
            )
            validation_pos = camera_position(validation_target, yaw_deg=156.0, pitch_deg=-23.0, distance=1.68)
            hero_pitch = -22.0
            hero_yaw = 148.0
            hero_fov = 36.0
            validation_pitch = -23.0
            validation_yaw = 156.0
            validation_fov = 38.0

        return {
            "hero": {
                "pos": hero_pos.astype(np.float32, copy=False).tolist(),
                "pitch": hero_pitch,
                "yaw": hero_yaw,
                "fov": hero_fov,
                "target": hero_target.astype(np.float32, copy=False).tolist(),
            },
            "validation": {
                "pos": validation_pos.astype(np.float32, copy=False).tolist(),
                "pitch": validation_pitch,
                "yaw": validation_yaw,
                "fov": validation_fov,
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


def _reshape_rope_for_tabletop(
    points: np.ndarray,
    *,
    table_top_z: float,
    particle_radius: float,
    pose_mode: str = "tabletop_curve",
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    n = points.shape[0]
    if n <= 1:
        return points
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    surface_z = float(table_top_z) + float(particle_radius) + 0.004
    if str(pose_mode) == "tabletop_shallow_curve":
        x_span = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        lateral_amp = float(np.clip(max(3.0 * float(particle_radius), 0.04 * x_span), 0.006, 0.010))
        arch_amp = float(np.clip(max(1.5 * float(particle_radius), 0.015 * x_span), 0.002, 0.005))
        # Use a low-amplitude S curve with zero offset at both ends.
        s_curve = lateral_amp * np.sin(2.0 * np.pi * u) * np.sin(np.pi * u)
        # Keep endpoints near the tabletop while giving the middle a mild crown.
        arch = arch_amp * np.sin(np.pi * u) ** 2
        points[:, 1] = points[:, 1] + s_curve.astype(np.float32, copy=False)
        points[:, 2] = (surface_z + arch).astype(np.float32, copy=False)
        # Light smoothing suppresses high-frequency kinks when the rope radius is small.
        if n >= 5:
            kernel = np.asarray([0.2, 0.6, 0.2], dtype=np.float32)
            y = np.convolve(points[:, 1], kernel, mode="same")
            z = np.convolve(points[:, 2], kernel, mode="same")
            points[:, 1] = y.astype(np.float32, copy=False)
            points[:, 2] = z.astype(np.float32, copy=False)
            points[0, 2] = surface_z
            points[-1, 2] = surface_z
    else:
        arch = 0.016 * np.square(2.0 * u - 1.0)
        s_curve = 0.028 * np.sin(2.0 * np.pi * u)
        points[:, 1] = points[:, 1] + s_curve.astype(np.float32, copy=False)
        points[:, 2] = (surface_z + arch).astype(np.float32, copy=False)
    return points


def _task_phase_definitions(
    rope_center: np.ndarray,
    args: argparse.Namespace,
    *,
    tabletop_target_z: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
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
        settle = rope_center + np.asarray(args.tabletop_push_start_offset, dtype=np.float32)
        contact_ready = rope_center + np.asarray(args.tabletop_push_contact_offset, dtype=np.float32)
        push_end = rope_center + np.asarray(args.tabletop_push_end_offset, dtype=np.float32)
        retract = rope_center + np.asarray(args.tabletop_retract_offset, dtype=np.float32)
        if tabletop_target_z is not None:
            settle[2] = float(tabletop_target_z["settle"])
            contact_ready[2] = float(tabletop_target_z["contact_ready"])
            push_end[2] = float(tabletop_target_z["push"])
            retract[2] = float(tabletop_target_z["retract"])
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
                "end": contact_ready,
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


def _tabletop_joint_phase_waypoints(reference_family: str = "accepted") -> list[dict[str, Any]]:
    if str(reference_family) == "blocking_lowprofile":
        q_pre = TABLETOP_FRANKA_Q_PRE.copy()
        q_push_start = TABLETOP_BLOCKING_Q_PUSH_START.copy()
        q_push_end = TABLETOP_BLOCKING_Q_PUSH_END.copy()
    elif str(reference_family) == "blocking_highclearance":
        q_pre = TABLETOP_BLOCKING_Q_HIGH_PRE.copy()
        q_push_start = TABLETOP_BLOCKING_Q_HIGH_APPROACH.copy()
        q_push_end = TABLETOP_BLOCKING_Q_PUSH_END.copy()
    elif str(reference_family) == "blocking_upright":
        q_pre = TABLETOP_BLOCKING_Q_UPRIGHT_PRE.copy()
        q_push_start = TABLETOP_BLOCKING_Q_UPRIGHT_APPROACH.copy()
        q_push_end = TABLETOP_BLOCKING_Q_PUSH_END.copy()
    else:
        q_pre = TABLETOP_FRANKA_Q_PRE.copy()
        q_push_start = TABLETOP_FRANKA_Q_PUSH_START.copy()
        q_push_end = TABLETOP_FRANKA_Q_PUSH_END.copy()
    return [
        {"name": "settle", "start_q": q_pre.copy(), "end_q": q_pre.copy()},
        {"name": "approach", "start_q": q_pre.copy(), "end_q": q_push_start.copy()},
        {"name": "push", "start_q": q_push_start.copy(), "end_q": q_push_end.copy()},
        {"name": "hold", "start_q": q_push_end.copy(), "end_q": q_push_end.copy()},
        {"name": "retract", "start_q": q_push_end.copy(), "end_q": q_pre.copy()},
    ]


def _joint_phase_state(t: float, meta: dict[str, Any]) -> tuple[str, np.ndarray]:
    phases = meta["joint_task_phases"]
    elapsed = 0.0
    for phase in phases:
        duration = max(float(phase["duration"]), 1.0e-8)
        end_t = elapsed + duration
        if t <= end_t:
            alpha = 0.0 if np.allclose(phase["start_q"], phase["end_q"]) else np.clip((t - elapsed) / duration, 0.0, 1.0)
            q = (1.0 - alpha) * np.asarray(phase["start_q"], dtype=np.float32) + alpha * np.asarray(phase["end_q"], dtype=np.float32)
            return str(phase["name"]), q.astype(np.float32, copy=False)
        elapsed = end_t
    last = phases[-1]
    return str(last["name"]), np.asarray(last["end_q"], dtype=np.float32)


def _support_axis_name(axis_idx: int) -> str:
    return ("x", "y", "z")[int(axis_idx)]


def _infer_tabletop_support_axis(robot_base_center: np.ndarray, stage_center: np.ndarray) -> tuple[int, float]:
    delta = np.asarray(stage_center, dtype=np.float32) - np.asarray(robot_base_center, dtype=np.float32)
    delta = delta.astype(np.float32, copy=True)
    delta[2] = 0.0
    if float(np.max(np.abs(delta[:2]))) <= 1.0e-6:
        return 0, 1.0
    axis_idx = int(np.argmax(np.abs(delta[:2])))
    sign = 1.0 if float(delta[axis_idx]) >= 0.0 else -1.0
    return axis_idx, sign


def _default_tabletop_support_box_geometry(
    robot_base_center: np.ndarray,
    robot_base_scale: np.ndarray,
    stage_center: np.ndarray,
    stage_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center = np.asarray(robot_base_center, dtype=np.float32).astype(np.float32, copy=True)
    scale = np.asarray(robot_base_scale, dtype=np.float32).astype(np.float32, copy=True)
    axis_idx, axis_sign = _infer_tabletop_support_axis(center, np.asarray(stage_center, dtype=np.float32))
    center[axis_idx] -= float(axis_sign) * 0.16
    scale[axis_idx] = max(0.04, float(scale[axis_idx]) - 0.08)
    return center, scale, {
        "source": "blocking_backstop_default",
        "normal_axis_index": int(axis_idx),
        "normal_axis": _support_axis_name(axis_idx),
        "normal_sign": float(axis_sign),
    }


def _find_index_by_suffix(labels: list[str], suffix: str) -> int:
    for idx, label in enumerate(labels):
        if label.endswith(suffix):
            return idx
    raise KeyError(f"Could not find label suffix {suffix!r} in {labels}")


def build_model(args: argparse.Namespace, device: str) -> tuple[newton.Model, dict[str, Any], dict[str, Any], int]:
    raw_ir = load_ir(args.ir)
    drop_release_task = str(args.task) == DROP_RELEASE_TASK
    tabletop_task = str(args.task) == TABLETOP_PUSH_TASK
    rigid_only_stage = tabletop_task and str(args.blocking_stage) == "rigid_only"
    tabletop_joint_path_mode = tabletop_task and str(args.tabletop_control_mode) in {
        "joint_trajectory",
        "joint_target_drive",
    }
    _validate_scaling_args(args)
    _maybe_autoset_mass_spring_scale(args, raw_ir)
    ir_obj = _copy_object_only_ir(raw_ir, args)

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

    endpoint_indices = _rope_endpoints(edges, ir_n_obj, x0)
    endpoint_mid = 0.5 * (x0[int(endpoint_indices[0])] + x0[int(endpoint_indices[1])])
    support_patch_indices = np.empty((0,), dtype=np.int32)
    support_patch_center = endpoint_mid.astype(np.float32, copy=False)
    rope_height_target = float(args.tabletop_rope_height) if tabletop_task else float(args.anchor_height)
    shift = np.array(
        [
            -float(endpoint_mid[0]),
            -float(endpoint_mid[1]),
            rope_height_target - float(endpoint_mid[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    if tabletop_task and str(args.tabletop_initial_pose) in {"tabletop_curve", "tabletop_shallow_curve"}:
        shifted_q[:ir_n_obj] = _reshape_rope_for_tabletop(
            shifted_q[:ir_n_obj],
            table_top_z=float(args.tabletop_table_top_z),
            particle_radius=particle_radius_ref,
            pose_mode=str(args.tabletop_initial_pose),
        )
    rope_center = shifted_q[:ir_n_obj].mean(axis=0).astype(np.float32, copy=False)
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
    tabletop_joint_drive_requested = tabletop_task and str(args.tabletop_control_mode) == "joint_target_drive"
    if tabletop_joint_drive_requested:
        # SemiImplicit native articulation tracking is materially more stable on
        # the Franka import when we use a small geometry-based armature and let
        # Newton derive inertial properties from the imported geometry.
        builder.default_body_armature = float(args.default_body_armature)
        builder.default_joint_cfg.armature = float(args.default_joint_armature)
    if not rigid_only_stage:
        _, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=bool(particle_contacts))
        newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
        builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]
    if not tabletop_task and not rigid_only_stage:
        for idx in anchor_indices.tolist():
            builder.particle_flags[idx] = int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)
            if (not drop_release_task) and str(args.anchor_mass_mode) == "zero":
                builder.particle_mass[idx] = 0.0

    support_patch_center_shifted = np.mean(anchor_positions, axis=0).astype(np.float32, copy=False) if anchor_positions.size else rope_center
    if tabletop_task:
        robot_reference_point = rope_center
        robot_base_pos = robot_reference_point + np.asarray(args.tabletop_robot_base_offset, dtype=np.float32)
    else:
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
        ignore_inertial_definitions=bool(tabletop_joint_drive_requested and args.ignore_urdf_inertial_definitions),
    )

    if tabletop_task:
        tabletop_joint_waypoints = _tabletop_joint_phase_waypoints(str(args.tabletop_joint_reference_family))
        robot_joint_init = np.asarray(tabletop_joint_waypoints[0]["start_q"], dtype=np.float32).copy()
    else:
        robot_joint_init = FRANKA_INIT_Q.copy()
    builder.joint_q[:9] = robot_joint_init.tolist()
    builder.joint_target_pos[:9] = robot_joint_init.tolist()
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
    visible_tool_enabled = str(args.visible_tool_mode) != "none"
    visible_tool_body_index = None
    visible_tool_body_label = None
    visible_tool_local_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    visible_tool_local_transform = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if visible_tool_enabled:
        visible_tool_body_map = {
            "right_finger": int(right_finger_index),
            "left_finger": int(left_finger_index),
            "link7": int(ee_body_index),
        }
        visible_tool_body_index = int(visible_tool_body_map[str(args.visible_tool_body)])
        visible_tool_body_label = str(builder.body_label[visible_tool_body_index])
        visible_tool_local_quat = _visible_tool_local_quat(str(args.visible_tool_axis))
        visible_tool_local_transform = np.asarray(
            [
                float(args.visible_tool_offset[0]),
                float(args.visible_tool_offset[1]),
                float(args.visible_tool_offset[2]),
                float(visible_tool_local_quat[0]),
                float(visible_tool_local_quat[1]),
                float(visible_tool_local_quat[2]),
                float(visible_tool_local_quat[3]),
            ],
            dtype=np.float32,
        )
    visible_tool_color = np.asarray([0.98, 0.18, 0.10], dtype=np.float32)
    ee_offset_local = np.asarray(
        [0.0, 0.0, float(args.tabletop_ee_offset_z)] if tabletop_task else [0.0, 0.0, 0.22],
        dtype=np.float32,
    )
    ik_reference_body_index = int(ee_body_index)
    ik_reference_offset_local = ee_offset_local.astype(np.float32, copy=True)
    ik_reference_offset_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if tabletop_task and visible_tool_enabled and str(args.tabletop_control_mode) == "ik":
        ik_reference_body_index = int(visible_tool_body_index)
        ik_reference_offset_local = np.asarray(visible_tool_local_transform[:3], dtype=np.float32)
        ik_reference_offset_quat = np.asarray(visible_tool_local_transform[3:7], dtype=np.float32)
    floor_z = 0.0
    table_top_z = float(args.tabletop_table_top_z) if tabletop_task else None
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
    elif tabletop_task:
        support_point = np.asarray(
            [
                float(rope_center[0]),
                float(rope_center[1]),
                float(table_top_z),
            ],
            dtype=np.float32,
        )
        anchor_bar_center = np.asarray(
            [
                float(rope_center[0]),
                float(rope_center[1]),
                float(table_top_z - float(args.tabletop_table_hz)),
            ],
            dtype=np.float32,
        )
        anchor_bar_scale = np.asarray(
            [
                float(args.tabletop_table_hx),
                float(args.tabletop_table_hy),
                float(args.tabletop_table_hz),
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
    tabletop_target_z = None
    if tabletop_task:
        tabletop_rope_top_z = float(table_top_z) + 2.0 * float(particle_radius_ref) + 0.004
        tabletop_target_z = {
            "settle": float(tabletop_rope_top_z + float(args.tabletop_approach_clearance_z)),
            "contact_ready": float(tabletop_rope_top_z + float(args.tabletop_contact_clearance_z)),
            "push": float(tabletop_rope_top_z + float(args.tabletop_push_clearance_z)),
            "retract": float(tabletop_rope_top_z + float(args.tabletop_retract_clearance_z)),
        }
    task_phases = _task_phase_definitions(rope_center, args, tabletop_target_z=tabletop_target_z)
    if drop_release_task:
        task_phases[0]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[1]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[1]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[2]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[2]["end"] = support_point.astype(np.float32, copy=False)
        task_phases[3]["start"] = support_point.astype(np.float32, copy=False)
        task_phases[3]["end"] = support_point.astype(np.float32, copy=False)
    elif tabletop_task:
        task_phases[0]["end"] = task_phases[0]["start"].astype(np.float32, copy=False)
        task_phases[1]["start"] = task_phases[0]["end"].astype(np.float32, copy=False)
        task_phases[2]["start"] = np.asarray(task_phases[1]["end"], dtype=np.float32)
        task_phases[3]["start"] = task_phases[2]["end"].astype(np.float32, copy=False)
        task_phases[4]["start"] = task_phases[3]["end"].astype(np.float32, copy=False)
    ee_target_end = np.asarray(task_phases[-1]["end"], dtype=np.float32)
    ee_target_quat = np.asarray(task_phases[0]["quat"], dtype=np.float32)
    mid_segment_indices = _mid_segment_indices(shifted_q[:n_obj], rope_center)
    if tabletop_task:
        stage_center = np.asarray(
            [
                float(rope_center[0]),
                float(rope_center[1]),
                float(table_top_z - float(args.tabletop_table_hz)),
            ],
            dtype=np.float32,
        )
        stage_scale = np.asarray(
            [float(args.tabletop_table_hx), float(args.tabletop_table_hy), float(args.tabletop_table_hz)],
            dtype=np.float32,
        )
    else:
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
    support_box_uses_blocking_backstop_default = bool(
        tabletop_task and str(args.tabletop_control_mode) == "joint_target_drive"
    )
    if support_box_uses_blocking_backstop_default:
        support_box_default_center, support_box_default_scale, support_box_geometry = _default_tabletop_support_box_geometry(
            robot_base_center,
            robot_base_scale,
            stage_center,
            stage_scale,
        )
    else:
        support_box_default_center = robot_base_center.astype(np.float32, copy=True)
        support_box_default_scale = robot_base_scale.astype(np.float32, copy=True)
        support_axis_idx, support_axis_sign = _infer_tabletop_support_axis(robot_base_center, stage_center)
        support_box_geometry = {
            "source": "historical_pedestal_default",
            "normal_axis_index": int(support_axis_idx),
            "normal_axis": _support_axis_name(support_axis_idx),
            "normal_sign": float(support_axis_sign),
        }
    support_box_center = support_box_default_center.astype(np.float32, copy=True)
    if args.tabletop_support_box_scale is not None:
        support_box_scale = np.asarray(args.tabletop_support_box_scale, dtype=np.float32)
        support_box_geometry["source"] = "explicit_support_box_scale_override"
    else:
        support_box_scale = support_box_default_scale.astype(np.float32, copy=True)
    support_box_center = support_box_center + np.asarray(args.tabletop_support_box_offset, dtype=np.float32)
    if np.linalg.norm(np.asarray(args.tabletop_support_box_offset, dtype=np.float32)) > 0.0:
        support_box_geometry["source"] = "explicit_support_box_offset_override"
    support_box_enabled = bool(tabletop_task and str(args.tabletop_support_box_mode) != "none")
    support_box_physical = bool(tabletop_task and str(args.tabletop_support_box_mode) == "physical")
    if drop_release_task:
        anchor_left_center = np.zeros((3,), dtype=np.float32)
        anchor_right_center = np.zeros((3,), dtype=np.float32)
        floor_scale = np.asarray([1.45, 1.10, 0.012], dtype=np.float32)
        floor_center = np.asarray([0.0, 0.0, float(floor_z - floor_scale[2])], dtype=np.float32)
        anchor_post_scale = np.asarray([0.024, 0.024, 0.024], dtype=np.float32)
    elif tabletop_task:
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
            "table_top_z": table_top_z,
            "visible_tool_enabled": bool(visible_tool_enabled),
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
    if tabletop_task:
        table_cfg = builder.default_shape_cfg.copy()
        newton_import_ir._configure_ground_contact_material(
            table_cfg,
            ir_obj,
            cfg,
            checks,
            context="tabletop_table_box",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*stage_center.tolist()), wp.quat_identity()),
            hx=float(stage_scale[0]),
            hy=float(stage_scale[1]),
            hz=float(stage_scale[2]),
            cfg=table_cfg,
            label="tabletop_table_box",
        )
        if support_box_physical:
            support_cfg = builder.default_shape_cfg.copy()
            newton_import_ir._configure_ground_contact_material(
                support_cfg,
                ir_obj,
                cfg,
                checks,
                context="tabletop_support_box",
            )
            support_box_shape_index = int(
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(*support_box_center.tolist()), wp.quat_identity()),
                    hx=float(support_box_scale[0]),
                    hy=float(support_box_scale[1]),
                    hz=float(support_box_scale[2]),
                    cfg=support_cfg,
                    label="tabletop_support_box",
                )
            )
        else:
            support_box_shape_index = None
    else:
        support_box_shape_index = None
    visible_tool_shape_index = None
    if visible_tool_enabled:
        tool_cfg = builder.default_shape_cfg.copy()
        newton_import_ir._configure_ground_contact_material(
            tool_cfg,
            ir_obj,
            cfg,
            checks,
            context="visible_tool_capsule",
        )
        visible_tool_shape_index = int(
            builder.add_shape_capsule(
                body=visible_tool_body_index,
                xform=wp.transform(
                    wp.vec3(*np.asarray(args.visible_tool_offset, dtype=np.float32).tolist()),
                    wp.quat(*visible_tool_local_quat.tolist()),
                ),
                radius=float(args.visible_tool_radius),
                half_height=float(args.visible_tool_half_height),
                cfg=tool_cfg,
                label="visible_tool_capsule",
            )
        )
    model = builder.finalize(device=device)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    if not bool(particle_contact_kernel):
        model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    init_state = model.state()
    init_state.joint_q.assign(robot_joint_init.astype(np.float32))
    init_state.joint_qd.zero_()
    newton.eval_fk(model, init_state.joint_q, init_state.joint_qd, init_state)
    init_body_q = init_state.body_q.numpy().astype(np.float32)
    if tabletop_task:
        ee_target_quat = init_body_q[int(ee_body_index), 3:7].astype(np.float32, copy=False)
        for phase in task_phases:
            phase["quat"] = ee_target_quat.astype(np.float32, copy=False)
    ee_world_start = _ee_world_position(init_body_q[int(ik_reference_body_index)], ik_reference_offset_local)
    if tabletop_task:
        ee_target_start = np.asarray(task_phases[0]["start"], dtype=np.float32)
        if visible_tool_enabled and str(args.tabletop_control_mode) == "ik":
            ee_target_start = ee_world_start.astype(np.float32, copy=False)
            task_phases[0]["start"] = ee_target_start.astype(np.float32, copy=False)
            task_phases[0]["end"] = ee_target_start.astype(np.float32, copy=False)
            task_phases[1]["start"] = ee_target_start.astype(np.float32, copy=False)
        tabletop_push_focus = (
            0.5
            * (
                np.asarray(task_phases[2]["start"], dtype=np.float32)
                + np.asarray(task_phases[2]["end"], dtype=np.float32)
            )
        ).astype(np.float32, copy=False)
    else:
        ee_target_start = ee_world_start.astype(np.float32, copy=False)
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
        "ik_reference_body_index": int(ik_reference_body_index),
        "ik_reference_body_label": str(builder.body_label[int(ik_reference_body_index)]),
        "ik_reference_offset_local": ik_reference_offset_local.astype(np.float32),
        "ik_reference_offset_quat": ik_reference_offset_quat.astype(np.float32),
        "ee_target_quat": ee_target_quat.astype(np.float32),
        "ee_target_start": ee_target_start.astype(np.float32),
        "ee_target_end": ee_target_end.astype(np.float32),
        "ik_reference_body_index": (
            int(visible_tool_body_index)
            if tabletop_task and bool(visible_tool_enabled)
            else int(ee_body_index)
        ),
        "ik_reference_offset_local": (
            np.asarray(visible_tool_local_transform[:3], dtype=np.float32)
            if tabletop_task and bool(visible_tool_enabled)
            else ee_offset_local.astype(np.float32)
        ),
        "ik_reference_offset_quat": (
            np.asarray(visible_tool_local_transform[3:7], dtype=np.float32)
            if tabletop_task and bool(visible_tool_enabled)
            else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ),
        "task": str(args.task),
        "blocking_stage": str(args.blocking_stage),
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
        "support_box_enabled": bool(support_box_enabled),
        "support_box_physical": bool(support_box_physical),
        "support_box_default_center": support_box_default_center.astype(np.float32),
        "support_box_default_scale": support_box_default_scale.astype(np.float32),
        "support_box_center": support_box_center.astype(np.float32),
        "support_box_scale": support_box_scale.astype(np.float32),
        "support_box_shape_index": support_box_shape_index,
        "support_box_shape_label": ("tabletop_support_box" if support_box_physical else None),
        "support_box_geometry_source": str(support_box_geometry["source"]),
        "support_box_normal_axis_index": int(support_box_geometry["normal_axis_index"]),
        "support_box_normal_axis": str(support_box_geometry["normal_axis"]),
        "support_box_normal_sign": float(support_box_geometry["normal_sign"]),
        "floor_center": floor_center.astype(np.float32),
        "floor_scale": floor_scale.astype(np.float32),
        "floor_z": float(floor_z),
        "camera_presets": camera_presets,
        "joint_q_init": robot_joint_init.astype(np.float32),
        "tabletop_control_mode": str(args.tabletop_control_mode) if tabletop_task else None,
        "joint_task_phases": (
            [
                {
                    "name": pos_phase["name"],
                    "duration": pos_phase["duration"],
                    "start_q": joint_phase["start_q"],
                    "end_q": joint_phase["end_q"],
                }
                for pos_phase, joint_phase in zip(
                    task_phases,
                    _tabletop_joint_phase_waypoints(str(args.tabletop_joint_reference_family)),
                    strict=True,
                )
            ]
            if tabletop_joint_path_mode
            else None
        ),
        "endpoint_indices": endpoint_indices.astype(np.int32),
        "anchor_indices": anchor_indices.astype(np.int32),
        "anchor_positions": shifted_q[anchor_indices].astype(np.float32),
        "support_patch_indices": anchor_indices.astype(np.int32),
        "support_patch_center_m": support_patch_center_shifted.astype(np.float32),
        "visible_support_center_m": support_patch_center_shifted.astype(np.float32),
        "rope_center": rope_center.astype(np.float32),
        "table_center": stage_center.astype(np.float32),
        "table_top_z": (None if table_top_z is None else float(table_top_z)),
        "tabletop_rope_top_z": (None if tabletop_target_z is None else float(tabletop_rope_top_z)),
        "tabletop_target_z": tabletop_target_z,
        "tabletop_joint_reference_family": (None if not tabletop_task else str(args.tabletop_joint_reference_family)),
        "tabletop_push_focus": (
            tabletop_push_focus.astype(np.float32) if tabletop_task else rope_center.astype(np.float32)
        ),
        "render_edges": (np.zeros((0, 2), dtype=np.int32) if rigid_only_stage else render_edges),
        "particle_contacts": bool(particle_contacts),
        "particle_contact_kernel": bool(particle_contact_kernel),
        "total_object_mass": (0.0 if rigid_only_stage else float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum())),
        "release_phase_start_s": _phase_start_time(
            task_phases,
            "release" if drop_release_task else ("retract" if tabletop_task else "release_retract"),
        ),
        "visible_tool_enabled": bool(visible_tool_enabled),
        "visible_tool_mode": str(args.visible_tool_mode),
        "visible_tool_body_index": (None if visible_tool_body_index is None else int(visible_tool_body_index)),
        "visible_tool_body_label": visible_tool_body_label,
        "visible_tool_shape_index": visible_tool_shape_index,
        "visible_tool_shape_label": ("visible_tool_capsule" if visible_tool_enabled else None),
        "visible_tool_radius": (float(args.visible_tool_radius) if visible_tool_enabled else None),
        "visible_tool_half_height": (float(args.visible_tool_half_height) if visible_tool_enabled else None),
        "visible_tool_total_length": (
            2.0 * float(args.visible_tool_radius) + 2.0 * float(args.visible_tool_half_height)
            if visible_tool_enabled
            else None
        ),
        "visible_tool_offset_local": np.asarray(args.visible_tool_offset, dtype=np.float32),
        "visible_tool_local_transform": visible_tool_local_transform.astype(np.float32),
        "visible_tool_axis": str(args.visible_tool_axis),
        "visible_tool_color": visible_tool_color.astype(np.float32),
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
    tabletop_task = str(args.task) == TABLETOP_PUSH_TASK
    rigid_only_stage = tabletop_task and str(meta.get("blocking_stage", args.blocking_stage)) == "rigid_only"
    tabletop_joint_override_mode = tabletop_task and str(meta.get("tabletop_control_mode")) == "joint_trajectory"
    tabletop_joint_drive_mode = tabletop_task and str(meta.get("tabletop_control_mode")) == "joint_target_drive"
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
        joint_attach_ke=float(args.solver_joint_attach_ke),
        joint_attach_kd=float(args.solver_joint_attach_kd),
        enable_tri_contact=cfg.enable_tri_contact,
    )
    control = model.control() if tabletop_joint_drive_mode else None
    control_joint_target_pos = None
    control_joint_target_vel = None
    state_in = model.state()
    state_out = model.state()
    contacts = model.contacts() if newton_import_ir._use_collision_pipeline(cfg, ir_obj) else None
    prev_joint_q = np.asarray(meta["joint_q_init"], dtype=np.float32).copy()
    state_in.joint_q.assign(prev_joint_q)
    state_in.joint_qd.zero_()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    if control is not None and getattr(control, "joint_target_pos", None) is not None:
        control_joint_target_pos = wp.zeros(control.joint_target_pos.shape, dtype=wp.float32, device=device)
    if control is not None and getattr(control, "joint_target_vel", None) is not None:
        control_joint_target_vel = wp.zeros(control.joint_target_vel.shape, dtype=wp.float32, device=device)
    if control is not None and getattr(control, "joint_target_pos", None) is not None:
        control_joint_target_pos.assign(prev_joint_q.astype(np.float32))
        wp.copy(dest=control.joint_target_pos, src=control_joint_target_pos)
    if control is not None and getattr(control, "joint_target_vel", None) is not None:
        control_joint_target_vel.zero_()
        wp.copy(dest=control.joint_target_vel, src=control_joint_target_vel)
    if control is not None and getattr(control, "joint_f", None) is not None:
        control.joint_f.zero_()

    sim_dt = float(args.sim_dt) if args.sim_dt is not None else float(newton_import_ir.ir_scalar(ir_obj, "sim_dt"))
    substeps = max(1, int(args.substeps))
    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)
    settle_drag = 0.0
    if drop_release_task and "drag_damping" in ir_obj:
        settle_drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.pre_release_settle_damping_scale)
    tabletop_preroll_drag = 0.0
    if tabletop_task and "drag_damping" in ir_obj:
        tabletop_preroll_drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.tabletop_preroll_damping_scale)
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

    ik_joint_q = None
    pos_obj = None
    rot_obj = None
    ik_solver = None
    if not (tabletop_joint_override_mode or tabletop_joint_drive_mode):
        ik_joint_q = wp.array(np.asarray(meta["joint_q_init"], dtype=np.float32).reshape(1, -1), dtype=wp.float32, device=device)
        pos_obj = ik.IKObjectivePosition(
            link_index=int(meta.get("ik_reference_body_index", meta["ee_body_index"])),
            link_offset=wp.vec3(*np.asarray(meta.get("ik_reference_offset_local", meta["ee_offset_local"]), dtype=np.float32).tolist()),
            target_positions=wp.array([wp.vec3(*np.asarray(meta["ee_target_start"], dtype=np.float32).tolist())], dtype=wp.vec3, device=device),
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=int(meta.get("ik_reference_body_index", meta["ee_body_index"])),
            link_offset_rotation=wp.quat(*np.asarray(meta.get("ik_reference_offset_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32).tolist()),
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
    particle_q0 = (
        state_in.particle_q.numpy().astype(np.float32)
        if state_in.particle_q is not None
        else np.zeros((0, 3), dtype=np.float32)
    )
    body_q0 = state_in.body_q.numpy().astype(np.float32)
    body_qd0 = state_in.body_qd.numpy().astype(np.float32)
    particle_q_all = store.allocate("particle_q_all", (n_frames, particle_q0.shape[0], 3), np.float32)
    particle_q_object = store.allocate("particle_q_object", (n_frames, n_obj, 3), np.float32)
    body_q = store.allocate("body_q", (n_frames, body_q0.shape[0], body_q0.shape[1]), np.float32)
    body_vel = store.allocate("body_vel", (n_frames, body_qd0.shape[0], 3), np.float32)
    ee_target_pos = store.allocate("ee_target_pos", (n_frames, 3), np.float32)
    particle_radius = model.particle_radius.numpy().astype(np.float32)[:n_obj]
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

    if (drop_release_task and float(args.drop_preroll_settle_seconds) > 0.0) or (
        tabletop_task and (not rigid_only_stage) and float(args.tabletop_preroll_settle_seconds) > 0.0
    ):
        preroll_duration_s = (
            float(args.drop_preroll_settle_seconds)
            if drop_release_task
            else float(args.tabletop_preroll_settle_seconds)
        )
        preroll_max_frames = max(1, int(np.ceil(preroll_duration_s / max(frame_dt, 1.0e-12))))
        preroll_particle_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        preroll_support_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        preroll_com_h_speed = np.zeros((preroll_max_frames,), dtype=np.float32)
        prev_q_obj_preroll = None
        prev_rope_com_preroll = None
        preroll_target_pos = np.asarray(meta["task_phases"][0]["end"], dtype=np.float32)
        preroll_target_quat = np.asarray(meta["ee_target_quat"], dtype=np.float32)
        tabletop_preroll_joint_q = None
        if tabletop_joint_override_mode or tabletop_joint_drive_mode:
            _, tabletop_preroll_joint_q = _joint_phase_state(0.0, meta)
        for preroll_frame in range(preroll_max_frames):
            q_obj_preroll = (
                state_in.particle_q.numpy().astype(np.float32)[:n_obj]
                if state_in.particle_q is not None
                else np.zeros((0, 3), dtype=np.float32)
            )
            if q_obj_preroll.shape[0] == 0:
                preroll_frame_count = int(preroll_frame + 1)
                break
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

            if drop_release_task:
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
            else:
                preroll_settle_metrics, preroll_ok = _settle_window_metrics_from_series(
                    preroll_particle_speed,
                    preroll_com_h_speed,
                    preroll_support_speed,
                    frame_idx=preroll_frame,
                    window_frames=max(1, int(np.ceil(0.20 / max(frame_dt, 1.0e-12)))),
                    particle_speed_threshold_m_s=0.08,
                    com_horizontal_speed_threshold_m_s=0.05,
                    support_patch_speed_threshold_m_s=0.02,
                    frame_dt=float(frame_dt),
                )
            preroll_frame_count = int(preroll_frame + 1)
            if preroll_ok:
                preroll_settle_pass = True
                preroll_settle_time_s = float(preroll_frame * frame_dt)
                break

            for _ in range(substeps):
                state_in.clear_forces()
                if tabletop_joint_override_mode:
                    joint_target_np = np.asarray(tabletop_preroll_joint_q, dtype=np.float32).copy()
                    joint_target_np[7:9] = float(args.gripper_open)
                    joint_target_qd = np.zeros_like(joint_target_np, dtype=np.float32)
                    prev_joint_q = joint_target_np.copy()
                    state_in.joint_q.assign(joint_target_np)
                    state_in.joint_qd.assign(joint_target_qd)
                    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                elif tabletop_joint_drive_mode:
                    joint_target_np = np.asarray(tabletop_preroll_joint_q, dtype=np.float32).copy()
                    joint_target_np[7:9] = float(args.gripper_open)
                    prev_joint_q = joint_target_np.copy()
                    if control is not None and getattr(control, "joint_target_pos", None) is not None:
                        control_joint_target_pos.assign(joint_target_np)
                        wp.copy(dest=control.joint_target_pos, src=control_joint_target_pos)
                    if control is not None and getattr(control, "joint_target_vel", None) is not None:
                        control_joint_target_vel.zero_()
                        wp.copy(dest=control.joint_target_vel, src=control_joint_target_vel)
                    if control is not None and getattr(control, "joint_f", None) is not None:
                        control.joint_f.zero_()
                elif tabletop_task:
                    pos_obj.set_target_position(0, wp.vec3(*preroll_target_pos.tolist()))
                    rot_obj.set_target_rotation(0, wp.vec4(*preroll_target_quat.tolist()))
                    ik_solver.step(ik_joint_q, ik_joint_q, iterations=int(args.ik_iters))
                    joint_target_np = ik_joint_q.numpy().reshape(-1).astype(np.float32)
                    blend = float(np.clip(float(args.ik_target_blend), 0.0, 1.0))
                    if blend < 1.0:
                        joint_target_np = prev_joint_q + blend * (joint_target_np - prev_joint_q)
                    joint_target_np[7:9] = float(args.gripper_open)
                    joint_target_qd = (joint_target_np - prev_joint_q) / sim_dt
                    prev_joint_q = joint_target_np.copy()
                    state_in.joint_q.assign(joint_target_np)
                    state_in.joint_qd.assign(joint_target_qd)
                    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                if contacts is not None:
                    model.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, sim_dt)
                if tabletop_joint_override_mode:
                    state_out.joint_q.assign(joint_target_np)
                    state_out.joint_qd.assign(joint_target_qd)
                    newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
                elif tabletop_joint_drive_mode:
                    # SemiImplicit advances articulations in maximal/body coordinates.
                    # Recover reduced coordinates from the solver-integrated body state
                    # so diagnostics and any later consumers see the actual motion rather
                    # than a stale pre-step joint_q snapshot.
                    newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
                elif tabletop_task:
                    state_out.joint_q.assign(joint_target_np)
                    state_out.joint_qd.assign(joint_target_qd)
                    newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
                else:
                    state_out.joint_q.assign(prev_joint_q)
                    state_out.joint_qd.zero_()
                    newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
                state_in, state_out = state_out, state_in
                preroll_drag = tabletop_preroll_drag if tabletop_task else settle_drag
                if preroll_drag > 0.0 and n_obj > 0 and state_in.particle_q is not None and state_in.particle_qd is not None:
                    if gravity_axis is not None:
                        wp.launch(
                            _apply_drag_correction_ignore_axis,
                            dim=n_obj,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
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
                            inputs=[state_in.particle_q, state_in.particle_qd, n_obj, sim_dt, preroll_drag],
                            device=device,
                        )
        if preroll_settle_pass is None:
            preroll_settle_pass = False
            preroll_settle_time_s = float(preroll_frame_count * frame_dt)
        if tabletop_joint_drive_mode and bool(args.tabletop_reset_robot_after_preroll):
            joint_reset_np = np.asarray(meta["joint_q_init"], dtype=np.float32).copy()
            joint_reset_qd = np.zeros_like(joint_reset_np, dtype=np.float32)
            prev_joint_q = joint_reset_np.copy()
            for reset_state in (state_in, state_out):
                reset_state.joint_q.assign(joint_reset_np)
                reset_state.joint_qd.assign(joint_reset_qd)
                newton.eval_fk(model, reset_state.joint_q, reset_state.joint_qd, reset_state)
            if control is not None and getattr(control, "joint_target_pos", None) is not None:
                control_joint_target_pos.assign(joint_reset_np)
                wp.copy(dest=control.joint_target_pos, src=control_joint_target_pos)
            if control is not None and getattr(control, "joint_target_vel", None) is not None:
                control_joint_target_vel.zero_()
                wp.copy(dest=control.joint_target_vel, src=control_joint_target_vel)
            if control is not None and getattr(control, "joint_f", None) is not None:
                control.joint_f.zero_()

    t0 = time.perf_counter()
    for frame in range(n_frames):
        if replay is not None:
            state_in.body_q.assign(replay["body_q"][frame])
            state_in.body_qd.assign(replay["body_qd"][frame])
        q = (
            state_in.particle_q.numpy().astype(np.float32)
            if state_in.particle_q is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        particle_q_all[frame] = q
        particle_q_object[frame] = q[:n_obj]
        q_obj = particle_q_object[frame]
        if q_obj.shape[0] > 0:
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
        else:
            support_patch_center_series[frame] = np.asarray(meta["table_center"], dtype=np.float32)
        body_q[frame] = state_in.body_q.numpy().astype(np.float32)
        body_vel[frame] = state_in.body_qd.numpy().astype(np.float32)[:, :3]
        if replay is not None:
            ee_target_pos[frame] = replay["ee_target_pos"][frame]
        elif tabletop_joint_override_mode or tabletop_joint_drive_mode:
            _, joint_target_frame = _joint_phase_state(float(frame) * frame_dt, meta)
            joint_target_frame = np.asarray(joint_target_frame, dtype=np.float32).copy()
            joint_target_frame[7:9] = float(args.gripper_open)
            ee_target_pos[frame] = _fk_gripper_center_from_joint_q(
                model,
                joint_target_frame,
                left_finger_idx=int(meta["left_finger_index"]),
                right_finger_idx=int(meta["right_finger_index"]),
            )
        else:
            _, target_pos_frame, _ = _task_phase_state(float(frame) * frame_dt, meta)
            ee_target_pos[frame] = target_pos_frame

        frame_tracking_err = None
        frame_clearance = None
        if tabletop_task:
            frame_gripper_center = _gripper_center_world_position(
                body_q[frame],
                int(meta["left_finger_index"]),
                int(meta["right_finger_index"]),
            )
            frame_tracking_err = float(np.linalg.norm(frame_gripper_center - ee_target_pos[frame]))
            if q_obj.shape[0] > 0:
                frame_clearance, _, _ = _min_gripper_proxy_clearance(
                    q_obj,
                    particle_radius,
                    {
                        "gripper_center": frame_gripper_center,
                        "left_finger": body_q[frame, int(meta["left_finger_index"]), :3],
                        "right_finger": body_q[frame, int(meta["right_finger_index"]), :3],
                    },
                    float(args.ee_contact_radius),
                )

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
            elif tabletop_joint_override_mode:
                _, joint_target_np = _joint_phase_state(sim_t, meta)
                joint_target_np = np.asarray(joint_target_np, dtype=np.float32).copy()
                joint_target_np[7:9] = float(args.gripper_open)
                joint_target_qd = (joint_target_np - prev_joint_q) / sim_dt
                prev_joint_q = joint_target_np.copy()

                state_in.joint_q.assign(joint_target_np)
                state_in.joint_qd.assign(joint_target_qd)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            elif tabletop_joint_drive_mode:
                _, joint_target_np = _joint_phase_state(sim_t, meta)
                joint_target_np = np.asarray(joint_target_np, dtype=np.float32).copy()
                joint_target_np[7:9] = float(args.gripper_open)
                prev_joint_q = joint_target_np.copy()
                if control is not None and getattr(control, "joint_target_pos", None) is not None:
                    control_joint_target_pos.assign(joint_target_np)
                    wp.copy(dest=control.joint_target_pos, src=control_joint_target_pos)
                if control is not None and getattr(control, "joint_target_vel", None) is not None:
                    control_joint_target_vel.zero_()
                    wp.copy(dest=control.joint_target_vel, src=control_joint_target_vel)
                if control is not None and getattr(control, "joint_f", None) is not None:
                    control.joint_f.zero_()
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
            solver.step(state_in, state_out, control, contacts, sim_dt)
            if replay is not None:
                state_out.body_q.assign(body_q_frame)
                state_out.body_qd.assign(body_qd_frame)
            elif tabletop_joint_override_mode:
                state_out.joint_q.assign(joint_target_np)
                state_out.joint_qd.assign(joint_target_qd)
                newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
            elif tabletop_joint_drive_mode:
                newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
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
            if active_drag > 0.0 and n_obj > 0 and state_in.particle_q is not None and state_in.particle_qd is not None:
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

        if (frame + 1) % 25 == 0 or frame == n_frames - 1:
            if tabletop_task and frame_tracking_err is not None and frame_clearance is not None:
                phase_name_frame = str(_task_phase_state(float(frame) * frame_dt, meta)[0])
                print(
                    f"  frame {frame + 1}/{n_frames}"
                    f" phase={phase_name_frame}"
                    f" target_err={frame_tracking_err:.4f}m"
                    f" min_clearance={float(frame_clearance):.4f}m",
                    flush=True,
                )
            else:
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
    tabletop_task = str(args.task) == TABLETOP_PUSH_TASK
    rigid_only_stage = tabletop_task and str(meta.get("blocking_stage", args.blocking_stage)) == "rigid_only"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps)
    profile = str(getattr(args, "camera_profile", "hero"))
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
            extra_rules = None
            if bool(meta.get("visible_tool_enabled", False)):
                tool_color = np.asarray(meta.get("visible_tool_color", [0.96, 0.48, 0.14]), dtype=np.float32)
                extra_rules = [
                    (
                        lambda name: "visible_tool_capsule" in name,
                        tuple(float(v) for v in tool_color.tolist()),
                    )
                ]
            apply_viewer_shape_colors(viewer, model, extra_rules=extra_rules)
        except Exception:
            pass
        viewer.show_particles = True
        viewer.show_triangles = False
        viewer.show_visual = True
        viewer.show_static = True
        viewer.show_collision = str(args.render_mode) == "debug"
        viewer.show_contacts = False
        viewer.show_ui = False
        viewer.picking_enabled = False
        try:
            viewer.camera.fov = float(args.camera_fov)
        except Exception:
            pass
        cam_pos = np.asarray(args.camera_pos, dtype=np.float32)
        viewer.set_camera(wp.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])), float(args.camera_pitch), float(args.camera_yaw))

        particle_radius_sim = (
            model.particle_radius.numpy().astype(np.float32)
            if getattr(model, "particle_radius", None) is not None
            else np.zeros((0,), dtype=np.float32)
        )
        render_radii = compute_visual_particle_radii(
            particle_radius_sim,
            radius_scale=(
                None if args.particle_radius_vis_scale is None else float(args.particle_radius_vis_scale)
            ),
            radius_cap=(
                None if args.particle_radius_vis_min is None else float(args.particle_radius_vis_min)
            ),
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
        finger_box_entries = _finger_box_entries(model, meta) if tabletop_task else []
        visible_tool_entry = _visible_tool_entry(model, meta)
        visible_tool_color = np.asarray(meta.get("visible_tool_color", [0.96, 0.48, 0.14]), dtype=np.float32)
        stage_center = np.asarray(meta["stage_center"], dtype=np.float32)
        stage_scale = np.asarray(meta["stage_scale"], dtype=np.float32)
        robot_base_center = np.asarray(meta["robot_base_center"], dtype=np.float32)
        robot_base_scale = np.asarray(meta["robot_base_scale"], dtype=np.float32)
        support_box_enabled = bool(meta.get("support_box_enabled", False))
        support_box_center = np.asarray(meta.get("support_box_center", robot_base_center), dtype=np.float32)
        support_box_scale = np.asarray(meta.get("support_box_scale", robot_base_scale), dtype=np.float32)
        floor_center = np.asarray(meta["floor_center"], dtype=np.float32)
        floor_scale = np.asarray(meta["floor_scale"], dtype=np.float32)
        table_center = np.asarray(meta.get("table_center", stage_center), dtype=np.float32)
        table_top_z = meta.get("table_top_z")
        table_top_z = None if table_top_z is None else float(table_top_z)
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

            stage_banner = None
            if tabletop_task and str(args.tabletop_control_mode) == "joint_target_drive":
                if rigid_only_stage:
                    stage_banner = f"RIGID ONLY — NO ROPE BY DESIGN | {str(args.render_mode).upper()} | {profile.upper()}"
                else:
                    stage_banner = f"ROPE INTEGRATED | DIRECT-FINGER BLOCKING CANDIDATE | {str(args.render_mode).upper()} | {profile.upper()}"

            for out_idx, sim_idx in enumerate(render_indices):
                if state.particle_q is not None:
                    state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
                state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))

                sim_t = float(sim_idx) * sim_frame_dt
                phase_name, _, _ = _task_phase_state(sim_t, meta)
                viewer.begin_frame(sim_t)
                viewer.log_state(state)
                tool_center_w = None
                tool_quat_w = None
                if visible_tool_entry is not None:
                    tool_center_w, tool_quat_w = _tool_world_transform(
                        sim_data["body_q"][sim_idx],
                        visible_tool_entry,
                    )
                gripper_proxies = _gripper_contact_proxies_world(
                    sim_data["body_q"][sim_idx], left_finger_idx, right_finger_idx
                )
                gripper_center = gripper_proxies["gripper_center"].astype(np.float32, copy=False)
                q_obj = sim_data["particle_q_object"][sim_idx]
                particle_radius = particle_radius_sim[: q_obj.shape[0]]
                actual_min_clearance = None
                actual_contact_source = None
                actual_tool_clearance = None
                actual_tool_contact_source = None
                if str(args.render_mode) == "debug":
                    if tabletop_task and (not rigid_only_stage) and visible_tool_entry is not None:
                        actual_tool_clearance, actual_tool_contact_source = _min_capsule_clearance(
                            q_obj,
                            particle_radius,
                            sim_data["body_q"][sim_idx],
                            visible_tool_entry,
                        )
                        if tool_center_w is not None and tool_quat_w is not None:
                            tool_a, tool_b = _capsule_segment_endpoints(
                                tool_center_w,
                                tool_quat_w,
                                float(visible_tool_entry["half_height"]),
                            )
                            viewer.log_shapes(
                                "/demo/visible_tool_endpoints",
                                newton.GeoType.SPHERE,
                                0.004,
                                wp.array(
                                    [
                                        wp.transform(wp.vec3(*tool_a.tolist()), wp.quat_identity()),
                                        wp.transform(wp.vec3(*tool_b.tolist()), wp.quat_identity()),
                                    ],
                                    dtype=wp.transform,
                                    device=device,
                                ),
                                wp.array(
                                    [wp.vec3(*visible_tool_color.tolist()), wp.vec3(*visible_tool_color.tolist())],
                                    dtype=wp.vec3,
                                    device=device,
                                ),
                            )
                    if tabletop_task and (not rigid_only_stage) and finger_box_entries:
                        actual_min_clearance, actual_contact_source, _ = _min_finger_box_clearance(
                            q_obj,
                            particle_radius,
                            sim_data["body_q"][sim_idx],
                            finger_box_entries,
                        )
                        tip_xforms = []
                        for entry in finger_box_entries:
                            if not bool(entry.get("is_tip", False)):
                                continue
                            center_w, _ = _combine_world_transform(
                                sim_data["body_q"][sim_idx][int(entry["body_index"])],
                                np.asarray(entry["local_transform"], dtype=np.float32),
                            )
                            tip_xforms.append(wp.transform(wp.vec3(*center_w.tolist()), wp.quat_identity()))
                        if tip_xforms:
                            viewer.log_shapes(
                                "/demo/finger_tip_boxes",
                                newton.GeoType.SPHERE,
                                0.006,
                                wp.array(tip_xforms, dtype=wp.transform, device=device),
                                wp.array([wp.vec3(1.0, 0.40, 0.92) for _ in tip_xforms], dtype=wp.vec3, device=device),
                            )
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
                    elif tabletop_task:
                        viewer.log_shapes(
                            "/demo/floor",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in floor_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*floor_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.24, 0.22, 0.19)], dtype=wp.vec3, device=device),
                        )
                        # Presentation hides the oversized pedestal block so the contact patch stays readable.
                        # Debug and validation still render the full geometry honestly.
                        viewer.log_shapes(
                            "/demo/table_top",
                            newton.GeoType.BOX,
                            tuple(float(v) for v in stage_scale.tolist()),
                            wp.array(
                                [wp.transform(wp.vec3(*table_center.tolist()), wp.quat_identity())],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(0.56, 0.46, 0.34)], dtype=wp.vec3, device=device),
                        )
                        leg_height = max(0.10, float(table_center[2] - ground_grid_z - stage_scale[2]))
                        leg_hz = 0.5 * leg_height
                        if profile == "hero":
                            leg_scale = (0.018, 0.018, leg_hz)
                            leg_color = wp.vec3(0.34, 0.29, 0.23)
                        else:
                            leg_scale = (0.024, 0.024, leg_hz)
                            leg_color = wp.vec3(0.39, 0.31, 0.24)
                        leg_offset_x = max(0.06, 0.76 * float(stage_scale[0]))
                        leg_offset_y = max(0.05, 0.76 * float(stage_scale[1]))
                        leg_center_z = float(ground_grid_z + leg_hz)
                        leg_centers = [
                            [float(table_center[0] - leg_offset_x), float(table_center[1] - leg_offset_y), leg_center_z],
                            [float(table_center[0] - leg_offset_x), float(table_center[1] + leg_offset_y), leg_center_z],
                            [float(table_center[0] + leg_offset_x), float(table_center[1] - leg_offset_y), leg_center_z],
                            [float(table_center[0] + leg_offset_x), float(table_center[1] + leg_offset_y), leg_center_z],
                        ]
                        viewer.log_shapes(
                            "/demo/table_legs",
                            newton.GeoType.BOX,
                            leg_scale,
                            wp.array(
                                [wp.transform(wp.vec3(*center), wp.quat_identity()) for center in leg_centers],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([leg_color for _ in leg_centers], dtype=wp.vec3, device=device),
                        )
                        if support_box_enabled and not (profile == "hero" and bool(args.tabletop_hero_hide_pedestal)):
                            viewer.log_shapes(
                                "/demo/support_box",
                                newton.GeoType.BOX,
                                tuple(float(v) for v in support_box_scale.tolist()),
                                wp.array(
                                    [wp.transform(wp.vec3(*support_box_center.tolist()), wp.quat_identity())],
                                    dtype=wp.transform,
                                    device=device,
                                ),
                                wp.array([wp.vec3(0.44, 0.38, 0.31)], dtype=wp.vec3, device=device),
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
                    if visible_tool_entry is not None and tool_center_w is not None and tool_quat_w is not None:
                        viewer.log_shapes(
                            "/demo/visible_tool_capsule",
                            newton.GeoType.CAPSULE,
                            (
                                float(visible_tool_entry["radius"]),
                                float(visible_tool_entry["half_height"]),
                            ),
                            wp.array(
                                [
                                    wp.transform(
                                        wp.vec3(*tool_center_w.tolist()),
                                        wp.quat(*tool_quat_w.tolist()),
                                    )
                                ],
                                dtype=wp.transform,
                                device=device,
                            ),
                            wp.array([wp.vec3(*visible_tool_color.tolist())], dtype=wp.vec3, device=device),
                        )

                if q_obj.shape[0] > 0 and rope_edges.size and rope_line_starts_wp is not None and rope_line_ends_wp is not None:
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
                overlay_lines: list[str] = []
                if stage_banner is not None:
                    overlay_lines.append(stage_banner)
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
                    if q_obj.shape[0] > 0:
                        min_clearance, min_proxy_name, _ = _min_gripper_proxy_clearance(
                            q_obj,
                            particle_radius,
                            gripper_proxies,
                            float(args.ee_contact_radius),
                        )
                    else:
                        min_clearance, min_proxy_name = float("nan"), "none"
                    if tabletop_task and actual_tool_clearance is not None:
                        contact_state = "ON" if float(actual_tool_clearance) <= 0.0 else "OFF"
                        contact_line = (
                            f"contact: {contact_state} via {actual_tool_contact_source} | actual tool clearance: {1000.0 * float(actual_tool_clearance):.1f} mm"
                        )
                    elif tabletop_task and actual_min_clearance is not None:
                        contact_state = "ON" if float(actual_min_clearance) <= 0.0 else "OFF"
                        contact_line = (
                            f"contact: {contact_state} via {actual_contact_source} | actual finger-box clearance: {1000.0 * float(actual_min_clearance):.1f} mm"
                        )
                    else:
                        if q_obj.shape[0] > 0:
                            contact_state = "ON" if min_clearance <= 0.0 else "OFF"
                            contact_line = (
                                f"contact: {contact_state} via {min_proxy_name} | approx gripper clearance: {1000.0 * min_clearance:.1f} mm"
                            )
                        else:
                            contact_line = "contact: rope absent in rigid_only stage | direct finger vs table is audited offline"
                    tracking_line = f"tracking err: {tracking_err:.3f} m | gripper speed: {speed:.3f} m/s"
                    if release_time_s is not None:
                        tracking_line += f" | t_release: {float(release_time_s):.3f}s"
                    overlay_lines.extend(
                        [
                            f"task: {args.task} | phase: {phase_name}",
                            (
                                "Native Newton Franka Panda + tabletop rope push | yellow=target, cyan=gripper center"
                                if tabletop_task
                                else "Native Newton Franka Panda + rope drop baseline | yellow=target, cyan=gripper center"
                            ),
                            tracking_line,
                            contact_line,
                        ]
                    )
                if overlay_lines:
                    frame = overlay_text_lines_rgb(
                        frame,
                        overlay_lines,
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
    tabletop_task = str(args.task) == TABLETOP_PUSH_TASK
    rigid_only_stage = tabletop_task and str(meta.get("blocking_stage", args.blocking_stage)) == "rigid_only"
    particle_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    body_vel = np.asarray(sim_data["body_vel"], dtype=np.float32)
    target_pos = np.asarray(sim_data["ee_target_pos"], dtype=np.float32)
    if rigid_only_stage:
        left_finger_idx = int(meta["left_finger_index"])
        right_finger_idx = int(meta["right_finger_index"])
        gripper_center = np.stack(
            [_gripper_center_world_position(body_q[i], left_finger_idx, right_finger_idx) for i in range(body_q.shape[0])]
        )
        tracking_error = np.linalg.norm(gripper_center - target_pos, axis=1)
        frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
        gripper_speed = np.linalg.norm(np.diff(gripper_center, axis=0), axis=1) / frame_dt if body_q.shape[0] > 1 else np.zeros((1,), dtype=np.float32)
        return {
            "ir_path": str(args.ir.resolve()),
            "output_mp4": str(out_mp4),
            "frames": int(body_q.shape[0]),
            "sim_dt": float(sim_data["sim_dt"]),
            "substeps": int(sim_data["substeps"]),
            "frame_dt": float(frame_dt),
            "video_slowdown": float(args.slowdown),
            "task_duration_s": float(_total_task_duration(args)),
            "wall_time_sec": float(sim_data["wall_time"]),
            "task": str(args.task),
            "blocking_stage": "rigid_only",
            "rope_present": False,
            "render_mode": str(args.render_mode),
            "camera_profile": str(args.camera_profile),
            "recommended_hero_camera": meta["camera_presets"]["hero"],
            "recommended_validation_camera": meta["camera_presets"]["validation"],
            "robot_motion_mode": str(sim_data.get("robot_motion_mode", meta.get("robot_motion_mode", "ik"))),
            "history_storage_mode": str(sim_data["history_storage_mode"]),
            "history_storage_files": sim_data["history_storage_files"],
            "robot_geometry": "native_franka",
            "ee_body_index": int(meta["ee_body_index"]),
            "visible_tool_enabled": False,
            "tabletop_control_mode": str(args.tabletop_control_mode),
            "tabletop_initial_pose": str(args.tabletop_initial_pose),
            "tabletop_joint_reference_family": str(args.tabletop_joint_reference_family),
            "tabletop_settle_seconds": float(args.tabletop_settle_seconds),
            "tabletop_approach_seconds": float(args.tabletop_approach_seconds),
            "tabletop_push_seconds": float(args.tabletop_push_seconds),
            "tabletop_hold_seconds": float(args.tabletop_hold_seconds),
            "tabletop_retract_seconds": float(args.tabletop_retract_seconds),
            "tabletop_support_box_mode": str(args.tabletop_support_box_mode),
            "tabletop_support_box_offset": [float(v) for v in np.asarray(args.tabletop_support_box_offset, dtype=np.float32)],
            "tabletop_support_box_scale": [float(v) for v in np.asarray(support_box_scale, dtype=np.float32)],
            "support_box_enabled": bool(meta.get("support_box_enabled", False)),
            "support_box_physical": bool(meta.get("support_box_physical", False)),
            "support_box_center": np.asarray(meta.get("support_box_center"), dtype=np.float32).astype(float).tolist(),
            "support_box_default_center": np.asarray(meta.get("support_box_default_center"), dtype=np.float32).astype(float).tolist(),
            "support_box_default_scale": np.asarray(meta.get("support_box_default_scale"), dtype=np.float32).astype(float).tolist(),
            "support_box_geometry_source": str(meta.get("support_box_geometry_source")),
            "support_box_normal_axis": str(meta.get("support_box_normal_axis")),
            "support_box_normal_axis_index": int(meta.get("support_box_normal_axis_index", 0)),
            "support_box_normal_sign": float(meta.get("support_box_normal_sign", 1.0)),
            "gripper_center_tracking_error_mean_m": float(np.mean(tracking_error)),
            "gripper_center_tracking_error_max_m": float(np.max(tracking_error)),
            "gripper_center_speed_max_m_s": float(np.max(gripper_speed)) if gripper_speed.size else 0.0,
            "gripper_center_path_length_m": float(np.sum(np.linalg.norm(np.diff(gripper_center, axis=0), axis=1))) if body_q.shape[0] > 1 else 0.0,
            "ground_z_m": float(meta.get("floor_z", 0.0)),
            "table_top_z_m": float(meta.get("table_top_z")),
            "contact_proxy_mode": "actual_finger_boxes_vs_table_box",
            "contact_proxy_radius_m": None,
            "contact_proxy_radii_m": {},
            "support_contact_proxy_mode": "actual_finger_boxes_vs_table_box",
            "support_contact_proxy_counts": {},
            "actual_tool_contact_started": False,
            "actual_finger_box_contact_started": False,
        }
    particle_radius = model.particle_radius.numpy().astype(np.float32)[: particle_q.shape[1]]
    finger_box_entries = _finger_box_entries(model, meta) if tabletop_task else []
    visible_tool_entry = _visible_tool_entry(model, meta)
    visible_tool_enabled = visible_tool_entry is not None

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
    table_top_z = meta.get("table_top_z")
    table_top_z = None if table_top_z is None else float(table_top_z)
    rope_surface_clearance_to_ground = np.min(
        particle_q[:, :, 2] - particle_radius[None, :], axis=1
    ) - ground_z
    rope_surface_clearance_to_table = (
        None
        if table_top_z is None
        else np.min(particle_q[:, :, 2] - particle_radius[None, :], axis=1) - table_top_z
    )
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
    pre_frames = [i for i, name in enumerate(phase_names) if name in {"pre_approach", "settle"}]
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
    actual_finger_box_clearance = []
    actual_finger_box_source = []
    actual_left_finger_box_clearance = []
    actual_right_finger_box_clearance = []
    actual_left_tip_box_clearance = []
    actual_right_tip_box_clearance = []
    actual_tool_clearance = []
    actual_tool_source = []
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
        actual_min_d = None
        actual_source = None
        if tabletop_task and finger_box_entries:
            actual_min_d, actual_source, per_box_min = _min_finger_box_clearance(
                particle_q[frame_idx],
                particle_radius,
                body_q[frame_idx],
                finger_box_entries,
            )
            actual_finger_box_clearance.append(float(actual_min_d))
            actual_finger_box_source.append(str(actual_source))
            actual_left_finger_box_clearance.append(float(per_box_min.get("left_any_box", np.inf)))
            actual_right_finger_box_clearance.append(float(per_box_min.get("right_any_box", np.inf)))
            actual_left_tip_box_clearance.append(float(per_box_min.get("left_tip_box", np.inf)))
            actual_right_tip_box_clearance.append(float(per_box_min.get("right_tip_box", np.inf)))
        tool_min_d, tool_source = _min_capsule_clearance(
            particle_q[frame_idx],
            particle_radius,
            body_q[frame_idx],
            visible_tool_entry,
        )
        if tool_min_d is not None:
            actual_tool_clearance.append(float(tool_min_d))
            actual_tool_source.append(str(tool_source))
        frame_has_contact = False
        if tabletop_task and visible_tool_enabled and tool_min_d is not None:
            frame_has_contact = bool(tool_min_d <= 0.0)
        elif tabletop_task and actual_min_d is not None:
            frame_has_contact = bool(actual_min_d <= 0.0)
        else:
            frame_has_contact = bool(min_d <= 0.0)
        if frame_has_contact:
            contact_frames.append(frame_idx)

    if tabletop_task and visible_tool_enabled and actual_tool_clearance:
        support_contact_mask = np.asarray([m <= 0.0 for m in actual_tool_clearance], dtype=bool)
    elif tabletop_task and actual_finger_box_clearance:
        support_contact_mask = np.asarray([m <= 0.0 for m in actual_finger_box_clearance], dtype=bool)
    else:
        support_contact_mask = np.asarray([m <= 0.0 for m in min_clearance], dtype=bool)
    tool_contact_mask = (
        np.asarray([m <= 0.0 for m in actual_tool_clearance], dtype=bool)
        if tabletop_task and visible_tool_enabled and actual_tool_clearance
        else np.zeros((particle_q.shape[0],), dtype=bool)
    )
    finger_box_contact_mask = (
        np.asarray([m <= 0.0 for m in actual_finger_box_clearance], dtype=bool)
        if tabletop_task and actual_finger_box_clearance
        else np.zeros((particle_q.shape[0],), dtype=bool)
    )
    support_contact_frames = np.flatnonzero(support_contact_mask).tolist()
    tool_contact_frames = np.flatnonzero(tool_contact_mask).tolist()
    finger_box_contact_frames = np.flatnonzero(finger_box_contact_mask).tolist()
    support_contact_duration_s = float(np.count_nonzero(support_contact_mask) * frame_dt)
    support_first_contact_frame = int(support_contact_frames[0]) if support_contact_frames else None
    support_last_contact_frame = int(support_contact_frames[-1]) if support_contact_frames else None
    if tabletop_task and visible_tool_enabled and actual_tool_clearance:
        support_contact_peak_frame = int(np.argmin(np.asarray(actual_tool_clearance, dtype=np.float32)))
    elif tabletop_task and actual_finger_box_clearance:
        support_contact_peak_frame = int(np.argmin(np.asarray(actual_finger_box_clearance, dtype=np.float32)))
    else:
        support_contact_peak_frame = int(np.argmin(np.asarray(min_clearance, dtype=np.float32))) if min_clearance else None

    ground_contact_mask = np.asarray(rope_surface_clearance_to_ground <= 0.0, dtype=bool)
    ground_contact_frames = np.flatnonzero(ground_contact_mask).tolist()
    table_contact_mask = (
        np.asarray(rope_surface_clearance_to_table <= 0.0, dtype=bool)
        if rope_surface_clearance_to_table is not None
        else np.zeros_like(ground_contact_mask)
    )
    table_contact_frames = np.flatnonzero(table_contact_mask).tolist()
    contact_mask = ground_contact_mask if drop_release_task else support_contact_mask
    contact_frames = ground_contact_frames if drop_release_task else support_contact_frames
    if drop_release_task:
        primary_clearance = rope_surface_clearance_to_ground
    elif tabletop_task and visible_tool_enabled and actual_tool_clearance:
        primary_clearance = np.asarray(actual_tool_clearance, dtype=np.float32)
    elif tabletop_task and actual_finger_box_clearance:
        primary_clearance = np.asarray(actual_finger_box_clearance, dtype=np.float32)
    else:
        primary_clearance = np.asarray(min_clearance, dtype=np.float32)

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
    elif drop_release_task or tabletop_task:
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
    if tabletop_task and visible_tool_enabled and actual_tool_source:
        support_source_series = actual_tool_source
    elif tabletop_task and actual_finger_box_source:
        support_source_series = actual_finger_box_source
    else:
        support_source_series = min_clearance_source
    for name, active in zip(support_source_series, support_contact_mask.tolist(), strict=False):
        if not active:
            continue
        support_source_counts[name] = int(support_source_counts.get(name, 0) + 1)

    min_clearance_np = np.asarray(min_clearance, dtype=np.float32)
    gripper_center_clearance_np = np.asarray(gripper_center_clearance, dtype=np.float32)
    left_finger_clearance_np = np.asarray(left_finger_clearance, dtype=np.float32)
    right_finger_clearance_np = np.asarray(right_finger_clearance, dtype=np.float32)
    finger_span_clearance_np = np.asarray(finger_span_clearance, dtype=np.float32)
    actual_finger_box_clearance_np = np.asarray(actual_finger_box_clearance, dtype=np.float32)
    actual_left_finger_box_clearance_np = np.asarray(actual_left_finger_box_clearance, dtype=np.float32)
    actual_right_finger_box_clearance_np = np.asarray(actual_right_finger_box_clearance, dtype=np.float32)
    actual_left_tip_box_clearance_np = np.asarray(actual_left_tip_box_clearance, dtype=np.float32)
    actual_right_tip_box_clearance_np = np.asarray(actual_right_tip_box_clearance, dtype=np.float32)
    actual_tool_clearance_np = np.asarray(actual_tool_clearance, dtype=np.float32)
    particle_radius_sim = model.particle_radius.numpy().astype(np.float32)
    render_radii = compute_visual_particle_radii(
        particle_radius_sim,
        radius_scale=(
            None if args.particle_radius_vis_scale is None else float(args.particle_radius_vis_scale)
        ),
        radius_cap=(
            None if args.particle_radius_vis_min is None else float(args.particle_radius_vis_min)
        ),
    )
    contact_peak_source = (
        None
        if contact_peak_frame is None
        else str(
            (
                actual_tool_source[int(contact_peak_frame)]
                if tabletop_task and visible_tool_enabled and actual_tool_source
                else (
                    actual_finger_box_source[int(contact_peak_frame)]
                    if tabletop_task and actual_finger_box_source
                    else min_clearance_source[int(contact_peak_frame)]
                )
            )
        )
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
            (
                "inactive_fixed_particles_until_release"
                if drop_release_task
                else ("free_tabletop_rope" if tabletop_task else "inactive_fixed_particles")
            )
        ),
        "anchor_mass_mode": str(args.anchor_mass_mode),
        "task": str(args.task),
        "blocking_stage": (None if not tabletop_task else str(meta.get("blocking_stage", args.blocking_stage))),
        "render_mode": str(args.render_mode),
        "camera_profile": str(args.camera_profile),
        "recommended_hero_camera": meta["camera_presets"]["hero"],
        "recommended_validation_camera": meta["camera_presets"]["validation"],
        "particle_contacts": bool(meta["particle_contacts"]),
        "particle_contact_kernel": bool(meta["particle_contact_kernel"]),
        "particle_radius_scale": float(args.particle_radius_scale),
        "particle_radius_mean_m": float(np.mean(particle_radius_sim)) if particle_radius_sim.size else None,
        "particle_radius_render_mean_m": float(np.mean(render_radii)) if render_radii.size else None,
        "particle_radius_vis_scale": (
            None if args.particle_radius_vis_scale is None else float(args.particle_radius_vis_scale)
        ),
        "particle_radius_vis_min": (
            None if args.particle_radius_vis_min is None else float(args.particle_radius_vis_min)
        ),
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
        "visible_tool_enabled": bool(visible_tool_enabled),
        "visible_tool_mode": meta.get("visible_tool_mode"),
        "visible_tool_body_index": meta.get("visible_tool_body_index"),
        "visible_tool_body_label": meta.get("visible_tool_body_label"),
        "visible_tool_shape_index": meta.get("visible_tool_shape_index"),
        "visible_tool_shape_label": meta.get("visible_tool_shape_label"),
        "visible_tool_axis": meta.get("visible_tool_axis"),
        "visible_tool_offset_local": (
            None
            if not visible_tool_enabled
            else np.asarray(meta.get("visible_tool_offset_local"), dtype=np.float32).astype(float).tolist()
        ),
        "visible_tool_radius_m": (None if not visible_tool_enabled else float(meta.get("visible_tool_radius"))),
        "visible_tool_half_height_m": (None if not visible_tool_enabled else float(meta.get("visible_tool_half_height"))),
        "visible_tool_total_length_m": (None if not visible_tool_enabled else float(meta.get("visible_tool_total_length"))),
        "ik_target_blend": float(args.ik_target_blend),
        "solver_joint_attach_ke": float(args.solver_joint_attach_ke),
        "solver_joint_attach_kd": float(args.solver_joint_attach_kd),
        "tabletop_control_mode": (None if not tabletop_task else str(args.tabletop_control_mode)),
        "tabletop_initial_pose": (None if not tabletop_task else str(args.tabletop_initial_pose)),
        "tabletop_joint_reference_family": (None if not tabletop_task else str(args.tabletop_joint_reference_family)),
        "tabletop_settle_seconds": (None if not tabletop_task else float(args.tabletop_settle_seconds)),
        "tabletop_approach_seconds": (None if not tabletop_task else float(args.tabletop_approach_seconds)),
        "tabletop_push_seconds": (None if not tabletop_task else float(args.tabletop_push_seconds)),
        "tabletop_hold_seconds": (None if not tabletop_task else float(args.tabletop_hold_seconds)),
        "tabletop_retract_seconds": (None if not tabletop_task else float(args.tabletop_retract_seconds)),
        "tabletop_support_box_mode": (None if not tabletop_task else str(args.tabletop_support_box_mode)),
        "tabletop_support_box_offset": (
            None if not tabletop_task else [float(v) for v in np.asarray(args.tabletop_support_box_offset, dtype=np.float32)]
        ),
        "tabletop_support_box_scale": (
            None if not tabletop_task else [float(v) for v in np.asarray(meta.get("support_box_scale", [0.0, 0.0, 0.0]), dtype=np.float32)]
        ),
        "support_box_enabled": bool(meta.get("support_box_enabled", False)),
        "support_box_physical": bool(meta.get("support_box_physical", False)),
        "support_box_shape_index": meta.get("support_box_shape_index"),
        "support_box_center": (
            None if not tabletop_task else np.asarray(meta.get("support_box_center", [0.0, 0.0, 0.0]), dtype=np.float32).astype(float).tolist()
        ),
        "support_box_default_center": (
            None if not tabletop_task else np.asarray(meta.get("support_box_default_center", [0.0, 0.0, 0.0]), dtype=np.float32).astype(float).tolist()
        ),
        "support_box_default_scale": (
            None if not tabletop_task else np.asarray(meta.get("support_box_default_scale", [0.0, 0.0, 0.0]), dtype=np.float32).astype(float).tolist()
        ),
        "support_box_geometry_source": (None if not tabletop_task else str(meta.get("support_box_geometry_source"))),
        "support_box_normal_axis": (None if not tabletop_task else str(meta.get("support_box_normal_axis"))),
        "support_box_normal_axis_index": (
            None if not tabletop_task else int(meta.get("support_box_normal_axis_index", 0))
        ),
        "support_box_normal_sign": (
            None if not tabletop_task else float(meta.get("support_box_normal_sign", 1.0))
        ),
        "tabletop_reset_robot_after_preroll": (None if not tabletop_task else bool(args.tabletop_reset_robot_after_preroll)),
        "tabletop_robot_base_offset": (
            None if not tabletop_task else [float(v) for v in np.asarray(args.tabletop_robot_base_offset, dtype=np.float32)]
        ),
        "tabletop_rope_height": (None if not tabletop_task else float(args.tabletop_rope_height)),
        "tabletop_approach_clearance_z": (
            None if not tabletop_task else float(args.tabletop_approach_clearance_z)
        ),
        "tabletop_contact_clearance_z": (
            None if not tabletop_task else float(args.tabletop_contact_clearance_z)
        ),
        "tabletop_push_clearance_z": (
            None if not tabletop_task else float(args.tabletop_push_clearance_z)
        ),
        "tabletop_retract_clearance_z": (
            None if not tabletop_task else float(args.tabletop_retract_clearance_z)
        ),
        "tabletop_ee_offset_z": (None if not tabletop_task else float(args.tabletop_ee_offset_z)),
        "tabletop_target_z": meta.get("tabletop_target_z"),
        "tabletop_rope_top_z": meta.get("tabletop_rope_top_z"),
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
        "table_top_z_m": table_top_z,
        "rope_min_z_m": float(np.min(rope_min_z)),
        "rope_surface_clearance_min_m": float(np.min(rope_surface_clearance_to_ground)),
        "rope_surface_clearance_to_table_min_m": (
            None if rope_surface_clearance_to_table is None else float(np.min(rope_surface_clearance_to_table))
        ),
        "table_contact_active_frames": int(len(table_contact_frames)),
        "table_contact_duration_s": float(np.count_nonzero(table_contact_mask) * frame_dt),
        "table_contact_peak_frame": (
            None if rope_surface_clearance_to_table is None else int(np.argmin(rope_surface_clearance_to_table))
        ),
        "support_contact_active_frames": int(len(support_contact_frames)),
        "support_contact_duration_s": float(support_contact_duration_s),
        "support_first_contact_frame": support_first_contact_frame,
        "support_last_contact_frame": support_last_contact_frame,
        "support_contact_peak_frame": support_contact_peak_frame,
        "support_contact_proxy_mode": (
            "actual_visible_tool_capsule"
            if tabletop_task and visible_tool_enabled and actual_tool_clearance
            else (
                "actual_finger_boxes"
                if tabletop_task and actual_finger_box_clearance
                else "min(gripper_center,left_finger,right_finger,finger_span)"
            )
        ),
        "support_contact_proxy_counts": support_source_counts,
        "actual_tool_contact_started": bool(tabletop_task and visible_tool_enabled and tool_contact_frames),
        "actual_tool_first_contact_frame": (
            (None if not tool_contact_frames else int(tool_contact_frames[0]))
            if tabletop_task and visible_tool_enabled
            else None
        ),
        "actual_tool_first_contact_time_s": (
            None
            if not (tabletop_task and visible_tool_enabled and tool_contact_frames)
            else float(int(tool_contact_frames[0]) * frame_dt)
        ),
        "actual_tool_clearance_min_m": (
            None if actual_tool_clearance_np.size == 0 else float(np.min(actual_tool_clearance_np))
        ),
        "actual_tool_peak_source": (
            contact_peak_source if tabletop_task and visible_tool_enabled and actual_tool_clearance else None
        ),
        "actual_tool_collider_mode": (
            None
            if not visible_tool_enabled
            else {
                "shape_type": "capsule",
                "radius_m": float(meta.get("visible_tool_radius")),
                "half_height_m": float(meta.get("visible_tool_half_height")),
            }
        ),
        "actual_finger_box_contact_started": bool(tabletop_task and actual_finger_box_clearance and finger_box_contact_frames),
        "actual_finger_box_first_contact_frame": (
            (None if not finger_box_contact_frames else int(finger_box_contact_frames[0]))
            if tabletop_task and actual_finger_box_clearance
            else None
        ),
        "actual_finger_box_first_contact_time_s": (
            None
            if not (tabletop_task and actual_finger_box_clearance and finger_box_contact_frames)
            else float(int(finger_box_contact_frames[0]) * frame_dt)
        ),
        "actual_finger_box_peak_source": (
            contact_peak_source if tabletop_task and actual_finger_box_clearance else None
        ),
        "actual_finger_box_clearance_min_m": (
            None if actual_finger_box_clearance_np.size == 0 else float(np.min(actual_finger_box_clearance_np))
        ),
        "actual_left_finger_box_clearance_min_m": (
            None if actual_left_finger_box_clearance_np.size == 0 else float(np.min(actual_left_finger_box_clearance_np))
        ),
        "actual_right_finger_box_clearance_min_m": (
            None if actual_right_finger_box_clearance_np.size == 0 else float(np.min(actual_right_finger_box_clearance_np))
        ),
        "actual_left_tip_box_clearance_min_m": (
            None if actual_left_tip_box_clearance_np.size == 0 else float(np.min(actual_left_tip_box_clearance_np))
        ),
        "actual_right_tip_box_clearance_min_m": (
            None if actual_right_tip_box_clearance_np.size == 0 else float(np.min(actual_right_tip_box_clearance_np))
        ),
        "support_patch_indices": np.asarray(meta.get("support_patch_indices", meta["anchor_indices"]), dtype=np.int32).astype(int).tolist(),
        "support_patch_center_m": np.asarray(meta.get("support_patch_center_m", meta["anchor_positions"].mean(axis=0)), dtype=np.float32).astype(float).tolist(),
        "visible_support_center_m": np.asarray(meta.get("visible_support_center_m", meta["anchor_positions"].mean(axis=0)), dtype=np.float32).astype(float).tolist(),
        "contact_proxy_mode": (
            "rope_surface_vs_ground_plane"
            if drop_release_task
            else ("actual_visible_tool_capsule" if tabletop_task and visible_tool_enabled and actual_tool_clearance else "min(gripper_center,left_finger,right_finger,finger_span)")
        ),
        "contact_proxy_radius_m": (
            0.0
            if drop_release_task
            else (float(meta.get("visible_tool_radius")) if tabletop_task and visible_tool_enabled else float(_gripper_contact_proxy_radii(float(args.ee_contact_radius))["gripper_center"]))
        ),
        "contact_proxy_radii_m": (
            {"ground_plane": 0.0}
            if drop_release_task
            else (
                {"visible_tool_capsule": float(meta.get("visible_tool_radius"))}
                if tabletop_task and visible_tool_enabled and actual_tool_clearance
                else _gripper_contact_proxy_radii(float(args.ee_contact_radius))
            )
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
        "proxy_contact_counts": (
            {"ground_plane": int(np.count_nonzero(contact_mask))}
            if drop_release_task
            else (
                {"visible_tool_capsule": int(np.count_nonzero(support_contact_mask))}
                if tabletop_task and visible_tool_enabled and actual_tool_clearance
                else {
                    name: int(sum(1 for src, active in zip(min_clearance_source, support_contact_mask.tolist(), strict=False) if active and src == name))
                    for name in sorted(set(min_clearance_source))
                    if any(active and src == name for src, active in zip(min_clearance_source, support_contact_mask.tolist(), strict=False))
                }
            )
        ),
        "drag_phase_gating": (
            "pre_approach + release_retract"
            if str(args.task) == "lift_release"
            else ("always_on_quasi_static" if tabletop_task else "always_on")
        ),
    }


def build_physics_validation(
    model: newton.Model,
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> dict[str, Any]:
    rigid_only_stage = str(summary.get("blocking_stage", meta.get("blocking_stage", args.blocking_stage))) == "rigid_only"
    if rigid_only_stage:
        return {
            "task": str(args.task),
            "blocking_stage": "rigid_only",
            "rope_present": False,
            "robot_geometry": "native_franka",
            "table_top_z_m": float(meta.get("table_top_z")),
            "summary_contact_proxy_mode": "actual_finger_boxes_vs_table_box",
        }
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


def _load_saved_history(
    history_dir: Path,
    *,
    prefix: str,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> dict[str, Any]:
    history_dir = history_dir.expanduser().resolve()
    if not history_dir.exists():
        raise FileNotFoundError(history_dir)

    def _load_required(name: str) -> np.ndarray:
        path = history_dir / f"{prefix}_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path).astype(np.float32, copy=False)

    particle_q_all = _load_required("particle_q_all")
    particle_q_object = _load_required("particle_q_object")
    body_q = _load_required("body_q")
    body_vel = _load_required("body_vel")
    ee_target_pos = _load_required("ee_target_pos")
    n_frames = int(particle_q_all.shape[0])

    summary_path = history_dir / f"{prefix}_summary.json"
    loaded_summary: dict[str, Any] = {}
    if summary_path.exists():
        loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    zero_series = np.zeros((n_frames,), dtype=np.float32)
    history_files = {
        "particle_q_all": str(history_dir / f"{prefix}_particle_q_all.npy"),
        "particle_q_object": str(history_dir / f"{prefix}_particle_q_object.npy"),
        "body_q": str(history_dir / f"{prefix}_body_q.npy"),
        "body_vel": str(history_dir / f"{prefix}_body_vel.npy"),
        "ee_target_pos": str(history_dir / f"{prefix}_ee_target_pos.npy"),
    }
    return {
        "particle_q_all": particle_q_all,
        "particle_q_object": particle_q_object,
        "body_q": body_q,
        "body_vel": body_vel,
        "ee_target_pos": ee_target_pos,
        "sim_dt": float(args.sim_dt),
        "substeps": int(args.substeps),
        "wall_time": float(loaded_summary.get("wall_time_sec", 0.0)),
        "history_storage_mode": "loaded_history",
        "history_storage_files": history_files,
        "robot_motion_mode": str(loaded_summary.get("robot_motion_mode", meta.get("robot_motion_mode", "ik"))),
        "replay_source": loaded_summary.get("replay_source"),
        "release_frame_actual": loaded_summary.get("release_frame"),
        "settle_gate_metrics": loaded_summary.get("settle_gate_metrics"),
        "settle_gate_pass": loaded_summary.get("settle_gate_pass"),
        "settle_gate_frame": loaded_summary.get("settle_gate_frame"),
        "pre_release_settle_damping_scale": float(
            loaded_summary.get("pre_release_settle_damping_scale", args.pre_release_settle_damping_scale)
        ),
        "preroll_settle_pass": loaded_summary.get("preroll_settle_pass"),
        "preroll_settle_time_s": loaded_summary.get("preroll_settle_time_s"),
        "preroll_settle_metrics": loaded_summary.get("preroll_settle_metrics"),
        "preroll_frame_count": loaded_summary.get("preroll_frame_count"),
        "particle_speed_mean_series": zero_series.copy(),
        "support_patch_speed_mean_series": zero_series.copy(),
        "rope_com_horizontal_speed_series": zero_series.copy(),
        "rope_com_vertical_speed_series": zero_series.copy(),
        "rope_spring_energy_proxy_series": zero_series.copy(),
    }


def _materialize_loaded_history(args: argparse.Namespace, sim_data: dict[str, Any]) -> None:
    history_names = [
        "particle_q_all",
        "particle_q_object",
        "body_q",
        "body_vel",
        "ee_target_pos",
    ]
    files: dict[str, str] = {}
    for name in history_names:
        value = sim_data.get(name)
        if value is None:
            continue
        path = args.out_dir / f"{args.prefix}_{name}.npy"
        np.save(path, np.asarray(value, dtype=np.float32))
        files[name] = str(path)
    sim_data["history_storage_mode"] = "loaded_history_materialized"
    sim_data["history_storage_files"] = files


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
    if args.load_history_from_dir is not None:
        history_prefix = str(args.load_history_prefix or args.prefix)
        sim_data = _load_saved_history(
            Path(args.load_history_from_dir),
            prefix=history_prefix,
            args=args,
            meta=meta,
        )
        _materialize_loaded_history(args, sim_data)
    else:
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
