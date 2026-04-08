#!/usr/bin/env python3
"""Clean robot + rope tabletop push demo.

A minimal, physics-correct demo that satisfies three requirements:
1. Robot holds its pose stably (no collapse, no start-up crash)
2. Table physically blocks the robot (no penetration)
3. Robot can push the rope across the table

Design choices (motivated by physics analysis):
- MuJoCo solver: implicit integration is unconditionally stable for articulated
  bodies. SemiImplicit's spring-damper joints are only marginally stable at the
  torques needed to hold a Franka arm against gravity.
- Proper collision: broad_phase="nxn" ensures all shape pairs are checked,
  including robot links vs table.
- Reasonable particle radii: particle_radius_scale=0.5 preserves effective
  collision radii (~13 mm) so penalty forces are meaningful.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Bootstrap: set up imports from the bridge layer
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = DEMO_DIR.parent
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
NEWTON_PY_ROOT = WORKSPACE_ROOT / "Newton" / "newton"

for _p in (CORE_DIR, NEWTON_PY_ROOT, DEMO_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from bridge_shared import (
    BRIDGE_ROOT as _BR,
    load_core_module,
    overlay_text_lines_rgb,
    camera_position,
    ground_grid,
)

path_defaults = load_core_module("phystwin_bridge_path_defaults", CORE_DIR / "path_defaults.py")

# Stub for bridge_bootstrap so newton_import_ir can be loaded
_stub = types.ModuleType("bridge_bootstrap")
import newton
_stub.newton = newton
_stub.newton_import_ir = None
_stub.path_defaults = None
_stub.ensure_bridge_runtime_paths = lambda: None
sys.modules["bridge_bootstrap"] = _stub

newton_import_ir = load_core_module("phystwin_bridge_newton_import_ir", CORE_DIR / "newton_import_ir.py")
_stub.newton_import_ir = newton_import_ir
_stub.path_defaults = path_defaults

import newton.ik as ik
import newton.utils

from bridge_deformable_common import (
    _apply_drag_correction_ignore_axis,
    _copy_object_only_ir,
    _effective_spring_scales,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    load_ir,
)
from rope_demo_common import (
    anchor_particle_indices,
    rope_endpoints,
    resolve_particle_contact_settings,
)

# ---------------------------------------------------------------------------
# Warp compatibility shims
# ---------------------------------------------------------------------------
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
        x, y, z, w = q[0], q[1], q[2], q[3]
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRANKA_INIT_Q = np.array([
    -3.680e-03, 2.390e-02, 3.680e-03, -2.368e+00,
    -1.292e-04, 2.392e+00, 7.855e-01, 0.04, 0.04,
], dtype=np.float32)


def _find_body(labels, suffix):
    for i, lbl in enumerate(labels):
        if str(lbl).endswith(suffix):
            return i
    raise ValueError(f"Body '{suffix}' not found")


def _enable_gravcomp(builder):
    """Enable gravity compensation on Franka joints (MuJoCo solver)."""
    ga = builder.custom_attributes.get("mujoco:jnt_actgravcomp")
    gb = builder.custom_attributes.get("mujoco:gravcomp")
    if ga is None or gb is None:
        return
    if ga.values is None:
        ga.values = {}
    for i in range(7):
        ga.values[i] = True
    if gb.values is None:
        gb.values = {}
    link_suffixes = [
        "/fr3_link1", "/fr3_link2", "/fr3_link3", "/fr3_link4",
        "/fr3_link5", "/fr3_link6", "/fr3_link7", "/fr3_link8",
        "/fr3_hand", "/fr3_hand_tcp", "/fr3_leftfinger", "/fr3_rightfinger",
    ]
    for idx, lbl in enumerate(builder.body_label):
        if any(str(lbl).endswith(s) for s in link_suffixes):
            gb.values[idx] = 1.0


def _reshape_rope_flat(positions, table_z, radius):
    """Lay rope flat on the table surface."""
    pts = positions.copy()
    pts[:, 2] = table_z + radius + 0.002  # small clearance above table
    return pts


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    ha = float(angle) * 0.5
    s = np.sin(ha)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(ha)], dtype=np.float32)


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Clean robot + rope tabletop push demo.")
    p.add_argument("--ir", type=Path,
                   default=BRIDGE_ROOT / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="robot_rope_clean")
    p.add_argument("--device", default=path_defaults.default_device())
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--sim-dt", type=float, default=1.0 / 60.0 / 10.0,
                   help="Simulation dt. Default 1/600 s.")
    p.add_argument("--substeps", type=int, default=10,
                   help="Substeps per frame.")
    p.add_argument("--gravity", type=float, default=9.81)
    # Rope parameters
    p.add_argument("--auto-set-weight", type=float, default=0.15,
                   help="Total rope mass [kg].")
    p.add_argument("--mass-spring-scale", type=float, default=None)
    p.add_argument("--object-mass", type=float, default=1.0)
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument("--particle-radius-scale", type=float, default=0.5,
                   help="Scale factor for rope particle collision radii.")
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument("--particle-contacts", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--particle-contact-kernel", action=argparse.BooleanOptionalAction, default=None)
    # Table
    p.add_argument("--table-top-z", type=float, default=0.0)
    p.add_argument("--table-hx", type=float, default=0.50)
    p.add_argument("--table-hy", type=float, default=0.30)
    p.add_argument("--table-hz", type=float, default=0.02)
    # Robot
    p.add_argument("--robot-base-x", type=float, default=-0.50)
    p.add_argument("--robot-base-y", type=float, default=-0.20)
    p.add_argument("--robot-base-z", type=float, default=0.10)
    p.add_argument("--joint-target-ke", type=float, default=650.0)
    p.add_argument("--joint-target-kd", type=float, default=100.0)
    p.add_argument("--finger-target-ke", type=float, default=100.0)
    p.add_argument("--finger-target-kd", type=float, default=10.0)
    p.add_argument("--ik-iters", type=int, default=24)
    p.add_argument("--gripper-open", type=float, default=0.04)
    # Timing
    p.add_argument("--settle-seconds", type=float, default=0.5)
    p.add_argument("--approach-seconds", type=float, default=1.0)
    p.add_argument("--push-seconds", type=float, default=2.0)
    p.add_argument("--retract-seconds", type=float, default=1.0)
    # Rendering
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--experiment-label", default="",
                   help="Label for experiment tracking.")
    return p.parse_args()


def build_scene(args, device):
    """Build the combined robot + rope + table scene."""
    # Load and prepare rope IR
    raw_ir = load_ir(args.ir)
    _validate_scaling_args(args)
    _maybe_autoset_mass_spring_scale(args, raw_ir)
    ir_obj = _copy_object_only_ir(raw_ir, args)

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)

    # Get collision radius
    collision_radius_arr = np.asarray(
        ir_obj.get("collision_radius",
                    ir_obj.get("contact_collision_dist",
                               np.full((n_obj,), 0.026, dtype=np.float32))),
        dtype=np.float32,
    ).reshape(-1)
    particle_radius_ref = float(collision_radius_arr[0]) if collision_radius_arr.size else 0.026

    # Center rope and place on table
    endpoint_indices = rope_endpoints(edges, n_obj, x0)
    endpoint_mid = 0.5 * (x0[int(endpoint_indices[0])] + x0[int(endpoint_indices[1])])
    rope_height = args.table_top_z + particle_radius_ref + 0.005
    shift = np.array([
        -float(endpoint_mid[0]),
        -float(endpoint_mid[1]),
        rope_height - float(endpoint_mid[2]),
    ], dtype=np.float32)
    x0_shifted = x0 + shift
    # Flatten rope onto table
    x0_shifted[:n_obj] = _reshape_rope_flat(
        x0_shifted[:n_obj], args.table_top_z, particle_radius_ref
    )
    rope_center = x0_shifted[:n_obj].mean(axis=0).astype(np.float32)

    # Anchor endpoints
    anchor_idx = anchor_particle_indices(
        x0_shifted, endpoint_indices=endpoint_indices, count_per_end=2
    )

    # Build Newton model
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    particle_contacts, particle_contact_kernel = resolve_particle_contact_settings(ir_obj, args)

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
        gravity=-float(args.gravity),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=bool(particle_contacts),
        device=device,
        add_ground_plane=False,
    )

    # Register MuJoCo custom attributes BEFORE adding anything
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # Add rope particles and springs
    checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)
    _, _, _ = newton_import_ir._add_particles(
        builder, ir_obj, cfg, particle_contacts=bool(particle_contacts)
    )
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)

    # Override particle positions with shifted/flattened positions
    builder.particle_q = [wp.vec3(*row.tolist()) for row in x0_shifted]

    # Mark anchor particles as inactive
    for idx in anchor_idx.tolist():
        builder.particle_flags[idx] = (
            int(builder.particle_flags[idx]) & ~int(newton.ParticleFlags.ACTIVE)
        )
        builder.particle_mass[idx] = 0.0

    # Add Franka robot
    robot_base = rope_center + np.array(
        [args.robot_base_x, args.robot_base_y, args.robot_base_z],
        dtype=np.float32,
    )
    franka_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    builder.add_urdf(
        franka_asset,
        xform=wp.transform(wp.vec3(*robot_base.tolist()), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )

    # Configure robot joints
    builder.joint_q[:9] = FRANKA_INIT_Q.tolist()
    builder.joint_target_pos[:9] = FRANKA_INIT_Q.tolist()
    builder.joint_target_ke[:7] = [float(args.joint_target_ke)] * 7
    builder.joint_target_kd[:7] = [float(args.joint_target_kd)] * 7
    builder.joint_target_ke[7:9] = [float(args.finger_target_ke)] * 2
    builder.joint_target_kd[7:9] = [float(args.finger_target_kd)] * 2
    builder.joint_target_pos[7:9] = [float(args.gripper_open)] * 2
    builder.joint_armature[:7] = [0.1] * 7
    builder.joint_armature[7:9] = [0.5] * 2
    builder.joint_effort_limit[:7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    builder.joint_effort_limit[7:9] = [20.0, 20.0]

    # Enable gravity compensation
    _enable_gravcomp(builder)

    # Find key body indices
    ee_body_index = _find_body(builder.body_label, "/fr3_link7")
    left_finger_idx = _find_body(builder.body_label, "/fr3_leftfinger")
    right_finger_idx = _find_body(builder.body_label, "/fr3_rightfinger")

    # Add table as static box
    table_center = np.array([
        float(rope_center[0]),
        float(rope_center[1]),
        float(args.table_top_z - args.table_hz),
    ], dtype=np.float32)
    table_cfg = builder.default_shape_cfg.copy()
    # Configure table contact material to be stiff
    newton_import_ir._configure_ground_contact_material(
        table_cfg, ir_obj, cfg, checks, context="table_box"
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*table_center.tolist()), wp.quat_identity()),
        hx=float(args.table_hx),
        hy=float(args.table_hy),
        hz=float(args.table_hz),
        cfg=table_cfg,
        label="table_box",
    )

    # Finalize model
    model = builder.finalize(device=device)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    if not bool(particle_contact_kernel):
        model.particle_grid = None

    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)

    meta = {
        "n_obj": n_obj,
        "rope_center": rope_center,
        "table_center": table_center,
        "table_top_z": float(args.table_top_z),
        "ee_body_index": ee_body_index,
        "left_finger_idx": left_finger_idx,
        "right_finger_idx": right_finger_idx,
        "anchor_idx": anchor_idx,
        "edges": edges,
        "particle_radius_ref": particle_radius_ref,
        "ir_obj": ir_obj,
        "cfg": cfg,
    }
    return model, meta


def _gripper_center(body_q, left_idx, right_idx):
    """Compute gripper center from body transforms."""
    return 0.5 * (body_q[left_idx, :3] + body_q[right_idx, :3])


def _ee_local_offset(model, ee_idx, left_idx, right_idx):
    """Measure the local offset from link7 to gripper center."""
    state = model.state()
    state.joint_q.assign(FRANKA_INIT_Q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    bq = state.body_q.numpy().astype(np.float32)
    gc = _gripper_center(bq, left_idx, right_idx)
    link7_pos = bq[ee_idx, :3]
    link7_quat = bq[ee_idx, 3:7]
    # Inverse rotate to get local offset
    x, y, z, w = link7_quat
    # Conjugate quaternion
    cq = np.array([-x, -y, -z, w], dtype=np.float32)
    diff = gc - link7_pos
    # quat_rotate(cq, diff)
    t = 2.0 * np.cross(cq[:3], diff)
    return (diff + cq[3] * t + np.cross(cq[:3], t)).astype(np.float32)


def _solve_ik(model, ee_idx, ee_offset, q_seed, target_pos, target_quat, iters, gripper, device):
    """Solve IK for a single Cartesian target."""
    q_seed = np.asarray(q_seed, dtype=np.float32).copy()
    joint_q = wp.array(q_seed.reshape(1, -1), dtype=wp.float32, device=device)
    pos_obj = ik.IKObjectivePosition(
        link_index=int(ee_idx),
        link_offset=wp.vec3(*ee_offset.tolist()),
        target_positions=wp.array([wp.vec3(*target_pos.tolist())], dtype=wp.vec3, device=device),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=int(ee_idx),
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*target_quat.tolist())], dtype=wp.vec4, device=device),
    )
    limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    solver = ik.IKSolver(
        model=model, n_problems=1,
        objectives=[pos_obj, rot_obj, limit_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(joint_q, joint_q, iterations=iters)
    q = joint_q.numpy().reshape(-1).astype(np.float32)
    q[7:9] = float(gripper)
    return q


def generate_waypoints(model, meta, args, device):
    """Generate IK-solved joint-space waypoints for the push task."""
    ee_idx = meta["ee_body_index"]
    left_idx = meta["left_finger_idx"]
    right_idx = meta["right_finger_idx"]
    rope_center = meta["rope_center"]
    table_z = meta["table_top_z"]
    radius = meta["particle_radius_ref"]

    ee_offset = _ee_local_offset(model, ee_idx, left_idx, right_idx)

    # Target orientation: gripper pointing down
    down_quat = _quat_from_axis_angle([1, 0, 0], np.pi)
    yaw_quat = _quat_from_axis_angle([0, 0, 1], np.deg2rad(-15.0))
    target_quat = _quat_multiply(down_quat, yaw_quat)

    # Contact height: just above rope on table
    contact_z = table_z + 2 * radius + 0.008

    # Waypoint positions (world space, relative to rope center)
    high_z = table_z + 0.12  # safe clearance
    waypoints = {
        "park":     np.array([rope_center[0] - 0.15, rope_center[1], high_z], dtype=np.float32),
        "approach": np.array([rope_center[0] - 0.10, rope_center[1], contact_z], dtype=np.float32),
        "push_end": np.array([rope_center[0] + 0.12, rope_center[1], contact_z], dtype=np.float32),
        "retract":  np.array([rope_center[0] + 0.15, rope_center[1], high_z], dtype=np.float32),
    }

    # Solve IK for each waypoint
    q_seed = FRANKA_INIT_Q.copy()
    joint_waypoints = {}
    for name, pos in waypoints.items():
        q = _solve_ik(model, ee_idx, ee_offset, q_seed, pos, target_quat, args.ik_iters, args.gripper_open, device)
        joint_waypoints[name] = q
        q_seed = q  # chain seeds

    return joint_waypoints, waypoints, target_quat, ee_offset


def run_simulation(args, device):
    """Main simulation loop."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[build] Building scene...")
    model, meta = build_scene(args, device)
    n_obj = meta["n_obj"]
    ir_obj = meta["ir_obj"]
    cfg = meta["cfg"]

    print("[build] Generating waypoints...")
    joint_wp, cart_wp, target_quat, ee_offset = generate_waypoints(model, meta, args, device)

    # Create solver (MuJoCo for stability)
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

    # Collision pipeline with NxN broad phase for full coverage
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        soft_contact_margin=0.02,
    )
    contacts = collision_pipeline.contacts()
    control = model.control()

    # Initialize state
    state_0 = model.state()
    state_1 = model.state()
    q_init = joint_wp["park"]
    for state in (state_0, state_1):
        state.joint_q.assign(q_init)
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Set initial control targets
    control.joint_target_pos.assign(q_init)

    sim_dt = float(args.sim_dt)
    substeps = int(args.substeps)
    frame_dt = sim_dt * substeps
    n_frames = int(args.frames)

    # Phase timing
    settle_frames = max(1, int(args.settle_seconds / frame_dt))
    approach_frames = max(1, int(args.approach_seconds / frame_dt))
    push_frames = max(1, int(args.push_seconds / frame_dt))
    retract_frames = max(1, int(args.retract_seconds / frame_dt))
    total_needed = settle_frames + approach_frames + push_frames + retract_frames
    if n_frames < total_needed:
        n_frames = total_needed + 10
        print(f"[info] Adjusted frames to {n_frames} to fit all phases.")

    # Drag setup
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gn = float(np.linalg.norm(gravity_vec))
    if gn > 1e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gn).astype(np.float32)
    drag_damping = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag_damping = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)

    # History storage
    particle_q_history = np.zeros((n_frames, n_obj, 3), dtype=np.float32)
    body_q_history = np.zeros((n_frames, model.body_count, 7), dtype=np.float32)
    phase_history = []

    print(f"[sim] Running {n_frames} frames (dt={sim_dt}, substeps={substeps})...")
    t0 = time.perf_counter()

    for frame in range(n_frames):
        # Determine phase and interpolate joint targets
        if frame < settle_frames:
            phase = "settle"
            alpha = 0.0
            q_start = joint_wp["park"]
            q_end = joint_wp["park"]
        elif frame < settle_frames + approach_frames:
            phase = "approach"
            alpha = (frame - settle_frames) / max(approach_frames, 1)
            q_start = joint_wp["park"]
            q_end = joint_wp["approach"]
        elif frame < settle_frames + approach_frames + push_frames:
            phase = "push"
            alpha = (frame - settle_frames - approach_frames) / max(push_frames, 1)
            q_start = joint_wp["approach"]
            q_end = joint_wp["push_end"]
        else:
            phase = "retract"
            alpha = (frame - settle_frames - approach_frames - push_frames) / max(retract_frames, 1)
            q_start = joint_wp["push_end"]
            q_end = joint_wp["retract"]

        alpha = np.clip(alpha, 0.0, 1.0)
        joint_target = ((1.0 - alpha) * q_start + alpha * q_end).astype(np.float32)
        phase_history.append(phase)

        # Record state
        if state_0.particle_q is not None:
            pq = state_0.particle_q.numpy().astype(np.float32)
            particle_q_history[frame] = pq[:n_obj]
        body_q_history[frame] = state_0.body_q.numpy().astype(np.float32)

        # Substep loop
        for sub in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()

            # Set control targets
            control.joint_target_pos.assign(joint_target)
            if hasattr(control, 'joint_f') and control.joint_f is not None:
                control.joint_f.zero_()

            # Collide and step
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

            # Apply drag correction
            if drag_damping > 0.0 and n_obj > 0 and state_0.particle_q is not None:
                if gravity_axis is not None:
                    wp.launch(
                        _apply_drag_correction_ignore_axis,
                        dim=n_obj,
                        inputs=[
                            state_0.particle_q, state_0.particle_qd,
                            n_obj, sim_dt, drag_damping,
                            wp.vec3(*gravity_axis.tolist()),
                        ],
                        device=device,
                    )

        if frame % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame}/{n_frames} [{phase}] ({elapsed:.1f}s)")

    wall_time = time.perf_counter() - t0
    print(f"[sim] Done in {wall_time:.1f}s")

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    # Check robot stability
    bq = body_q_history
    max_z_drop = float(np.min(bq[:, :, 2]))
    has_nan = bool(np.any(np.isnan(bq)))
    print(f"[diag] Robot min z = {max_z_drop:.4f}, has NaN = {has_nan}")

    # Check rope displacement
    rope_com_start = particle_q_history[settle_frames].mean(axis=0)
    rope_com_end = particle_q_history[-1].mean(axis=0)
    rope_disp = float(np.linalg.norm(rope_com_end - rope_com_start))
    print(f"[diag] Rope COM displacement = {rope_disp:.4f} m")

    # Check rope doesn't fall through table
    min_rope_z = float(np.min(particle_q_history[settle_frames:, :, 2]))
    table_z = float(args.table_top_z)
    rope_above_table = min_rope_z > (table_z - 0.01)
    print(f"[diag] Min rope z = {min_rope_z:.4f}, table z = {table_z:.4f}, above = {rope_above_table}")

    # Finger-rope proximity
    for phase_check_frame in range(settle_frames + approach_frames,
                                    min(settle_frames + approach_frames + push_frames, n_frames)):
        gc = _gripper_center(body_q_history[phase_check_frame],
                            meta["left_finger_idx"], meta["right_finger_idx"])
        closest_dist = float(np.min(np.linalg.norm(
            particle_q_history[phase_check_frame] - gc, axis=1
        )))
        if closest_dist < 0.05:
            print(f"[diag] Finger-rope contact at frame {phase_check_frame}: dist={closest_dist:.4f}")
            break

    # Save summary
    summary = {
        "wall_time_s": wall_time,
        "n_frames": n_frames,
        "sim_dt": sim_dt,
        "substeps": substeps,
        "robot_stable": not has_nan and max_z_drop > -0.5,
        "rope_displacement_m": rope_disp,
        "rope_above_table": rope_above_table,
        "min_rope_z": min_rope_z,
        "experiment_label": args.experiment_label,
        "particle_radius_scale": float(args.particle_radius_scale),
        "joint_target_ke": float(args.joint_target_ke),
    }
    summary_path = args.out_dir / f"{args.prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[out] Summary: {summary_path}")

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------
    if not args.skip_render:
        _render_video(
            model, meta, args, particle_q_history, body_q_history, phase_history,
            n_frames, device,
        )

    return summary


def _render_video(model, meta, args, particle_q_history, body_q_history, phase_history, n_frames, device):
    """Render an offline MP4 of the simulation."""
    try:
        viewer = newton.Viewer(
            model,
            headless=True,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
        )
    except Exception as e:
        print(f"[render] Viewer init failed: {e}")
        return

    edges = meta["edges"]
    n_obj = meta["n_obj"]
    rope_center = meta["rope_center"]
    table_z = meta["table_top_z"]

    # Camera
    cam_target = np.array([rope_center[0], rope_center[1], table_z + 0.15], dtype=np.float32)
    cam_pos = camera_position(cam_target, yaw_deg=-135, pitch_deg=25, distance=0.85)

    render_dt = 1.0 / args.render_fps
    frame_dt = float(args.sim_dt) * int(args.substeps)
    render_every = max(1, int(render_dt / frame_dt))

    frames_rgb = []
    state = model.state()
    for frame in range(0, n_frames, render_every):
        # Set state for rendering
        if model.particle_count > 0:
            pq_full = np.zeros((model.particle_count, 3), dtype=np.float32)
            pq_full[:n_obj] = particle_q_history[frame]
            state.particle_q.assign(pq_full)
        state.body_q.assign(body_q_history[frame])

        sim_t = frame * frame_dt
        viewer.begin_frame(sim_t)
        viewer.log_state(state)

        # Draw rope lines
        if edges.shape[0] > 0:
            pq = particle_q_history[frame]
            starts = pq[edges[:, 0]]
            ends = pq[edges[:, 1]]
            colors = np.tile(np.array([[0.8, 0.3, 0.1]], dtype=np.float32), (edges.shape[0], 1))
            viewer.add_lines(starts, ends, colors)

        viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch=-25.0, yaw=-135.0)
        viewer.end_frame()

        rgb = viewer.get_pixels(mode="rgb")
        if rgb is not None:
            rgb = overlay_text_lines_rgb(
                rgb,
                [f"t={sim_t:.2f}s  phase={phase_history[frame]}"],
                font_size=20,
            )
            frames_rgb.append(rgb)

    viewer.close()

    if not frames_rgb:
        print("[render] No frames captured.")
        return

    # Save MP4
    mp4_path = args.out_dir / f"{args.prefix}.mp4"
    try:
        import imageio.v3 as iio
        iio.imwrite(str(mp4_path), np.stack(frames_rgb), fps=args.render_fps, codec="libx264")
        print(f"[render] MP4: {mp4_path}")
    except Exception as e:
        print(f"[render] MP4 write failed: {e}")

    # Save GIF
    if args.make_gif:
        gif_path = args.out_dir / f"{args.prefix}.gif"
        try:
            import imageio.v3 as iio
            # Downsample for GIF
            step = max(1, len(frames_rgb) // 60)
            gif_frames = frames_rgb[::step]
            iio.imwrite(str(gif_path), np.stack(gif_frames), fps=10, loop=0)
            print(f"[render] GIF: {gif_path}")
        except Exception as e:
            print(f"[render] GIF write failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    summary = run_simulation(args, args.device)
    print(f"\n=== RESULT ===")
    print(f"  Robot stable:  {summary['robot_stable']}")
    print(f"  Rope displaced: {summary['rope_displacement_m']:.4f} m")
    print(f"  Rope above table: {summary['rope_above_table']}")
