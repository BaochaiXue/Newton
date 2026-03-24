#!/usr/bin/env python3
"""Object-only cloth drop onto a rigid box using Newton SemiImplicit.

This script is intentionally ON-only:

- SELF COLLISION: ON

Unlike the previous VBD-based control experiment, this ON path now stays on the
same bridge semantics as the OFF cloth-bunny demo:

- load the PhysTwin strict IR directly
- drop controller particles and keep only the object spring-mass cloth
- run Newton :class:`newton.solvers.SolverSemiImplicit`
- enable Newton's particle-particle contact kernel

This preserves the core question we actually care about: what happens if we try
to make ON work on the same spring-mass representation that OFF already uses.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from demo_cloth_bunny_common import (
    _apply_particle_contact_scaling,
    _apply_shape_contact_scaling,
    _box_signed_distance,
    _copy_object_only_ir,
    _default_cloth_ir,
    _effective_spring_scales,
    _mass_tag,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    load_ir,
)
from demo_rope_bunny_drop import (
    _apply_drag_correction_ignore_axis,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
)
from demo_shared import compute_visual_particle_radii, temporary_particle_radius_override
from newton._src.solvers.semi_implicit.kernels_body import eval_body_joint_forces
from newton._src.solvers.semi_implicit.kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from newton._src.solvers.semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


DEFAULT_TARGET_TOTAL_MASS = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-only cloth drops onto a rigid box.")
    p.add_argument("--ir", type=Path, default=_default_cloth_ir(), help="Path to PhysTwin cloth IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="cloth_box_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--sim-dt", type=float, default=1.0 / 600.0)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--drop-height", type=float, default=0.5, help="Cloth bottom height above box top, in meters.")
    p.add_argument("--object-mass", type=float, default=None, help="Optional per-particle cloth object mass override.")
    p.add_argument(
        "--target-total-mass",
        type=float,
        default=DEFAULT_TARGET_TOTAL_MASS,
        help=(
            "If neither --object-mass nor --mass-spring-scale is provided, "
            "auto-compute a stable cloth mass scale so the total object mass "
            "matches this value."
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

    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)
    p.add_argument(
        "--drag-ignore-gravity-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply drag only orthogonal to gravity so free-fall acceleration is preserved.",
    )
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)
    p.add_argument(
        "--shape-contact-scale",
        type=float,
        default=None,
        help=(
            "Scale factor applied to the actual particle-shape contact stiffness chain used by "
            "SemiImplicit: soft_contact_ke and shape_material_ke. "
            "If omitted, particle-shape contact stays at baseline values."
        ),
    )
    p.add_argument(
        "--shape-contact-damping-multiplier",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier applied on top of shape-contact-scale for kd/kf terms in the "
            "actual particle-shape contact chain. This is useful when minimizing rollout RMSE "
            "requires more damping than pure alpha-scaling."
        ),
    )
    p.add_argument(
        "--ground-shape-contact-scale",
        type=float,
        default=None,
        help=(
            "Optional override for ground shape_material_ke scaling. "
            "If omitted, ground uses --shape-contact-scale."
        ),
    )
    p.add_argument(
        "--ground-shape-contact-damping-multiplier",
        type=float,
        default=None,
        help=(
            "Optional override for ground shape_material_kd/kf scaling multiplier. "
            "If omitted, ground uses --shape-contact-damping-multiplier."
        ),
    )
    p.add_argument(
        "--box-shape-contact-scale",
        type=float,
        default=None,
        help=(
            "Optional override for box shape_material_ke scaling. "
            "If omitted, the box uses --shape-contact-scale."
        ),
    )
    p.add_argument(
        "--box-shape-contact-damping-multiplier",
        type=float,
        default=None,
        help=(
            "Optional override for box shape_material_kd/kf scaling multiplier. "
            "If omitted, the box uses --shape-contact-damping-multiplier."
        ),
    )
    p.add_argument("--disable-particle-contact-kernel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--contact-dist-scale",
        type=float,
        default=1.0,
        help="Global scale factor applied to contact_collision_dist before Newton radius mapping.",
    )
    p.add_argument(
        "--on-contact-dist-scale",
        type=float,
        default=1.0,
        help="Scale factor applied only to ON-case contact_collision_dist before Newton particle-contact radius mapping.",
    )
    p.add_argument(
        "--add-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the rigid box in the scene.",
    )
    p.add_argument(
        "--add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the z=0 ground plane in the scene.",
    )
    p.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable particle-vs-shape contacts for the rigid box.",
    )

    p.add_argument("--rigid-mass", type=float, default=4000.0)
    p.add_argument("--body-mu", type=float, default=0.5)
    p.add_argument("--body-ke", type=float, default=1.0e4)
    p.add_argument("--body-kd", type=float, default=1.0e-2)
    p.add_argument(
        "--dynamic-box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a dynamic rigid box body. Disabled by default because the ON "
            "self-collision demo is meant to isolate stable cloth self-contact "
            "against a fixed support shape."
        ),
    )
    p.add_argument("--box-hx", type=float, default=0.12)
    p.add_argument("--box-hy", type=float, default=0.12)
    p.add_argument("--box-hz", type=float, default=0.08)

    # Compatibility-only knobs retained so older CLI invocations keep working.
    p.add_argument("--vbd-iterations", type=int, default=15)
    p.add_argument("--vbd-tri-ke", type=float, default=1.0e4)
    p.add_argument("--vbd-tri-ka", type=float, default=1.0e4)
    p.add_argument("--vbd-tri-kd", type=float, default=1.0e-1)
    p.add_argument("--vbd-edge-ke", type=float, default=1.0e4)
    p.add_argument("--vbd-edge-kd", type=float, default=1.0e0)
    p.add_argument("--self-contact-radius-scale", type=float, default=0.35)
    p.add_argument("--self-contact-margin-scale", type=float, default=0.60)
    p.add_argument("--soft-contact-margin-scale", type=float, default=1.0)

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)

    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=2.0)
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-pos", type=float, nargs=3, default=(-1.55, 1.35, 1.18), metavar=("X", "Y", "Z"))
    p.add_argument("--camera-pitch", type=float, default=-10.0)
    p.add_argument("--camera-yaw", type=float, default=-40.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    p.add_argument(
        "--particle-radius-vis-min",
        type=float,
        default=0.005,
        help="Visualization-only radius cap: render_radius = min(physical_radius * scale, this value).",
    )
    p.add_argument("--render-springs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--spring-stride", type=int, default=8)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--decouple-shape-materials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: use baseline shape_material_ke/kd/kf for rigid-rigid contacts, "
            "but scaled shape_material_ke/kd/kf for particle-shape contacts. "
            "This avoids softening box-ground support when low-mass cloth contact is rescaled."
        ),
    )
    p.add_argument(
        "--particle-self-contact-scale",
        type=float,
        default=1.0,
        help=(
            "Compatibility knob for older ON experiments. The current SemiImplicit ON path "
            "does not use it directly because particle contact coefficients are loaded from "
            "the PhysTwin bridge mapping, but we keep it in the summary for bookkeeping."
        ),
    )
    return p.parse_args()


def _box_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "add_box", getattr(args, "add_bunny", True)))


def _shape_scaling_args(args: argparse.Namespace) -> argparse.Namespace:
    alias_args = copy.copy(args)
    alias_args.add_bunny = _box_enabled(args)
    alias_args.bunny_shape_contact_scale = getattr(
        args,
        "box_shape_contact_scale",
        getattr(args, "bunny_shape_contact_scale", None),
    )
    alias_args.bunny_shape_contact_damping_multiplier = getattr(
        args,
        "box_shape_contact_damping_multiplier",
        getattr(args, "bunny_shape_contact_damping_multiplier", None),
    )
    return alias_args


def _assign_shape_material_triplet(model, ke: np.ndarray, kd: np.ndarray, kf: np.ndarray) -> None:
    model.shape_material_ke.assign(np.asarray(ke, dtype=np.float32))
    model.shape_material_kd.assign(np.asarray(kd, dtype=np.float32))
    model.shape_material_kf.assign(np.asarray(kf, dtype=np.float32))


def build_model(ir_obj: dict[str, Any], args: argparse.Namespace, device: str):
    box_enabled = _box_enabled(args)
    shape_contacts_enabled = bool(args.shape_contacts) and box_enabled
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(spring_ke_scale),
        spring_kd_scale=float(spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=False,
        shape_contacts=shape_contacts_enabled,
        add_ground_plane=bool(args.add_ground_plane),
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=True,
        device=device,
    )

    checks = newton_import_ir.validate_ir_physics(ir_obj, cfg)

    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
    radius, _, _ = newton_import_ir._add_particles(builder, ir_obj, cfg, particle_contacts=True)
    newton_import_ir._add_springs(builder, ir_obj, cfg, checks)
    newton_import_ir._add_ground_plane(builder, ir_obj, cfg, checks)

    box_half_extents = np.array(
        [float(args.box_hx), float(args.box_hy), float(args.box_hz)],
        dtype=np.float32,
    )
    box_pos = np.array([0.0, 0.0, float(args.box_hz)], dtype=np.float32)
    box_top_z = float(args.box_hz * 2.0) if box_enabled else 0.0

    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    cloth_center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    reference_top_z = float(box_top_z) if box_enabled else 0.0
    shift = np.array(
        [
            -float(cloth_center_xy[0]),
            -float(cloth_center_xy[1]),
            reference_top_z + float(args.drop_height) - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]

    if box_enabled:
        rigid_cfg = builder.default_shape_cfg.copy()
        rigid_cfg.mu = float(args.body_mu)
        rigid_cfg.ke = float(args.body_ke)
        rigid_cfg.kd = float(args.body_kd)
        if bool(args.dynamic_box):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*box_pos.tolist()), wp.quat_identity()),
                mass=float(args.rigid_mass),
                inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
                lock_inertia=True,
                label="box",
            )
            builder.add_shape_box(
                body=body,
                hx=float(args.box_hx),
                hy=float(args.box_hy),
                hz=float(args.box_hz),
                cfg=rigid_cfg,
            )
        else:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*box_pos.tolist()), wp.quat_identity()),
                hx=float(args.box_hx),
                hy=float(args.box_hy),
                hz=float(args.box_hz),
                cfg=rigid_cfg,
                label="box",
            )

    model = builder.finalize(device=device)
    shape_material_ke_base = model.shape_material_ke.numpy().astype(np.float32, copy=False).copy()
    shape_material_kd_base = model.shape_material_kd.numpy().astype(np.float32, copy=False).copy()
    shape_material_kf_base = model.shape_material_kf.numpy().astype(np.float32, copy=False).copy()

    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    _apply_particle_contact_scaling(
        model, weight_scale=float(ir_obj.get("weight_scale", 1.0))
    )
    _apply_shape_contact_scaling(
        model,
        _shape_scaling_args(args),
        weight_scale=float(ir_obj.get("weight_scale", 1.0)),
    )
    shape_material_ke_scaled = model.shape_material_ke.numpy().astype(np.float32, copy=False).copy()
    shape_material_kd_scaled = model.shape_material_kd.numpy().astype(np.float32, copy=False).copy()
    shape_material_kf_scaled = model.shape_material_kf.numpy().astype(np.float32, copy=False).copy()

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "solver_name": "semiimplicit",
        "rigid_shape": "box",
        "box_half_extents": box_half_extents,
        "box_pos": box_pos.astype(np.float32, copy=False),
        "box_top_z": float(box_top_z),
        "box_is_dynamic": bool(args.dynamic_box),
        "has_box": box_enabled,
        "has_ground_plane": bool(args.add_ground_plane),
        "gravity_vec": np.asarray(gravity_vec, dtype=np.float32),
        "cloth_shift": shift.astype(np.float32, copy=False),
        "render_edges": render_edges,
        "particle_contacts_enabled": True,
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "shape_material_ke_base": shape_material_ke_base,
        "shape_material_kd_base": shape_material_kd_base,
        "shape_material_kf_base": shape_material_kf_base,
        "shape_material_ke_scaled": shape_material_ke_scaled,
        "shape_material_kd_scaled": shape_material_kd_scaled,
        "shape_material_kf_scaled": shape_material_kf_scaled,
    }
    return model, meta, n_obj


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    shape_contacts_enabled = bool(args.shape_contacts) and _box_enabled(args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=False,
        shape_contacts=shape_contacts_enabled,
        add_ground_plane=bool(args.add_ground_plane),
        strict_physics_checks=False,
        apply_drag=bool(args.apply_drag),
        drag_damping_scale=float(args.drag_damping_scale),
        gravity=-float(args.gravity_mag),
        gravity_from_reverse_z=False,
        up_axis="Z",
        particle_contacts=True,
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

    sim_dt = float(args.sim_dt)
    substeps = max(1, int(args.substeps))

    particle_grid = model.particle_grid
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))

    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)
    use_decoupled_shape_materials = bool(args.decouple_shape_materials) or (
        not np.isclose(float(ir_obj.get("weight_scale", 1.0)), 1.0)
    )

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
        if state_in.body_q is None:
            body_q.append(np.zeros((0, 7), dtype=np.float32))
        else:
            body_q.append(state_in.body_q.numpy().astype(np.float32).copy())
        if state_in.body_qd is None:
            body_vel.append(np.zeros((0, 3), dtype=np.float32))
        else:
            body_vel.append(state_in.body_qd.numpy().astype(np.float32)[:, :3].copy())

        for _ in range(substeps):
            state_in.clear_forces()

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if contacts is not None:
                model.collide(state_in, contacts)

            if not use_decoupled_shape_materials:
                solver.step(state_in, state_out, control, contacts, sim_dt)
            else:
                particle_f = state_in.particle_f if state_in.particle_count else None
                body_f = state_in.body_f if state_in.body_count else None
                body_f_work = body_f
                if body_f is not None and model.joint_count and control.joint_f is not None:
                    body_f_work = wp.clone(body_f)

                eval_spring_forces(model, state_in, particle_f)
                eval_triangle_forces(model, state_in, control, particle_f)
                eval_bending_forces(model, state_in, particle_f)
                eval_tetrahedra_forces(model, state_in, control, particle_f)
                eval_body_joint_forces(
                    model, state_in, control, body_f_work, solver.joint_attach_ke, solver.joint_attach_kd
                )
                eval_particle_contact_forces(model, state_in, particle_f)
                if solver.enable_tri_contact:
                    eval_triangle_contact_forces(model, state_in, particle_f)

                _assign_shape_material_triplet(
                    model,
                    meta["shape_material_ke_base"],
                    meta["shape_material_kd_base"],
                    meta["shape_material_kf_base"],
                )
                eval_body_contact_forces(
                    model,
                    state_in,
                    contacts,
                    friction_smoothing=solver.friction_smoothing,
                    body_f_out=body_f_work,
                )

                _assign_shape_material_triplet(
                    model,
                    meta["shape_material_ke_scaled"],
                    meta["shape_material_kd_scaled"],
                    meta["shape_material_kf_scaled"],
                )
                eval_particle_body_contact_forces(
                    model, state_in, contacts, particle_f, body_f_work, body_f_in_world_frame=False
                )

                solver.integrate_particles(model, state_in, state_out, sim_dt)
                if body_f_work is body_f:
                    solver.integrate_bodies(model, state_in, state_out, sim_dt, solver.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_work
                    solver.integrate_bodies(model, state_in, state_out, sim_dt, solver.angular_damping)
                    state_in.body_f = body_f_prev
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

        if frame == 49 or frame == 99 or frame == 199 or frame == n_frames - 1:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    print(f"  Simulation done: {n_frames} frames in {wall_time:.1f}s", flush=True)
    return {
        "particle_q_all": np.stack(particle_q_all, axis=0),
        "particle_q_object": np.stack(particle_q_object, axis=0),
        "body_q": np.stack(body_q, axis=0),
        "body_vel": np.stack(body_vel, axis=0),
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
    out_mp4: Path,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    import newton.viewer  # noqa: PLC0415

    width = int(args.screen_width)
    height = int(args.screen_height)
    fps_out = float(args.render_fps)
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
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]

    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=bool(args.viewer_headless))
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
        cam_pos = np.asarray(args.camera_pos, dtype=np.float32)
        viewer.set_camera(
            wp.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            float(args.camera_pitch),
            float(args.camera_yaw),
        )

        render_radii = compute_visual_particle_radii(
            model.particle_radius.numpy(),
            radius_scale=float(args.particle_radius_vis_scale),
            radius_cap=float(args.particle_radius_vis_min),
        )

        try:
            shape_colors = {}
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "box" in name:
                    shape_colors[idx] = (0.88, 0.35, 0.28)
                elif "ground" in name or "plane" in name:
                    shape_colors[idx] = (0.23, 0.26, 0.31)
            if shape_colors:
                viewer.update_shape_colors(shape_colors)
        except Exception:
            pass

        cloth_edges = np.asarray(meta["render_edges"], dtype=np.int32)
        starts_wp = wp.empty(len(cloth_edges), dtype=wp.vec3, device=device) if cloth_edges.size else None
        ends_wp = wp.empty(len(cloth_edges), dtype=wp.vec3, device=device) if cloth_edges.size else None

        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
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

            mass_label = (
                f"CLOTH TOTAL MASS: {float(meta.get('total_object_mass', 0.0)):.3g} kg"
                f" | BOX MASS: {float(args.rigid_mass):.3g} kg"
            )
            for out_idx, sim_idx in enumerate(render_indices):
                state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
                if state.body_q is not None and sim_data["body_q"].shape[1] > 0:
                    state.body_q.assign(sim_data["body_q"][sim_idx].astype(np.float32, copy=False))

                sim_t = float(sim_idx) * sim_frame_dt
                viewer.begin_frame(sim_t)
                viewer.log_state(state)

                if args.render_springs and cloth_edges.size and starts_wp is not None and ends_wp is not None:
                    q_obj = sim_data["particle_q_object"][sim_idx]
                    starts_wp.assign(q_obj[cloth_edges[:, 0]].astype(np.float32, copy=False))
                    ends_wp.assign(q_obj[cloth_edges[:, 1]].astype(np.float32, copy=False))
                    viewer.log_lines(
                        "/demo/cloth_springs",
                        starts_wp,
                        ends_wp,
                        (0.28, 0.54, 0.88),
                        width=0.01,
                        hidden=False,
                    )

                viewer.end_frame()
                frame = viewer.get_frame(render_ui=False).numpy()
                if args.overlay_label:
                    frame = overlay_text_lines_rgb(
                        frame,
                        [
                            "CLOTH BOX DROP",
                            "SELF COLLISION: ON",
                            mass_label,
                            f"frame {out_idx + 1:03d}/{n_out_frames:03d}  t={sim_t:.3f}s",
                        ],
                        font_size=int(args.label_font_size),
                    )
                assert ffmpeg_proc.stdin is not None
                ffmpeg_proc.stdin.write(frame.tobytes())

            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.close()
            if ffmpeg_proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
    finally:
        try:
            viewer.close()
        except Exception:
            pass
    print(f"  Video saved: {out_mp4}", flush=True)
    return out_mp4


def save_scene_npz(args: argparse.Namespace, sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_npz = args.out_dir / f"{args.prefix}_{_mass_tag(args.rigid_mass)}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        particle_q_object=sim_data["particle_q_object"],
        body_q=sim_data["body_q"],
        body_vel=sim_data["body_vel"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        rigid_box_half_extents=np.asarray(meta["box_half_extents"], dtype=np.float32),
        rigid_mass=np.float32(args.rigid_mass),
        drop_height=np.float32(args.drop_height),
        num_object_points=np.int32(n_obj),
    )
    return scene_npz


def build_summary(
    model,
    args: argparse.Namespace,
    ir_obj: dict[str, Any],
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
    out_mp4: Path | None,
    particle_contacts_enabled: bool,
    disable_particle_contact_kernel: bool,
) -> dict[str, Any]:
    use_decoupled_shape_materials = bool(args.decouple_shape_materials)
    cloth_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    if body_q.ndim == 3 and body_q.shape[1] > 0 and bool(meta.get("has_box", False)):
        body_speed = np.linalg.norm(sim_data["body_vel"][:, 0, :], axis=1)
        body_z = body_q[:, 0, 2]
        box_top_offset = float(meta["box_top_z"]) - float(body_z[0])
        dynamic_box_top = body_z + box_top_offset
        box_positions = body_q[:, 0, :3].astype(np.float32, copy=False)
    else:
        body_speed = np.zeros((cloth_q.shape[0],), dtype=np.float32)
        dynamic_box_top = np.full(
            (cloth_q.shape[0],), float(meta["box_top_z"]), dtype=np.float32
        )
        box_positions = np.repeat(
            np.asarray(meta["box_pos"], dtype=np.float32)[None, :],
            cloth_q.shape[0],
            axis=0,
        )
    cloth_z_min = cloth_q[:, :, 2].min(axis=1)
    cloth_z_mean = cloth_q[:, :, 2].mean(axis=1)
    cloth_z_max = cloth_q[:, :, 2].max(axis=1)
    clearance_min = cloth_z_min - dynamic_box_top
    clearance_mean = cloth_z_mean - dynamic_box_top
    clearance_max = cloth_z_max - dynamic_box_top

    def _first_negative(values: np.ndarray) -> int | None:
        idx = np.flatnonzero(values < 0.0)
        return int(idx[0]) if idx.size else None

    sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    sim_duration = max((sim_data["particle_q_all"].shape[0] - 1) * sim_frame_dt, 0.0)
    video_duration = sim_duration * float(args.slowdown)
    rendered_frame_count = max(1, int(round(video_duration * float(args.render_fps))))
    object_mass = float(np.asarray(ir_obj["mass"], dtype=np.float32).mean())
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    bbox_span = cloth_q.reshape(-1, 3).max(axis=0) - cloth_q.reshape(-1, 3).min(axis=0)
    radii = model.particle_radius.numpy().astype(np.float32)[:n_obj]
    max_penetration_depth = None
    final_penetration_p99 = None
    if bool(meta.get("has_box", False)):
        hx, hy, hz = [float(v) for v in np.asarray(meta["box_half_extents"]).ravel()]
        penetration = np.zeros((cloth_q.shape[0], cloth_q.shape[1]), dtype=np.float32)
        for frame_idx in range(cloth_q.shape[0]):
            pos = box_positions[frame_idx].astype(np.float32, copy=False)
            local = cloth_q[frame_idx] - pos[None, :]
            sdf = _box_signed_distance(local.astype(np.float32, copy=False), hx, hy, hz)
            penetration[frame_idx] = np.maximum(radii - sdf, 0.0)
        max_penetration_depth = float(np.max(penetration))
        final_penetration_p99 = float(np.quantile(penetration[-1], 0.99))
    return {
        "experiment": "cloth_box_drop_object_only",
        "self_collision_case": "on",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "drop_height_m": float(args.drop_height),
        "object_mass_per_particle": object_mass,
        "weight_scale": float(ir_obj.get("weight_scale", 1.0)),
        "mass_spring_scale": (
            None if args.mass_spring_scale is None else float(args.mass_spring_scale)
        ),
        "weight_scale": float(ir_obj.get("weight_scale", 1.0)),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(meta["total_object_mass"]),
        "rigid_mass": float(args.rigid_mass),
        "mass_ratio": float(meta["total_object_mass"] / max(args.rigid_mass, 1.0e-8)),
        "rigid_geometry": "box",
        "has_box": bool(meta.get("has_box", False)),
        "box_is_dynamic": bool(meta.get("box_is_dynamic", False)),
        "has_ground_plane": bool(meta.get("has_ground_plane", False)),
        "box_half_extents": [float(v) for v in np.asarray(meta["box_half_extents"]).ravel()],
        "box_top_z": float(meta["box_top_z"]),
        "reverse_z": bool(newton_import_ir.ir_bool(ir_obj, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "contact_collision_dist_used": float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]),
        "contact_dist_scale": float(args.contact_dist_scale),
        "on_contact_dist_scale": float(args.on_contact_dist_scale),
        "frames": int(args.frames),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_frame_dt),
        "sim_duration_sec": float(sim_duration),
        "wall_time_sec": float(sim_data["wall_time"]),
        "slowdown_factor": float(args.slowdown),
        "render_fps": float(args.render_fps),
        "rendered_frame_count": int(rendered_frame_count),
        "video_duration_target_sec": float(video_duration),
        "all_particle_positions_finite": bool(np.isfinite(cloth_q).all()),
        "particle_bbox_span_xyz_m": [float(v) for v in bbox_span],
        "body_speed_initial": float(body_speed[0]),
        "body_speed_final": float(body_speed[-1]),
        "body_speed_max": float(np.max(body_speed)),
        "dynamic_box_top_final_z": float(dynamic_box_top[-1]),
        "first_min_below_top_frame": _first_negative(clearance_min),
        "first_mean_below_top_frame": _first_negative(clearance_mean),
        "first_all_below_top_frame": _first_negative(clearance_max),
        "min_clearance_min_to_box_top": float(np.min(clearance_min)),
        "min_clearance_mean_to_box_top": float(np.min(clearance_mean)),
        "min_clearance_max_to_box_top": float(np.min(clearance_max)),
        "final_clearance_min_to_box_top": float(clearance_min[-1]),
        "final_clearance_mean_to_box_top": float(clearance_mean[-1]),
        "final_clearance_max_to_box_top": float(clearance_max[-1]),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "spring_ke_scale": float(spring_ke_scale),
        "spring_kd_scale": float(spring_kd_scale),
        "shape_contact_scale": (
            None if args.shape_contact_scale is None else float(args.shape_contact_scale)
        ),
        "shape_contact_damping_multiplier": float(args.shape_contact_damping_multiplier),
        "use_decoupled_shape_materials": bool(use_decoupled_shape_materials),
        "soft_contact_ke_final": float(model.soft_contact_ke),
        "soft_contact_kd_final": float(model.soft_contact_kd),
        "soft_contact_kf_final": float(model.soft_contact_kf),
        "max_penetration_depth_box_m": max_penetration_depth,
        "final_penetration_p99_box_m": final_penetration_p99,
        "particle_contact_ke_final": float(model.particle_ke),
        "particle_contact_kd_final": float(model.particle_kd),
        "particle_contact_kf_final": float(model.particle_kf),
        "particle_self_contact_scale": float(args.particle_self_contact_scale),
        "particle_contacts_enabled": bool(particle_contacts_enabled),
        "disable_particle_contact_kernel": bool(disable_particle_contact_kernel),
        "camera_pos": [float(v) for v in args.camera_pos],
        "camera_pitch": float(args.camera_pitch),
        "camera_yaw": float(args.camera_yaw),
        "camera_fov": float(args.camera_fov),
        "camera_mode": "manual",
        "render_video": str(out_mp4) if out_mp4 is not None else None,
    }


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.prefix}_{_mass_tag(args.rigid_mass)}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _case_out_dir(base_out_dir: Path) -> Path:
    return base_out_dir / "self_on"


def run_case(base_args: argparse.Namespace, raw_ir: dict[str, np.ndarray], device: str) -> dict[str, Path]:
    args = copy.deepcopy(base_args)
    args.out_dir = _case_out_dir(base_args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.prefix = f"{base_args.prefix}_on"
    args.disable_particle_contact_kernel = False
    ir_obj = _copy_object_only_ir(raw_ir, args)
    if "contact_collision_dist" in ir_obj:
        scaled_dist = float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]) * float(args.contact_dist_scale)
        scaled_dist *= float(args.on_contact_dist_scale)
        ir_obj["contact_collision_dist"] = np.asarray(scaled_dist, dtype=np.float32)

    print(f"Building cloth+box model (on) from {args.ir.resolve()}", flush=True)
    model, meta, n_obj = build_model(ir_obj, args, device)

    print("Running simulation (on)...", flush=True)
    sim_data = simulate(model, ir_obj, meta, args, n_obj, device)

    scene_npz = save_scene_npz(args, sim_data, meta, n_obj)
    print(f"  Scene NPZ: {scene_npz}", flush=True)

    out_mp4: Path | None = None
    if not bool(args.skip_render):
        print("Rendering video...", flush=True)
        out_mp4 = args.out_dir / f"{args.prefix}_{_mass_tag(args.rigid_mass)}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        render_video(model, sim_data, meta, args, device, out_mp4)

    summary = build_summary(
        model,
        args,
        ir_obj,
        sim_data,
        meta,
        n_obj,
        out_mp4,
        particle_contacts_enabled=True,
        disable_particle_contact_kernel=False,
    )
    summary_path = save_summary_json(args, summary)
    print(f"  Summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return {
        "scene_npz": scene_npz,
        "summary_json": summary_path,
        "video_mp4": out_mp4 if out_mp4 is not None else Path(""),
    }


def main() -> int:
    args = parse_args()
    _validate_scaling_args(args)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    raw_ir = load_ir(args.ir)
    _maybe_autoset_mass_spring_scale(
        args, raw_ir, target_total_mass=float(args.target_total_mass)
    )

    run_case(args, raw_ir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
