#!/usr/bin/env python3
"""Object-only cloth drop onto a Stanford Bunny in Newton SemiImplicit.

This script is intentionally OFF-only:

- SELF COLLISION: OFF

The cloth is loaded from a PhysTwin IR bundle, controllers are dropped, and the
object particles are released from rest above the bunny.
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
import trimesh
import warp as wp

from rope_bunny_drop_experiment import (
    _apply_drag_correction_ignore_axis,
    load_bunny_mesh,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
    quat_to_rotmat,
)
from newton._src.solvers.semi_implicit.kernels_body import eval_body_joint_forces
from newton._src.solvers.semi_implicit.kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_triangle_contact_forces,
)
from newton._src.solvers.semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)



def _default_cloth_ir() -> Path:
    return Path("Newton/phystwin_bridge/ir/blue_cloth_double_lift_around/phystwin_ir_v2_bf_strict.npz")


def load_ir(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _mass_tag(value: float) -> str:
    text = f"{float(value):g}"
    text = text.replace("-", "neg")
    return "m" + text.replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-only cloth drops onto a Stanford Bunny.")
    p.add_argument("--ir", type=Path, default=_default_cloth_ir(), help="Path to PhysTwin cloth IR .npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="cloth_bunny_drop")
    p.add_argument("--device", default=path_defaults.default_device())

    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--sim-dt", type=float, default=5.0e-5)
    p.add_argument("--substeps", type=int, default=667)
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--drop-height", type=float, default=0.5, help="Cloth bottom height above bunny top, in meters.")
    p.add_argument("--cloth-shift-x", type=float, default=0.0)
    p.add_argument("--cloth-shift-y", type=float, default=0.0)
    p.add_argument("--object-mass", type=float, default=None, help="Optional per-particle cloth object mass override.")
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
        "--bunny-shape-contact-scale",
        type=float,
        default=None,
        help=(
            "Optional override for bunny shape_material_ke scaling. "
            "If omitted, bunny uses --shape-contact-scale."
        ),
    )
    p.add_argument(
        "--bunny-shape-contact-damping-multiplier",
        type=float,
        default=None,
        help=(
            "Optional override for bunny shape_material_kd/kf scaling multiplier. "
            "If omitted, bunny uses --shape-contact-damping-multiplier."
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
        "--add-bunny",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the Stanford Bunny rigid body in the scene.",
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
        help="Enable particle-vs-shape contacts for rigid shapes such as the bunny.",
    )

    p.add_argument("--rigid-mass", type=float, default=4000.0)
    p.add_argument("--rigid-shape", choices=["bunny", "box"], default="bunny")
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
    p.add_argument("--box-hx", type=float, default=0.12)
    p.add_argument("--box-hy", type=float, default=0.12)
    p.add_argument("--box-hz", type=float, default=0.08)

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
            "This avoids softening bunny-ground support when low-mass cloth contact is rescaled."
        ),
    )
    return p.parse_args()


def _validate_scaling_args(args: argparse.Namespace) -> None:
    if args.mass_spring_scale is None:
        return
    scale = float(args.mass_spring_scale)
    if scale <= 0.0:
        raise ValueError(f"--mass-spring-scale must be > 0, got {scale}")
    if args.object_mass is not None:
        raise ValueError(
            "--mass-spring-scale cannot be used together with --object-mass. "
            "Use only one source of mass scaling."
        )
    if not np.isclose(float(args.spring_ke_scale), 1.0):
        raise ValueError(
            "--mass-spring-scale cannot be used together with --spring-ke-scale. "
            "The unified scale already controls spring_ke."
        )
    if not np.isclose(float(args.spring_kd_scale), 1.0):
        raise ValueError(
            "--mass-spring-scale cannot be used together with --spring-kd-scale. "
            "The unified scale already controls spring_kd."
        )


def _effective_object_mass_array(ir_demo: dict[str, Any], args: argparse.Namespace, n_obj: int) -> np.ndarray:
    mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()[:n_obj]
    if args.object_mass is not None:
        mass.fill(float(args.object_mass))
    elif args.mass_spring_scale is not None:
        mass *= float(args.mass_spring_scale)
    return mass


def _effective_spring_scales(args: argparse.Namespace) -> tuple[float, float]:
    if args.mass_spring_scale is not None:
        scale = float(args.mass_spring_scale)
        return scale, scale
    return float(args.spring_ke_scale), float(args.spring_kd_scale)


def _apply_shape_contact_scaling(model, args: argparse.Namespace) -> None:
    if args.shape_contact_scale is None:
        return
    alpha = float(args.shape_contact_scale)
    if alpha <= 0.0:
        raise ValueError(f"--shape-contact-scale must be > 0, got {alpha}")
    damping_mult = float(args.shape_contact_damping_multiplier)
    if damping_mult <= 0.0:
        raise ValueError(
            f"--shape-contact-damping-multiplier must be > 0, got {damping_mult}"
        )

    # Scale the actual particle-shape soft-contact path used by
    # eval_particle_body_contact_forces():
    # - model.soft_contact_ke/kd/kf
    # - model.shape_material_ke/kd/kf
    model.soft_contact_ke = float(model.soft_contact_ke * alpha)
    model.soft_contact_kd = float(model.soft_contact_kd * alpha * damping_mult)
    model.soft_contact_kf = float(model.soft_contact_kf * alpha * damping_mult)

    labels = [str(label).lower() for label in list(model.shape_label)]
    has_ground = any(("ground" in label or "plane" in label) for label in labels)

    ground_alpha = alpha if args.ground_shape_contact_scale is None else float(args.ground_shape_contact_scale)
    ground_dmult = (
        damping_mult
        if args.ground_shape_contact_damping_multiplier is None
        else float(args.ground_shape_contact_damping_multiplier)
    )
    bunny_alpha = alpha if args.bunny_shape_contact_scale is None else float(args.bunny_shape_contact_scale)
    bunny_dmult = (
        damping_mult
        if args.bunny_shape_contact_damping_multiplier is None
        else float(args.bunny_shape_contact_damping_multiplier)
    )

    for arr_name in ("shape_material_ke", "shape_material_kd", "shape_material_kf"):
        arr = getattr(model, arr_name)
        vals = arr.numpy().astype(np.float32, copy=False).copy()
        for idx, label in enumerate(labels):
            scale = np.float32(alpha)
            dmult = np.float32(damping_mult)
            if "ground" in label or "plane" in label:
                scale = np.float32(ground_alpha)
                dmult = np.float32(ground_dmult)
            elif "bunny" in label or (bool(args.add_bunny) and (not ("ground" in label or "plane" in label)) and has_ground):
                scale = np.float32(bunny_alpha)
                dmult = np.float32(bunny_dmult)
            vals[idx] *= scale
            if arr_name != "shape_material_ke":
                vals[idx] *= dmult
        arr.assign(vals)


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

    mass = _effective_object_mass_array(ir_demo, args, n_obj)

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = mass
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


def _box_signed_distance(local_points: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    q = np.abs(local_points) - np.array([hx, hy, hz], dtype=np.float32)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def build_model(ir_obj: dict[str, Any], args: argparse.Namespace, device: str):
    shape_contacts_enabled = bool(args.shape_contacts) and bool(args.add_bunny)
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(args)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(spring_ke_scale),
        spring_kd_scale=float(spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
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

    qx, qy, qz, qw = [float(v) for v in args.bunny_quat_xyzw]
    bunny_quat_xyzw = [qx, qy, qz, qw]
    bunny_quat = wp.quat(qx, qy, qz, qw)
    mesh_verts_local = np.zeros((0, 3), dtype=np.float32)
    mesh_tri_indices = np.zeros((0, 3), dtype=np.int32)
    mesh_render_edges = np.zeros((0, 2), dtype=np.int32)
    mesh_asset_path = ""
    rigid_pos = np.zeros((3,), dtype=np.float32)
    rigid_top_z = 0.0
    rigid_quat_xyzw = [1.0, 0.0, 0.0, 0.0]
    rigid_quat = wp.quat(0.0, 0.0, 0.0, 1.0)
    rigid_shape = str(args.rigid_shape)
    box_half_extents = np.array([float(args.box_hx), float(args.box_hy), float(args.box_hz)], dtype=np.float32)
    if bool(args.add_bunny) and rigid_shape == "bunny":
        mesh, mesh_verts_local, mesh_tri_indices, mesh_render_edges, mesh_asset_path = load_bunny_mesh(
            args.bunny_asset, args.bunny_prim
        )
        verts_rotated = (mesh_verts_local * float(args.bunny_scale)) @ quat_to_rotmat(bunny_quat_xyzw).T
        bunny_z_min = float(verts_rotated[:, 2].min())
        bunny_z_max = float(verts_rotated[:, 2].max())
        rigid_pos = np.array([0.0, 0.0, -bunny_z_min], dtype=np.float32)
        rigid_top_z = rigid_pos[2] + bunny_z_max
        rigid_quat_xyzw = bunny_quat_xyzw
        rigid_quat = bunny_quat
    elif bool(args.add_bunny) and rigid_shape == "box":
        rigid_pos = np.array([0.0, 0.0, float(args.box_hz)], dtype=np.float32)
        rigid_top_z = float(args.box_hz * 2.0)

    x0 = np.asarray(ir_obj["x0"], dtype=np.float32).copy()
    bbox_min = x0.min(axis=0)
    bbox_max = x0.max(axis=0)
    cloth_center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    reference_top_z = float(rigid_top_z) if bool(args.add_bunny) else 0.0
    shift = np.array(
        [
            -float(cloth_center_xy[0]) + float(args.cloth_shift_x),
            -float(cloth_center_xy[1]) + float(args.cloth_shift_y),
            reference_top_z + float(args.drop_height) - float(bbox_min[2]),
        ],
        dtype=np.float32,
    )
    shifted_q = x0 + shift
    builder.particle_q = [wp.vec3(*row.tolist()) for row in shifted_q]

    if bool(args.add_bunny):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*rigid_pos.tolist()), rigid_quat),
            mass=float(args.rigid_mass),
            inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
            lock_inertia=True,
            label=str(rigid_shape),
        )
        rigid_cfg = builder.default_shape_cfg.copy()
        rigid_cfg.mu = float(args.body_mu)
        rigid_cfg.ke = float(args.body_ke)
        rigid_cfg.kd = float(args.body_kd)
        if rigid_shape == "bunny":
            builder.add_shape_mesh(
                body=body,
                mesh=mesh,
                scale=(float(args.bunny_scale), float(args.bunny_scale), float(args.bunny_scale)),
                cfg=rigid_cfg,
            )
        else:
            builder.add_shape_box(
                body=body,
                hx=float(args.box_hx),
                hy=float(args.box_hy),
                hz=float(args.box_hz),
                cfg=rigid_cfg,
            )

    model = builder.finalize(device=device)
    shape_material_ke_base = model.shape_material_ke.numpy().astype(np.float32, copy=False).copy()
    shape_material_kd_base = model.shape_material_kd.numpy().astype(np.float32, copy=False).copy()
    shape_material_kf_base = model.shape_material_kf.numpy().astype(np.float32, copy=False).copy()
    model.particle_grid = None
    _, gravity_vec = newton_import_ir.resolve_gravity(cfg, ir_obj)
    model.set_gravity(gravity_vec)
    _ = newton_import_ir._apply_ps_object_collision_mapping(model, ir_obj, cfg, checks)
    _apply_shape_contact_scaling(model, args)
    shape_material_ke_scaled = model.shape_material_ke.numpy().astype(np.float32, copy=False).copy()
    shape_material_kd_scaled = model.shape_material_kd.numpy().astype(np.float32, copy=False).copy()
    shape_material_kf_scaled = model.shape_material_kf.numpy().astype(np.float32, copy=False).copy()

    n_obj = int(np.asarray(ir_obj["num_object_points"]).ravel()[0])
    edges = np.asarray(ir_obj["spring_edges"], dtype=np.int32)
    render_edges = edges[:: max(1, int(args.spring_stride))].astype(np.int32, copy=True)
    meta = {
        "mesh_verts_local": mesh_verts_local.astype(np.float32, copy=False),
        "mesh_tri_indices": mesh_tri_indices.astype(np.int32, copy=False),
        "mesh_scale": float(args.bunny_scale),
        "mesh_render_edges": mesh_render_edges.astype(np.int32, copy=False),
        "mesh_asset_path": mesh_asset_path,
        "rigid_shape": rigid_shape,
        "bunny_quat_xyzw": rigid_quat_xyzw,
        "bunny_pos": rigid_pos.astype(np.float32, copy=False),
        "bunny_top_z": float(rigid_top_z),
        "box_half_extents": box_half_extents,
        "has_bunny": bool(args.add_bunny),
        "has_ground_plane": bool(args.add_ground_plane),
        "cloth_shift": shift.astype(np.float32, copy=False),
        "render_edges": render_edges,
        "particle_contacts_enabled": False,
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "shape_material_ke_base": shape_material_ke_base,
        "shape_material_kd_base": shape_material_kd_base,
        "shape_material_kf_base": shape_material_kf_base,
        "shape_material_ke_scaled": shape_material_ke_scaled,
        "shape_material_kd_scaled": shape_material_kd_scaled,
        "shape_material_kf_scaled": shape_material_kf_scaled,
    }
    return model, meta, n_obj


def _assign_shape_material_triplet(model, ke: np.ndarray, kd: np.ndarray, kf: np.ndarray) -> None:
    model.shape_material_ke.assign(np.asarray(ke, dtype=np.float32))
    model.shape_material_kd.assign(np.asarray(kd, dtype=np.float32))
    model.shape_material_kf.assign(np.asarray(kf, dtype=np.float32))


def simulate(
    model: newton.Model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    shape_contacts_enabled = bool(args.shape_contacts) and bool(args.add_bunny)
    cfg = newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir.resolve(),
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=True,
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

            if not bool(args.decouple_shape_materials):
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
                if solver.enable_tri_contact:
                    eval_triangle_contact_forces(model, state_in, particle_f)

                _assign_shape_material_triplet(
                    model,
                    meta["shape_material_ke_base"],
                    meta["shape_material_kd_base"],
                    meta["shape_material_kf_base"],
                )
                eval_body_contact_forces(
                    model, state_in, contacts, friction_smoothing=solver.friction_smoothing, body_f_out=body_f_work
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

        radii = model.particle_radius.numpy().astype(np.float32, copy=False)
        radii = np.minimum(radii * float(args.particle_radius_vis_scale), float(args.particle_radius_vis_min))
        model.particle_radius.assign(radii)

        try:
            shape_colors = {}
            for idx, label in enumerate(list(model.shape_label)):
                name = str(label).lower()
                if "bunny" in name or (meta.get("rigid_shape") == "box" and "shape_1" in name):
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
            f" | BUNNY MASS: {float(args.rigid_mass):.3g} kg"
        )
        for out_idx, sim_idx in enumerate(render_indices):
            state.particle_q.assign(sim_data["particle_q_all"][sim_idx].astype(np.float32, copy=False))
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
                        "CLOTH BUNNY DROP",
                        "SELF COLLISION: OFF",
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
        rigid_mesh_vertices_local=meta["mesh_verts_local"],
        rigid_mesh_indices=meta["mesh_tri_indices"],
        rigid_mesh_scale=np.float32(meta["mesh_scale"]),
        rigid_mesh_render_edges=meta["mesh_render_edges"],
        rigid_shape_kind=np.asarray(str(meta.get("rigid_shape", "bunny"))),
        rigid_box_half_extents=np.asarray(meta.get("box_half_extents", np.zeros(3, dtype=np.float32)), dtype=np.float32),
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
    cloth_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    if body_q.ndim == 3 and body_q.shape[1] > 0 and bool(meta.get("has_bunny", False)):
        body_speed = np.linalg.norm(sim_data["body_vel"][:, 0, :], axis=1)
        body_z = body_q[:, 0, 2]
        bunny_top_offset = float(meta["bunny_top_z"]) - float(body_z[0])
        dynamic_bunny_top = body_z + bunny_top_offset
    else:
        body_speed = np.zeros((cloth_q.shape[0],), dtype=np.float32)
        dynamic_bunny_top = np.zeros((cloth_q.shape[0],), dtype=np.float32)
    cloth_z_min = cloth_q[:, :, 2].min(axis=1)
    cloth_z_mean = cloth_q[:, :, 2].mean(axis=1)
    cloth_z_max = cloth_q[:, :, 2].max(axis=1)
    clearance_min = cloth_z_min - dynamic_bunny_top
    clearance_mean = cloth_z_mean - dynamic_bunny_top
    clearance_max = cloth_z_max - dynamic_bunny_top

    def _first_negative(values: np.ndarray) -> int | None:
        idx = np.flatnonzero(values < 0.0)
        return int(idx[0]) if idx.size else None

    sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    sim_duration = max((sim_data["particle_q_all"].shape[0] - 1) * sim_frame_dt, 0.0)
    video_duration = sim_duration * float(args.slowdown)
    rendered_frame_count = max(1, int(round(video_duration * float(args.render_fps))))
    object_mass = float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].mean())
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(args)
    max_penetration_depth = None
    final_penetration_p99 = None
    if str(meta.get("rigid_shape")) == "box" and body_q.ndim == 3 and body_q.shape[1] > 0:
        hx, hy, hz = [float(v) for v in np.asarray(meta["box_half_extents"]).ravel()]
        radii = model.particle_radius.numpy().astype(np.float32)[:n_obj]
        penetration = np.zeros((cloth_q.shape[0], cloth_q.shape[1]), dtype=np.float32)
        for frame_idx in range(cloth_q.shape[0]):
            pos = body_q[frame_idx, 0, :3].astype(np.float32)
            quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
            rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
            local = (cloth_q[frame_idx] - pos[None, :]) @ rot
            sdf = _box_signed_distance(local.astype(np.float32, copy=False), hx, hy, hz)
            penetration[frame_idx] = np.maximum(radii - sdf, 0.0)
        max_penetration_depth = float(np.max(penetration))
        final_penetration_p99 = float(np.quantile(penetration[-1], 0.99))
    bunny_mesh_max_penetration = None
    bunny_mesh_final_p99_penetration = None
    if str(meta.get("rigid_shape")) == "bunny" and body_q.ndim == 3 and body_q.shape[1] > 0:
        faces = np.asarray(meta.get("mesh_tri_indices", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        verts_local = np.asarray(meta.get("mesh_verts_local", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if faces.size and verts_local.size:
            radius = model.particle_radius.numpy().astype(np.float32)[:n_obj]
            scale = float(meta["mesh_scale"])
            sampled_max = 0.0
            frame_ids = list(range(0, cloth_q.shape[0], 15))
            if frame_ids[-1] != cloth_q.shape[0] - 1:
                frame_ids.append(cloth_q.shape[0] - 1)
            final_penetration = None
            for frame_idx in frame_ids:
                pos = body_q[frame_idx, 0, :3].astype(np.float32)
                quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
                rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
                verts = (verts_local * scale) @ rot.T + pos[None, :]
                tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                _, dist, _ = trimesh.proximity.closest_point(tri, cloth_q[frame_idx].astype(np.float64))
                penetration = np.maximum(radius - dist.astype(np.float32, copy=False), 0.0)
                sampled_max = max(sampled_max, float(np.max(penetration)))
                if frame_idx == cloth_q.shape[0] - 1:
                    final_penetration = penetration
            bunny_mesh_max_penetration = float(sampled_max)
            if final_penetration is not None:
                bunny_mesh_final_p99_penetration = float(np.quantile(final_penetration, 0.99))
    return {
        "experiment": "cloth_bunny_drop_object_only",
        "self_collision_case": "off",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "drop_height_m": float(args.drop_height),
        "cloth_shift_x_m": float(args.cloth_shift_x),
        "cloth_shift_y_m": float(args.cloth_shift_y),
        "object_mass_per_particle": object_mass,
        "mass_spring_scale": (
            None if args.mass_spring_scale is None else float(args.mass_spring_scale)
        ),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "rigid_mass": float(args.rigid_mass),
        "mass_ratio": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum() / max(args.rigid_mass, 1.0e-8)),
        "rigid_geometry": str(meta.get("rigid_shape", "bunny")),
        "has_bunny": bool(meta.get("has_bunny", False)),
        "has_ground_plane": bool(meta.get("has_ground_plane", False)),
        "bunny_scale": float(args.bunny_scale),
        "bunny_quat_xyzw": [float(v) for v in meta["bunny_quat_xyzw"]],
        "bunny_top_z": float(meta["bunny_top_z"]),
        "reverse_z": bool(newton_import_ir.ir_bool(ir_obj, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "contact_collision_dist_used": float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]),
        "contact_dist_scale": float(args.contact_dist_scale),
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
        "body_speed_initial": float(body_speed[0]),
        "body_speed_final": float(body_speed[-1]),
        "body_speed_max": float(np.max(body_speed)),
        "dynamic_bunny_top_final_z": float(dynamic_bunny_top[-1]),
        "first_min_below_top_frame": _first_negative(clearance_min),
        "first_mean_below_top_frame": _first_negative(clearance_mean),
        "first_all_below_top_frame": _first_negative(clearance_max),
        "min_clearance_min_to_bunny_top": float(np.min(clearance_min)),
        "min_clearance_mean_to_bunny_top": float(np.min(clearance_mean)),
        "min_clearance_max_to_bunny_top": float(np.min(clearance_max)),
        "final_clearance_min_to_bunny_top": float(clearance_min[-1]),
        "final_clearance_mean_to_bunny_top": float(clearance_mean[-1]),
        "final_clearance_max_to_bunny_top": float(clearance_max[-1]),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "spring_ke_scale": float(spring_ke_scale),
        "spring_kd_scale": float(spring_kd_scale),
        "shape_contact_scale": (
            None if args.shape_contact_scale is None else float(args.shape_contact_scale)
        ),
        "shape_contact_damping_multiplier": float(args.shape_contact_damping_multiplier),
        "soft_contact_ke_final": float(model.soft_contact_ke),
        "soft_contact_kd_final": float(model.soft_contact_kd),
        "soft_contact_kf_final": float(model.soft_contact_kf),
        "max_penetration_depth_rigid_m": max_penetration_depth,
        "final_penetration_p99_rigid_m": final_penetration_p99,
        "max_penetration_depth_bunny_mesh_m": bunny_mesh_max_penetration,
        "final_penetration_p99_bunny_mesh_m": bunny_mesh_final_p99_penetration,
        "particle_contact_ke_final": float(model.particle_ke),
        "particle_contact_kd_final": float(model.particle_kd),
        "particle_contact_kf_final": float(model.particle_kf),
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
    return base_out_dir / "self_off"


def run_case(base_args: argparse.Namespace, raw_ir: dict[str, np.ndarray], device: str) -> dict[str, Path]:
    args = copy.deepcopy(base_args)
    args.out_dir = _case_out_dir(base_args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.prefix = f"{base_args.prefix}_off"
    args.disable_particle_contact_kernel = True
    ir_obj = _copy_object_only_ir(raw_ir, args)
    if "contact_collision_dist" in ir_obj:
        scaled_dist = float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]) * float(args.contact_dist_scale)
        ir_obj["contact_collision_dist"] = np.asarray(scaled_dist, dtype=np.float32)

    print(f"Building cloth+bunny model (off) from {args.ir.resolve()}", flush=True)
    model, meta, n_obj = build_model(ir_obj, args, device)

    print("Running simulation (off)...", flush=True)
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
        particle_contacts_enabled=False,
        disable_particle_contact_kernel=True,
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

    run_case(args, raw_ir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
