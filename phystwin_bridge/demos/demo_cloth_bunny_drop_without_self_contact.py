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
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
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
    load_bunny_mesh,
    newton,
    newton_import_ir,
    overlay_text_lines_rgb,
    path_defaults,
    quat_to_rotmat,
)
from demo_shared import apply_viewer_shape_colors, compute_visual_particle_radii, temporary_particle_radius_override
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
    p.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(-0.95, 0.85, 0.78),
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--camera-pitch", type=float, default=-10.0)
    p.add_argument("--camera-yaw", type=float, default=-40.0)
    p.add_argument("--camera-fov", type=float, default=55.0)
    p.add_argument(
        "--post-contact-video-seconds",
        type=float,
        default=1.0,
        help=(
            "End the rendered video this many output-video seconds after the first cloth-rigid contact. "
            "Use <= 0 to disable trimming."
        ),
    )
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
        "--force-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture the exact collision-triggering substep, split internal vs external cloth forces, "
            "and dump diagnostic artifacts."
        ),
    )
    p.add_argument(
        "--force-dump-dir",
        type=Path,
        default=None,
        help="Optional output directory for force diagnostic artifacts. Defaults to <out-dir>/force_diagnostic.",
    )
    p.add_argument(
        "--force-topk",
        type=int,
        default=32,
        help="Number of highest external-force cloth nodes retained in the diagnostic summary and snapshot.",
    )
    p.add_argument(
        "--force-arrow-scale",
        type=float,
        default=0.05,
        help=(
            "Maximum rendered arrow length [m] for each diagnostic vector family in the trigger snapshot. "
            "Forces and accelerations are normalized per family before display."
        ),
    )
    p.add_argument(
        "--force-snapshot-frame",
        choices=["trigger", "trigger_plus_1"],
        default="trigger",
        help="Choose whether the rendered diagnostic snapshot uses the trigger substep or the immediately following substep.",
    )
    p.add_argument(
        "--stop-after-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the rollout immediately after the requested diagnostic snapshot has been captured.",
    )
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


def build_model(ir_obj: dict[str, Any], args: argparse.Namespace, device: str):
    shape_contacts_enabled = bool(args.shape_contacts) and bool(args.add_bunny)
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
    _apply_particle_contact_scaling(
        model, weight_scale=float(ir_obj.get("weight_scale", 1.0))
    )
    _apply_shape_contact_scaling(
        model, args, weight_scale=float(ir_obj.get("weight_scale", 1.0))
    )
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


def _resolve_force_dump_dir(args: argparse.Namespace) -> Path:
    if args.force_dump_dir is not None:
        return args.force_dump_dir.expanduser().resolve()
    return (args.out_dir / "force_diagnostic").resolve()


def _select_force_topk(external_norm: np.ndarray, *, topk: int, eps: float) -> np.ndarray:
    active = np.flatnonzero(external_norm > eps)
    candidates = active if active.size else np.arange(external_norm.shape[0], dtype=np.int32)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(external_norm[candidates])[::-1]
    limit = min(max(1, int(topk)), candidates.size)
    return candidates[order[:limit]].astype(np.int32, copy=False)


def _box_surface_info(
    points_world: np.ndarray,
    body_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    half_extents: np.ndarray,
    radius: np.ndarray,
) -> dict[str, np.ndarray]:
    rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
    local = (points_world - body_pos[None, :]) @ rot
    hx, hy, hz = [float(v) for v in half_extents.ravel()]
    closest_local = np.clip(local, [-hx, -hy, -hz], [hx, hy, hz]).astype(np.float32, copy=False)
    outside_delta = local - closest_local
    outside_dist = np.linalg.norm(outside_delta, axis=1)
    sdf = _box_signed_distance(local.astype(np.float32, copy=False), hx, hy, hz).astype(np.float32, copy=False)

    normals_local = np.zeros_like(local, dtype=np.float32)
    outside_mask = outside_dist > 1.0e-8
    if np.any(outside_mask):
        normals_local[outside_mask] = (
            outside_delta[outside_mask] / outside_dist[outside_mask, None]
        ).astype(np.float32, copy=False)

    inside_mask = ~outside_mask
    if np.any(inside_mask):
        local_inside = local[inside_mask]
        dist_to_face = np.stack(
            [
                hx - local_inside[:, 0],
                local_inside[:, 0] + hx,
                hy - local_inside[:, 1],
                local_inside[:, 1] + hy,
                hz - local_inside[:, 2],
                local_inside[:, 2] + hz,
            ],
            axis=1,
        )
        face_ids = np.argmin(dist_to_face, axis=1)
        face_normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        normals_local[inside_mask] = face_normals[face_ids]

        inside_indices = np.flatnonzero(inside_mask)
        for local_idx, point_idx in enumerate(inside_indices):
            face_id = int(face_ids[local_idx])
            if face_id == 0:
                closest_local[point_idx, 0] = hx
            elif face_id == 1:
                closest_local[point_idx, 0] = -hx
            elif face_id == 2:
                closest_local[point_idx, 1] = hy
            elif face_id == 3:
                closest_local[point_idx, 1] = -hy
            elif face_id == 4:
                closest_local[point_idx, 2] = hz
            else:
                closest_local[point_idx, 2] = -hz

    closest_world = closest_local @ rot.T + body_pos[None, :]
    face_normal_world = normals_local @ rot.T
    outward_normal = face_normal_world.copy()
    flip_mask = np.sum((points_world - closest_world) * outward_normal, axis=1) < 0.0
    outward_normal[flip_mask] *= -1.0
    penetration = np.maximum(radius - sdf, 0.0).astype(np.float32, copy=False)
    triangle_id = np.full((points_world.shape[0],), -1, dtype=np.int32)
    return {
        "closest_point_world": closest_world.astype(np.float32, copy=False),
        "face_normal_world": face_normal_world.astype(np.float32, copy=False),
        "outward_normal_world": outward_normal.astype(np.float32, copy=False),
        "penetration_depth": penetration,
        "triangle_id": triangle_id,
        "distance_to_surface": np.abs(sdf).astype(np.float32, copy=False),
    }


def _mesh_surface_info(
    points_world: np.ndarray,
    body_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    meta: dict[str, Any],
    radius: np.ndarray,
) -> dict[str, np.ndarray]:
    faces = np.asarray(meta.get("mesh_tri_indices", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
    verts_local = np.asarray(meta.get("mesh_verts_local", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if not faces.size or not verts_local.size:
        raise ValueError("Mesh surface info requested, but no bunny mesh data is available in meta.")

    scale = float(meta["mesh_scale"])
    rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
    verts = (verts_local * scale) @ rot.T + body_pos[None, :]
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    closest_point, dist, triangle_id = trimesh.proximity.closest_point(tri, points_world.astype(np.float64))
    triangle_id = np.asarray(triangle_id, dtype=np.int32)
    face_normal_world = tri.face_normals[triangle_id].astype(np.float32, copy=False)
    closest_world = np.asarray(closest_point, dtype=np.float32)
    outward_normal = face_normal_world.copy()
    flip_mask = np.sum((points_world - closest_world) * outward_normal, axis=1) < 0.0
    outward_normal[flip_mask] *= -1.0
    dist = np.asarray(dist, dtype=np.float32)
    penetration = np.maximum(radius - dist, 0.0).astype(np.float32, copy=False)
    return {
        "closest_point_world": closest_world,
        "face_normal_world": face_normal_world,
        "outward_normal_world": outward_normal.astype(np.float32, copy=False),
        "penetration_depth": penetration,
        "triangle_id": triangle_id,
        "distance_to_surface": dist,
    }


def _capture_force_snapshot(
    *,
    frame_index: int,
    substep_index_in_frame: int,
    global_substep_index: int,
    sim_time: float,
    sim_dt: float,
    particle_q: np.ndarray,
    particle_qd: np.ndarray,
    body_q: np.ndarray,
    body_qd: np.ndarray,
    particle_radius: np.ndarray,
    mass: np.ndarray,
    gravity_vec: np.ndarray,
    f_spring: np.ndarray,
    f_internal_total: np.ndarray,
    f_external_total: np.ndarray,
    meta: dict[str, Any],
    force_topk: int,
    force_eps: float,
) -> dict[str, Any]:
    if body_q.shape[0] == 0:
        raise ValueError("Force diagnostics require a rigid body pose, but no body state is present.")

    rigid_kind = str(meta.get("rigid_shape", "bunny"))
    body_pos = body_q[0, :3].astype(np.float32, copy=False)
    quat_xyzw = body_q[0, 3:7].astype(np.float32, copy=False)
    if rigid_kind == "box":
        surface = _box_surface_info(
            particle_q,
            body_pos,
            quat_xyzw,
            np.asarray(meta["box_half_extents"], dtype=np.float32),
            particle_radius,
        )
    else:
        surface = _mesh_surface_info(particle_q, body_pos, quat_xyzw, meta, particle_radius)

    acceleration_total = (
        (f_internal_total + f_external_total) / np.maximum(mass[:, None], 1.0e-12)
        + gravity_vec.reshape(1, 3)
    ).astype(np.float32, copy=False)
    external_norm = np.linalg.norm(f_external_total, axis=1).astype(np.float32, copy=False)
    internal_norm = np.linalg.norm(f_internal_total, axis=1).astype(np.float32, copy=False)
    spring_norm = np.linalg.norm(f_spring, axis=1).astype(np.float32, copy=False)
    accel_norm = np.linalg.norm(acceleration_total, axis=1).astype(np.float32, copy=False)

    outward_normal = surface["outward_normal_world"]
    face_normal = surface["face_normal_world"]
    closest_world = surface["closest_point_world"]
    penetration = surface["penetration_depth"]
    triangle_id = surface["triangle_id"]
    v_n = np.sum(particle_qd * outward_normal, axis=1).astype(np.float32, copy=False)
    f_ext_n = np.sum(f_external_total * outward_normal, axis=1).astype(np.float32, copy=False)
    f_int_n = np.sum(f_internal_total * outward_normal, axis=1).astype(np.float32, copy=False)
    a_total_n = np.sum(acceleration_total * outward_normal, axis=1).astype(np.float32, copy=False)
    f_stop_n = (mass * np.maximum(-v_n, 0.0) / max(float(sim_dt), 1.0e-12)).astype(np.float32)
    topk_indices = _select_force_topk(external_norm, topk=force_topk, eps=force_eps)

    return {
        "frame_index": int(frame_index),
        "substep_index_in_frame": int(substep_index_in_frame),
        "global_substep_index": int(global_substep_index),
        "sim_time": float(sim_time),
        "particle_q": particle_q.astype(np.float32, copy=False),
        "particle_qd": particle_qd.astype(np.float32, copy=False),
        "body_q": body_q.astype(np.float32, copy=False),
        "body_qd": body_qd.astype(np.float32, copy=False),
        "particle_radius": particle_radius.astype(np.float32, copy=False),
        "particle_mass": mass.astype(np.float32, copy=False),
        "gravity_vec": gravity_vec.astype(np.float32, copy=False),
        "f_spring": f_spring.astype(np.float32, copy=False),
        "f_internal_total": f_internal_total.astype(np.float32, copy=False),
        "f_external_total": f_external_total.astype(np.float32, copy=False),
        "a_total": acceleration_total,
        "closest_point_world": closest_world.astype(np.float32, copy=False),
        "face_normal_world": face_normal.astype(np.float32, copy=False),
        "outward_normal_world": outward_normal.astype(np.float32, copy=False),
        "penetration_depth": penetration.astype(np.float32, copy=False),
        "triangle_id": triangle_id.astype(np.int32, copy=False),
        "external_force_norm": external_norm,
        "internal_force_norm": internal_norm,
        "spring_force_norm": spring_norm,
        "acceleration_norm": accel_norm,
        "external_force_normal": f_ext_n,
        "internal_force_normal": f_int_n,
        "acceleration_normal": a_total_n,
        "normal_velocity": v_n,
        "stop_force_required_normal": f_stop_n,
        "topk_indices": topk_indices.astype(np.int32, copy=False),
        "rigid_shape": rigid_kind,
    }


def _force_summary_payload(snapshot: dict[str, Any], *, eps: float) -> dict[str, Any]:
    ext_norm = np.asarray(snapshot["external_force_norm"], dtype=np.float32)
    contact_mask = ext_norm > eps
    f_ext_n = np.asarray(snapshot["external_force_normal"], dtype=np.float32)
    f_int_n = np.asarray(snapshot["internal_force_normal"], dtype=np.float32)
    a_total_n = np.asarray(snapshot["acceleration_normal"], dtype=np.float32)
    f_stop_n = np.asarray(snapshot["stop_force_required_normal"], dtype=np.float32)
    topk = np.asarray(snapshot["topk_indices"], dtype=np.int32)

    if np.any(contact_mask & (f_ext_n < 0.0)):
        dominant_issue_guess = "direction_or_geometry"
    elif np.any(contact_mask & (a_total_n < 0.0)):
        dominant_issue_guess = "insufficient_contact_magnitude"
    elif np.any(contact_mask & (f_ext_n < f_stop_n)):
        dominant_issue_guess = "insufficient_contact_magnitude"
    else:
        dominant_issue_guess = "undetermined"

    topk_records = []
    for idx in topk.tolist():
        topk_records.append(
            {
                "particle_index": int(idx),
                "external_force_norm": float(ext_norm[idx]),
                "internal_force_norm": float(snapshot["internal_force_norm"][idx]),
                "spring_force_norm": float(snapshot["spring_force_norm"][idx]),
                "acceleration_norm": float(snapshot["acceleration_norm"][idx]),
                "penetration_depth_m": float(snapshot["penetration_depth"][idx]),
                "triangle_id": int(snapshot["triangle_id"][idx]),
                "external_force_normal": float(f_ext_n[idx]),
                "internal_force_normal": float(f_int_n[idx]),
                "acceleration_normal": float(a_total_n[idx]),
                "normal_velocity": float(snapshot["normal_velocity"][idx]),
                "stop_force_required_normal": float(f_stop_n[idx]),
                "particle_position_world": [
                    float(v) for v in np.asarray(snapshot["particle_q"][idx], dtype=np.float32)
                ],
                "closest_point_world": [
                    float(v) for v in np.asarray(snapshot["closest_point_world"][idx], dtype=np.float32)
                ],
                "face_normal_world": [
                    float(v) for v in np.asarray(snapshot["face_normal_world"][idx], dtype=np.float32)
                ],
                "outward_normal_world": [
                    float(v) for v in np.asarray(snapshot["outward_normal_world"][idx], dtype=np.float32)
                ],
            }
        )

    return {
        "trigger_substep_global": int(snapshot["global_substep_index"]),
        "trigger_frame_index": int(snapshot["frame_index"]),
        "trigger_substep_index_in_frame": int(snapshot["substep_index_in_frame"]),
        "trigger_sim_time_sec": float(snapshot["sim_time"]),
        "rigid_shape": str(snapshot["rigid_shape"]),
        "contact_node_count": int(np.count_nonzero(contact_mask)),
        "contact_nodes_with_wrong_direction": int(np.count_nonzero(contact_mask & (f_ext_n < 0.0))),
        "contact_nodes_with_inward_acceleration": int(np.count_nonzero(contact_mask & (a_total_n < 0.0))),
        "contact_nodes_force_below_stop": int(np.count_nonzero(contact_mask & (f_ext_n < f_stop_n))),
        "contact_nodes_internal_dominates": int(
            np.count_nonzero(contact_mask & (np.abs(f_int_n) > np.abs(f_ext_n)))
        ),
        "dominant_issue_guess": dominant_issue_guess,
        "topk_particle_records": topk_records,
    }


def _write_force_diagnostic_outputs(
    args: argparse.Namespace,
    trigger_snapshot: dict[str, Any],
    summary_payload: dict[str, Any],
) -> tuple[Path, Path]:
    dump_dir = _resolve_force_dump_dir(args)
    dump_dir.mkdir(parents=True, exist_ok=True)
    npz_path = dump_dir / "force_diag_trigger_substep.npz"
    json_path = dump_dir / "force_diag_trigger_summary.json"

    np.savez_compressed(
        npz_path,
        frame_index=np.int32(trigger_snapshot["frame_index"]),
        substep_index_in_frame=np.int32(trigger_snapshot["substep_index_in_frame"]),
        global_substep_index=np.int32(trigger_snapshot["global_substep_index"]),
        sim_time=np.float32(trigger_snapshot["sim_time"]),
        particle_q=trigger_snapshot["particle_q"],
        particle_qd=trigger_snapshot["particle_qd"],
        body_q=trigger_snapshot["body_q"],
        body_qd=trigger_snapshot["body_qd"],
        particle_radius=trigger_snapshot["particle_radius"],
        particle_mass=trigger_snapshot["particle_mass"],
        gravity_vec=trigger_snapshot["gravity_vec"],
        f_spring=trigger_snapshot["f_spring"],
        f_internal_total=trigger_snapshot["f_internal_total"],
        f_external_total=trigger_snapshot["f_external_total"],
        a_total=trigger_snapshot["a_total"],
        closest_point_world=trigger_snapshot["closest_point_world"],
        face_normal_world=trigger_snapshot["face_normal_world"],
        outward_normal_world=trigger_snapshot["outward_normal_world"],
        penetration_depth=trigger_snapshot["penetration_depth"],
        triangle_id=trigger_snapshot["triangle_id"],
        external_force_norm=trigger_snapshot["external_force_norm"],
        internal_force_norm=trigger_snapshot["internal_force_norm"],
        spring_force_norm=trigger_snapshot["spring_force_norm"],
        acceleration_norm=trigger_snapshot["acceleration_norm"],
        external_force_normal=trigger_snapshot["external_force_normal"],
        internal_force_normal=trigger_snapshot["internal_force_normal"],
        acceleration_normal=trigger_snapshot["acceleration_normal"],
        normal_velocity=trigger_snapshot["normal_velocity"],
        stop_force_required_normal=trigger_snapshot["stop_force_required_normal"],
        topk_indices=trigger_snapshot["topk_indices"],
        rigid_shape=np.asarray(str(trigger_snapshot["rigid_shape"])),
    )
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return npz_path, json_path


def _render_force_diagnostic_snapshot(
    model: newton.Model,
    meta: dict[str, Any],
    args: argparse.Namespace,
    device: str,
    snapshot: dict[str, Any],
    out_path: Path,
) -> Path:
    import newton.viewer  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(args.screen_width)
    height = int(args.screen_height)
    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, headless=True)
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
        state = model.state()
        if state.particle_qd is not None:
            state.particle_qd.zero_()
        if state.body_qd is not None:
            state.body_qd.zero_()

        cloth_edges = np.asarray(meta["render_edges"], dtype=np.int32)
        starts_wp = wp.empty(len(cloth_edges), dtype=wp.vec3, device=device) if cloth_edges.size else None
        ends_wp = wp.empty(len(cloth_edges), dtype=wp.vec3, device=device) if cloth_edges.size else None

        topk = np.asarray(snapshot["topk_indices"], dtype=np.int32)
        topk_q = np.asarray(snapshot["particle_q"], dtype=np.float32)[topk]
        arrow_specs = [
            ("diag_normal", np.asarray(snapshot["outward_normal_world"], dtype=np.float32)[topk], (1.0, 1.0, 1.0)),
            ("diag_external", np.asarray(snapshot["f_external_total"], dtype=np.float32)[topk], (0.95, 0.25, 0.25)),
            ("diag_spring", np.asarray(snapshot["f_spring"], dtype=np.float32)[topk], (0.25, 0.45, 0.95)),
            ("diag_accel", np.asarray(snapshot["a_total"], dtype=np.float32)[topk], (0.15, 0.85, 0.25)),
        ]

        with temporary_particle_radius_override(model, render_radii):
            state.particle_q.assign(np.asarray(snapshot["particle_q"], dtype=np.float32))
            if state.body_q is not None and np.asarray(snapshot["body_q"]).size:
                state.body_q.assign(np.asarray(snapshot["body_q"], dtype=np.float32))

            viewer.begin_frame(float(snapshot["sim_time"]))
            viewer.log_state(state)

            if args.render_springs and cloth_edges.size and starts_wp is not None and ends_wp is not None:
                q_obj = np.asarray(snapshot["particle_q"], dtype=np.float32)
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

            max_len = max(float(args.force_arrow_scale), 1.0e-6)
            for name, vectors, color in arrow_specs:
                if topk_q.shape[0] == 0:
                    continue
                if name == "diag_normal":
                    scaled = vectors * max_len
                else:
                    norms = np.linalg.norm(vectors, axis=1)
                    peak = float(np.max(norms)) if norms.size else 0.0
                    if peak <= 1.0e-12:
                        scaled = np.zeros_like(vectors, dtype=np.float32)
                    else:
                        scaled = (vectors / peak * max_len).astype(np.float32, copy=False)
                starts = wp.array(topk_q.astype(np.float32, copy=False), dtype=wp.vec3, device=device)
                ends = wp.array((topk_q + scaled).astype(np.float32, copy=False), dtype=wp.vec3, device=device)
                viewer.log_lines(f"/demo/{name}", starts, ends, color, width=0.012, hidden=False)

            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            frame = overlay_text_lines_rgb(
                frame,
                [
                    "FORCE DIAGNOSTIC SNAPSHOT",
                    f"snapshot = {args.force_snapshot_frame}",
                    f"trigger substep = {int(snapshot['global_substep_index'])}",
                    "white=normal red=external blue=spring green=acceleration",
                ],
                font_size=int(args.label_font_size),
            )
            Image.fromarray(frame, mode="RGB").save(out_path)
    finally:
        try:
            viewer.close()
        except Exception:
            pass
    return out_path


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
    use_decoupled_shape_materials = bool(args.decouple_shape_materials) or (
        not np.isclose(float(ir_obj.get("weight_scale", 1.0)), 1.0)
    )
    force_diagnostic_enabled = bool(args.force_diagnostic)
    use_manual_force_path = bool(use_decoupled_shape_materials) or force_diagnostic_enabled
    force_eps = 1.0e-8
    trigger_snapshot: dict[str, Any] | None = None
    render_snapshot: dict[str, Any] | None = None
    pending_snapshot_after_step = False
    previous_max_external_norm = 0.0
    stop_requested = False
    diagnostic_rollout_truncated = False
    global_substep_index = 0
    particle_mass_diag = np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].copy()
    particle_radius_diag = model.particle_radius.numpy().astype(np.float32)[:n_obj].copy()
    gravity_vec_diag = np.asarray(gravity_vec, dtype=np.float32).reshape(3)
    if force_diagnostic_enabled:
        _resolve_force_dump_dir(args).mkdir(parents=True, exist_ok=True)

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

        for substep_index_in_frame in range(substeps):
            state_in.clear_forces()

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if contacts is not None:
                model.collide(state_in, contacts)

            if not use_manual_force_path:
                solver.step(state_in, state_out, control, contacts, sim_dt)
            else:
                particle_f = state_in.particle_f if state_in.particle_count else None
                body_f = state_in.body_f if state_in.body_count else None
                body_f_work = body_f
                if body_f is not None and model.joint_count and control.joint_f is not None:
                    body_f_work = wp.clone(body_f)
                if particle_f is None:
                    raise RuntimeError("Force diagnostics require particle forces, but particle_f is not allocated.")

                eval_spring_forces(model, state_in, particle_f)
                f_spring_np = None
                f_internal_total_np = None
                f_before_particle_body_np = None
                if force_diagnostic_enabled:
                    f_spring_np = particle_f.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                eval_triangle_forces(model, state_in, control, particle_f)
                eval_bending_forces(model, state_in, particle_f)
                eval_tetrahedra_forces(model, state_in, control, particle_f)
                if force_diagnostic_enabled:
                    f_internal_total_np = particle_f.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                eval_body_joint_forces(
                    model, state_in, control, body_f_work, solver.joint_attach_ke, solver.joint_attach_kd
                )
                if solver.enable_tri_contact:
                    eval_triangle_contact_forces(model, state_in, particle_f)
                if force_diagnostic_enabled:
                    f_before_particle_body_np = particle_f.numpy().astype(np.float32, copy=False)[:n_obj].copy()

                if use_decoupled_shape_materials:
                    _assign_shape_material_triplet(
                        model,
                        meta["shape_material_ke_base"],
                        meta["shape_material_kd_base"],
                        meta["shape_material_kf_base"],
                    )
                eval_body_contact_forces(
                    model, state_in, contacts, friction_smoothing=solver.friction_smoothing, body_f_out=body_f_work
                )

                if use_decoupled_shape_materials:
                    _assign_shape_material_triplet(
                        model,
                        meta["shape_material_ke_scaled"],
                        meta["shape_material_kd_scaled"],
                        meta["shape_material_kf_scaled"],
                    )
                eval_particle_body_contact_forces(
                    model, state_in, contacts, particle_f, body_f_work, body_f_in_world_frame=False
                )
                if force_diagnostic_enabled:
                    f_after_particle_body_np = particle_f.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                    if f_spring_np is None or f_internal_total_np is None or f_before_particle_body_np is None:
                        raise RuntimeError("Force diagnostic bookkeeping is incomplete inside the manual force path.")
                    f_external_total_np = (
                        f_after_particle_body_np - f_before_particle_body_np
                    ).astype(np.float32, copy=False)
                    current_max_external_norm = float(
                        np.max(np.linalg.norm(f_external_total_np, axis=1))
                    ) if f_external_total_np.size else 0.0
                    sim_time = (frame * substeps + substep_index_in_frame) * sim_dt
                    if (
                        trigger_snapshot is None
                        and previous_max_external_norm <= force_eps
                        and current_max_external_norm > force_eps
                    ):
                        particle_q_np = state_in.particle_q.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                        particle_qd_np = state_in.particle_qd.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                        body_q_np = (
                            state_in.body_q.numpy().astype(np.float32).copy()
                            if state_in.body_q is not None
                            else np.zeros((0, 7), dtype=np.float32)
                        )
                        body_qd_np = (
                            state_in.body_qd.numpy().astype(np.float32).copy()
                            if state_in.body_qd is not None
                            else np.zeros((0, 6), dtype=np.float32)
                        )
                        trigger_snapshot = _capture_force_snapshot(
                            frame_index=frame,
                            substep_index_in_frame=substep_index_in_frame,
                            global_substep_index=global_substep_index,
                            sim_time=sim_time,
                            sim_dt=sim_dt,
                            particle_q=particle_q_np,
                            particle_qd=particle_qd_np,
                            body_q=body_q_np,
                            body_qd=body_qd_np,
                            particle_radius=particle_radius_diag,
                            mass=particle_mass_diag,
                            gravity_vec=gravity_vec_diag,
                            f_spring=f_spring_np,
                            f_internal_total=f_internal_total_np,
                            f_external_total=f_external_total_np,
                            meta=meta,
                            force_topk=int(args.force_topk),
                            force_eps=force_eps,
                        )
                        if str(args.force_snapshot_frame) == "trigger":
                            render_snapshot = trigger_snapshot
                        else:
                            pending_snapshot_after_step = True
                    previous_max_external_norm = current_max_external_norm

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

            global_substep_index += 1
            if pending_snapshot_after_step and trigger_snapshot is not None and render_snapshot is None:
                particle_q_np = state_in.particle_q.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                particle_qd_np = state_in.particle_qd.numpy().astype(np.float32, copy=False)[:n_obj].copy()
                body_q_np = (
                    state_in.body_q.numpy().astype(np.float32).copy()
                    if state_in.body_q is not None
                    else np.zeros((0, 7), dtype=np.float32)
                )
                body_qd_np = (
                    state_in.body_qd.numpy().astype(np.float32).copy()
                    if state_in.body_qd is not None
                    else np.zeros((0, 6), dtype=np.float32)
                )
                render_snapshot = _capture_force_snapshot(
                    frame_index=frame,
                    substep_index_in_frame=substep_index_in_frame + 1,
                    global_substep_index=global_substep_index,
                    sim_time=(frame * substeps + substep_index_in_frame + 1) * sim_dt,
                    sim_dt=sim_dt,
                    particle_q=particle_q_np,
                    particle_qd=particle_qd_np,
                    body_q=body_q_np,
                    body_qd=body_qd_np,
                    particle_radius=particle_radius_diag,
                    mass=particle_mass_diag,
                    gravity_vec=gravity_vec_diag,
                    f_spring=trigger_snapshot["f_spring"],
                    f_internal_total=trigger_snapshot["f_internal_total"],
                    f_external_total=trigger_snapshot["f_external_total"],
                    meta=meta,
                    force_topk=int(args.force_topk),
                    force_eps=force_eps,
                )
                pending_snapshot_after_step = False
            if force_diagnostic_enabled and trigger_snapshot is not None and bool(args.stop_after_diagnostic):
                if render_snapshot is not None:
                    stop_requested = True
                    diagnostic_rollout_truncated = True
            if stop_requested:
                break

        if stop_requested:
            break
        if frame == 49 or frame == 99 or frame == 199 or frame == n_frames - 1:
            print(f"  frame {frame + 1}/{n_frames}", flush=True)

    wall_time = time.perf_counter() - t0
    force_diagnostic: dict[str, Any] | None = None
    if force_diagnostic_enabled:
        if trigger_snapshot is None:
            raise RuntimeError(
                "Force diagnostic requested, but no trigger substep was found. "
                "Try increasing --frames or using a case that reaches cloth-rigid contact."
            )
        summary_payload = _force_summary_payload(trigger_snapshot, eps=force_eps)
        npz_path, json_path = _write_force_diagnostic_outputs(args, trigger_snapshot, summary_payload)
        snapshot_png_path = _resolve_force_dump_dir(args) / "force_diag_trigger_snapshot.png"
        _render_force_diagnostic_snapshot(
            model,
            meta,
            args,
            device,
            render_snapshot if render_snapshot is not None else trigger_snapshot,
            snapshot_png_path,
        )
        force_diagnostic = {
            "enabled": True,
            "snapshot_frame_mode": str(args.force_snapshot_frame),
            "trigger_substep_global": int(trigger_snapshot["global_substep_index"]),
            "trigger_frame_index": int(trigger_snapshot["frame_index"]),
            "trigger_substep_index_in_frame": int(trigger_snapshot["substep_index_in_frame"]),
            "force_eps": float(force_eps),
            "npz_path": str(npz_path),
            "summary_json_path": str(json_path),
            "snapshot_png_path": str(snapshot_png_path),
            "contact_node_count": int(summary_payload["contact_node_count"]),
            "dominant_issue_guess": str(summary_payload["dominant_issue_guess"]),
            "topk_particle_count": int(len(summary_payload["topk_particle_records"])),
            "rollout_truncated": bool(diagnostic_rollout_truncated),
        }
    print(f"  Simulation done: {len(particle_q_all)} recorded frames in {wall_time:.1f}s", flush=True)
    result = {
        "particle_q_all": np.stack(particle_q_all, axis=0),
        "particle_q_object": np.stack(particle_q_object, axis=0),
        "body_q": np.stack(body_q, axis=0),
        "body_vel": np.stack(body_vel, axis=0),
        "sim_dt": float(sim_dt),
        "substeps": int(substeps),
        "wall_time": float(wall_time),
        "use_decoupled_shape_materials": bool(use_decoupled_shape_materials),
    }
    if force_diagnostic is not None:
        result["force_diagnostic"] = force_diagnostic
    return result


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
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{out_mp4.stem}_",
        suffix=".mp4",
        dir="/tmp",
    )
    os.close(tmp_fd)
    tmp_out_mp4 = Path(tmp_name)

    import newton.viewer  # noqa: PLC0415

    def _first_rigid_contact_frame() -> int | None:
        cloth_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
        body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
        if body_q.ndim != 3 or body_q.shape[1] == 0:
            return None

        radius = model.particle_radius.numpy().astype(np.float32)[: cloth_q.shape[1]]
        rigid_kind = str(meta.get("rigid_shape", "bunny"))
        if rigid_kind == "box":
            hx, hy, hz = [float(v) for v in np.asarray(meta["box_half_extents"]).ravel()]
            for frame_idx in range(cloth_q.shape[0]):
                pos = body_q[frame_idx, 0, :3].astype(np.float32)
                quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
                rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
                local = (cloth_q[frame_idx] - pos[None, :]) @ rot
                sdf = _box_signed_distance(local.astype(np.float32, copy=False), hx, hy, hz)
                if np.any(radius - sdf > 0.0):
                    return int(frame_idx)
            return None

        faces = np.asarray(meta.get("mesh_tri_indices", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        verts_local = np.asarray(meta.get("mesh_verts_local", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if not faces.size or not verts_local.size:
            return None

        scale = float(meta["mesh_scale"])
        for frame_idx in range(cloth_q.shape[0]):
            pos = body_q[frame_idx, 0, :3].astype(np.float32)
            quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
            rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
            verts = (verts_local * scale) @ rot.T + pos[None, :]
            tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            _, dist, _ = trimesh.proximity.closest_point(tri, cloth_q[frame_idx].astype(np.float64))
            penetration = radius - dist.astype(np.float32, copy=False)
            if np.any(penetration > 0.0):
                return int(frame_idx)
        return None

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
        str(tmp_out_mp4),
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
            apply_viewer_shape_colors(
                viewer,
                model,
                extra_rules=[
                    (
                        lambda name: "bunny" in name
                        or (meta.get("rigid_shape") == "box" and "shape_1" in name),
                        (0.88, 0.35, 0.28),
                    )
                ],
            )
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
            first_contact_frame = _first_rigid_contact_frame()
            render_end_frame = n_sim_frames - 1
            if first_contact_frame is not None and float(args.post_contact_video_seconds) > 0.0:
                sim_window = float(args.post_contact_video_seconds) / max(float(args.slowdown), 1.0e-6)
                extra_frames = int(round(sim_window / max(sim_frame_dt, 1.0e-12)))
                render_end_frame = min(n_sim_frames - 1, int(first_contact_frame) + max(0, extra_frames))
            sim_duration = max(float(render_end_frame) * sim_frame_dt, 0.0)
            video_duration = sim_duration * max(float(args.slowdown), 1.0e-6)
            n_out_frames = max(1, int(round(video_duration * fps_out)))
            if n_out_frames == 1 or sim_duration <= 0.0:
                render_indices = np.zeros((1,), dtype=np.int32)
            else:
                sample_times = np.linspace(0.0, sim_duration, n_out_frames, endpoint=True, dtype=np.float64)
                render_indices = np.clip(np.rint(sample_times / sim_frame_dt).astype(np.int32), 0, render_end_frame)

            sim_data["first_rigid_contact_frame"] = -1 if first_contact_frame is None else int(first_contact_frame)
            sim_data["render_end_frame"] = int(render_end_frame)
            sim_data["rendered_frame_count"] = int(n_out_frames)
            sim_data["video_duration_target_sec"] = float(video_duration)

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
            tmp_out_mp4.replace(out_mp4)
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        if tmp_out_mp4.exists():
            tmp_out_mp4.unlink()
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
    if "rendered_frame_count" in sim_data:
        rendered_frame_count = int(sim_data["rendered_frame_count"])
    if "video_duration_target_sec" in sim_data:
        video_duration = float(sim_data["video_duration_target_sec"])
    object_mass = float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].mean())
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
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
    summary = {
        "experiment": "cloth_bunny_drop_object_only",
        "self_collision_case": "off",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "drop_height_m": float(args.drop_height),
        "cloth_shift_x_m": float(args.cloth_shift_x),
        "cloth_shift_y_m": float(args.cloth_shift_y),
        "object_mass_per_particle": object_mass,
        "auto_set_weight_target_kg": (
            None if args.auto_set_weight is None else float(args.auto_set_weight)
        ),
        "weight_scale": float(ir_obj.get("weight_scale", 1.0)),
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
        "first_rigid_contact_frame": (
            None
            if int(sim_data.get("first_rigid_contact_frame", -1)) < 0
            else int(sim_data["first_rigid_contact_frame"])
        ),
        "render_end_frame": int(sim_data.get("render_end_frame", cloth_q.shape[0] - 1)),
        "post_contact_video_seconds": float(args.post_contact_video_seconds),
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
        "use_decoupled_shape_materials": bool(
            sim_data.get("use_decoupled_shape_materials", False)
        ),
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
    force_diagnostic = sim_data.get("force_diagnostic")
    if isinstance(force_diagnostic, dict):
        summary.update(
            {
                "force_diagnostic_enabled": bool(force_diagnostic.get("enabled", False)),
                "force_snapshot_frame_mode": str(force_diagnostic.get("snapshot_frame_mode", "trigger")),
                "force_diagnostic_trigger_substep_global": int(
                    force_diagnostic.get("trigger_substep_global", -1)
                ),
                "force_diagnostic_trigger_frame_index": int(
                    force_diagnostic.get("trigger_frame_index", -1)
                ),
                "force_diagnostic_trigger_substep_index_in_frame": int(
                    force_diagnostic.get("trigger_substep_index_in_frame", -1)
                ),
                "force_diagnostic_contact_node_count": int(force_diagnostic.get("contact_node_count", 0)),
                "force_diagnostic_topk_particle_count": int(force_diagnostic.get("topk_particle_count", 0)),
                "force_diagnostic_dominant_issue_guess": str(
                    force_diagnostic.get("dominant_issue_guess", "undetermined")
                ),
                "force_diagnostic_force_eps": float(force_diagnostic.get("force_eps", 0.0)),
                "force_diagnostic_npz": str(force_diagnostic.get("npz_path", "")),
                "force_diagnostic_summary_json": str(force_diagnostic.get("summary_json_path", "")),
                "force_diagnostic_snapshot_png": str(force_diagnostic.get("snapshot_png_path", "")),
                "rollout_truncated_for_force_diagnostic": bool(
                    force_diagnostic.get("rollout_truncated", False)
                ),
            }
        )
    return summary


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
    skip_full_rollout_render = bool(args.force_diagnostic) and bool(args.stop_after_diagnostic)
    if skip_full_rollout_render and not bool(args.skip_render):
        print("Skipping full rollout video because --stop-after-diagnostic is active.", flush=True)
    if not bool(args.skip_render) and not skip_full_rollout_render:
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
    if args.auto_set_weight is not None:
        _maybe_autoset_mass_spring_scale(
            args,
            raw_ir,
            target_total_mass=float(args.auto_set_weight),
        )

    run_case(args, raw_ir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
