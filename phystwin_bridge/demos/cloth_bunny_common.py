#!/usr/bin/env python3
"""Shared helpers for cloth-vs-bunny SemiImplicit demos.

These helpers are intentionally demo-side only. They do not modify Newton core
behavior under ``Newton/newton``; they only prepare IR, scale bridge-side
contact knobs, and package summary metadata consistently across the ON/OFF
cloth scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from bridge_bootstrap import newton_import_ir

BRIDGE_ROOT = Path(__file__).resolve().parents[1]


def _default_cloth_ir() -> Path:
    return (
        BRIDGE_ROOT
        / "ir"
        / "blue_cloth_double_lift_around"
        / "phystwin_ir_v2_bf_strict.npz"
    )


def load_ir(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _mass_tag(value: float) -> str:
    text = f"{float(value):g}"
    text = text.replace("-", "neg")
    return "m" + text.replace(".", "p")


def _validate_scaling_args(args: argparse.Namespace) -> None:
    auto_set_weight = getattr(args, "auto_set_weight", None)
    if auto_set_weight is not None and float(auto_set_weight) <= 0.0:
        raise ValueError(f"--auto-set-weight must be > 0, got {auto_set_weight}")
    if auto_set_weight is not None and args.object_mass is not None:
        raise ValueError(
            "--auto-set-weight cannot be used together with --object-mass. "
            "Use only one source of deformable mass control."
        )
    if auto_set_weight is not None and args.mass_spring_scale is not None:
        raise ValueError(
            "--auto-set-weight cannot be used together with --mass-spring-scale. "
            "Use only one source of deformable mass scaling."
        )
    if auto_set_weight is not None and not np.isclose(float(args.spring_ke_scale), 1.0):
        raise ValueError(
            "--auto-set-weight cannot be used together with --spring-ke-scale. "
            "The auto-computed weight_scale already controls spring_ke."
        )
    if auto_set_weight is not None and not np.isclose(float(args.spring_kd_scale), 1.0):
        raise ValueError(
            "--auto-set-weight cannot be used together with --spring-kd-scale. "
            "The auto-computed weight_scale already controls spring_kd."
        )
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


def _effective_object_mass_array(
    ir_demo: dict[str, Any], args: argparse.Namespace, n_obj: int
) -> np.ndarray:
    mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()[:n_obj]
    if args.object_mass is not None:
        mass.fill(float(args.object_mass))
    elif args.mass_spring_scale is not None:
        mass *= float(args.mass_spring_scale)
    return mass


def _mass_ratio_scale(original_mass: np.ndarray, current_mass: np.ndarray) -> float:
    original_total = float(np.asarray(original_mass, dtype=np.float32).sum())
    current_total = float(np.asarray(current_mass, dtype=np.float32).sum())
    if original_total <= 0.0:
        raise ValueError(f"Original total mass must be > 0, got {original_total}")
    return current_total / original_total


def _effective_spring_scales(
    ir_demo: dict[str, Any], args: argparse.Namespace
) -> tuple[float, float]:
    weight_scale = float(ir_demo.get("weight_scale", 1.0))
    return (
        float(args.spring_ke_scale) * weight_scale,
        float(args.spring_kd_scale) * weight_scale,
    )


def _apply_particle_contact_scaling(
    model: Any, *, weight_scale: float
) -> None:
    """Scales particle-particle contact for any spring-mass object interaction.

    This affects:
    - self-collision / particle-particle contact in cloth demos
    - cross-object particle contact in rope/sloth demos when they reuse
      ``model.particle_ke/kd/kf``

    It does not touch rigid-rigid or rigid-ground contacts.
    """

    alpha = float(weight_scale)
    if np.isclose(alpha, 1.0):
        return
    if alpha <= 0.0:
        raise ValueError(f"weight_scale must be > 0, got {alpha}")
    model.particle_ke = float(model.particle_ke * alpha)
    model.particle_kd = float(model.particle_kd * alpha)
    model.particle_kf = float(model.particle_kf * alpha)


def _apply_shape_contact_scaling(
    model: Any, args: argparse.Namespace, *, weight_scale: float = 1.0
) -> None:
    """Scales the particle-shape penalty path used by the bridge demos.

    Physics note:
    - ``soft_contact_ke`` controls the normal elastic push-back once a particle
      overlaps the rigid surface.
    - ``soft_contact_kd`` and ``soft_contact_kf`` control dissipative normal
      damping and tangential/frictional stabilization.
    - We scale the per-shape material arrays as well so the effective contact
      pair coefficients stay consistent between the global soft-contact path
      and the bunny/ground-specific material response.
    """

    explicit_alpha = args.shape_contact_scale
    base_alpha = 1.0 if explicit_alpha is None else float(explicit_alpha)
    auto_alpha = float(weight_scale)
    alpha = base_alpha * auto_alpha
    if (
        explicit_alpha is None
        and np.isclose(auto_alpha, 1.0)
        and args.ground_shape_contact_scale is None
        and args.bunny_shape_contact_scale is None
    ):
        return
    if alpha <= 0.0:
        raise ValueError(f"shape-contact-scale must be > 0, got {alpha}")
    damping_mult = float(args.shape_contact_damping_multiplier)
    if damping_mult <= 0.0:
        raise ValueError(
            f"--shape-contact-damping-multiplier must be > 0, got {damping_mult}"
        )

    # Global soft-contact terms only affect particle-shape contact, so they can
    # safely follow the spring-mass weight scale without touching rigid-rigid or
    # rigid-ground contact behavior.
    model.soft_contact_ke = float(model.soft_contact_ke * alpha)
    model.soft_contact_kd = float(model.soft_contact_kd * alpha * damping_mult)
    model.soft_contact_kf = float(model.soft_contact_kf * alpha * damping_mult)

    labels = [str(label).lower() for label in list(model.shape_label)]
    has_ground = any(("ground" in label or "plane" in label) for label in labels)

    ground_alpha = (
        base_alpha
        if args.ground_shape_contact_scale is None
        else float(args.ground_shape_contact_scale)
    )
    ground_dmult = (
        (damping_mult if explicit_alpha is not None else 1.0)
        if args.ground_shape_contact_damping_multiplier is None
        else float(args.ground_shape_contact_damping_multiplier)
    )
    bunny_alpha = (
        base_alpha
        if args.bunny_shape_contact_scale is None
        else float(args.bunny_shape_contact_scale)
    )
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
                scale = np.float32(ground_alpha * auto_alpha)
                dmult = np.float32(ground_dmult)
            elif "bunny" in label or (
                bool(args.add_bunny)
                and (not ("ground" in label or "plane" in label))
                and has_ground
            ):
                scale = np.float32(bunny_alpha * auto_alpha)
                dmult = np.float32(bunny_dmult)
            vals[idx] *= scale
            if arr_name != "shape_material_ke":
                vals[idx] *= dmult
        arr.assign(vals)


def _copy_object_only_ir(
    ir: dict[str, np.ndarray], args: argparse.Namespace
) -> dict[str, Any]:
    """Drops controller particles and keeps only the deformable object state."""

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
    original_mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()[:n_obj]
    weight_scale = _mass_ratio_scale(original_mass, mass)

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = mass
    ir_demo["weight_scale"] = float(weight_scale)
    ir_demo["original_total_object_mass"] = float(original_mass.sum())
    ir_demo["current_total_object_mass"] = float(mass.sum())
    ir_demo["collision_radius"] = np.asarray(
        ir_demo["collision_radius"], dtype=np.float32
    ).copy()[:n_obj]
    ir_demo["num_object_points"] = np.asarray(n_obj, dtype=np.int32)
    ir_demo["reverse_z"] = np.asarray(False)

    edges = np.asarray(ir_demo["spring_edges"], dtype=np.int32)
    keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
    ir_demo["spring_edges"] = edges[keep].astype(np.int32, copy=True)
    for key in (
        "spring_ke",
        "spring_kd",
        "spring_rest_length",
        "spring_y",
        "spring_y_raw",
    ):
        if key in ir_demo:
            ir_demo[key] = (
                np.asarray(ir_demo[key], dtype=np.float32).copy().ravel()[keep]
            )

    ir_demo.pop("controller_idx", None)
    ir_demo.pop("controller_traj", None)
    return ir_demo


def _maybe_autoset_mass_spring_scale(
    args: argparse.Namespace,
    raw_ir: dict[str, np.ndarray],
    *,
    target_total_mass: float,
) -> None:
    """Auto-fills ``mass_spring_scale`` from a target total object mass."""

    auto_set_weight = getattr(args, "auto_set_weight", None)
    if auto_set_weight is not None:
        target_total_mass = float(auto_set_weight)
    if args.mass_spring_scale is not None or args.object_mass is not None:
        return
    n_obj = int(np.asarray(raw_ir["num_object_points"]).ravel()[0])
    total_mass = float(np.asarray(raw_ir["mass"], dtype=np.float32)[:n_obj].sum())
    if total_mass <= 0.0:
        raise ValueError(f"IR total object mass must be > 0, got {total_mass}")
    args.mass_spring_scale = float(target_total_mass) / total_mass


def _reconstruct_cloth_surface_mesh(
    x0: np.ndarray,
    spring_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstructs a triangle mesh from planar cloth vertices + spring graph.

    The PhysTwin IR stores object vertices and spring edges but not explicit
    triangle faces. For Newton's official cloth self-collision path we need a
    surface mesh, so we:

    1. project the nearly planar cloth to 2D using PCA,
    2. run Delaunay in that intrinsic 2D chart,
    3. keep only triangles whose three edges already exist in the spring graph.

    This keeps the reconstructed cloth connectivity consistent with the bridge
    topology instead of inventing arbitrary cross-cloth links.
    """

    points = np.asarray(x0, dtype=np.float64)
    edges = np.asarray(spring_edges, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"x0 must have shape (N, 3), got {points.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"spring_edges must have shape (E, 2), got {edges.shape}"
        )
    if points.shape[0] < 4:
        raise ValueError("Need at least four points to reconstruct a cloth mesh")

    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vh[:2].T
    tri = Delaunay(uv)
    edge_set = {tuple(sorted((int(a), int(b)))) for a, b in edges}

    faces: list[tuple[int, int, int]] = []
    for face in np.asarray(tri.simplices, dtype=np.int32):
        a, b, c = map(int, face)
        tri_edges = (
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((a, c))),
        )
        if all(edge in edge_set for edge in tri_edges):
            faces.append((a, b, c))

    if not faces:
        raise ValueError("Failed to reconstruct any cloth faces from spring graph")

    faces_np = np.asarray(faces, dtype=np.int32)
    used = np.unique(faces_np.reshape(-1))
    remap = -np.ones(points.shape[0], dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    points_used = points[used].astype(np.float32, copy=False)
    faces_used = remap[faces_np].astype(np.int32, copy=False)
    return points_used, faces_used, used.astype(np.int32, copy=False)


def _cloth_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Returns the total triangle area of a reconstructed cloth mesh."""

    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int32)
    area = 0.0
    for i, j, k in tri:
        area += 0.5 * np.linalg.norm(np.cross(verts[j] - verts[i], verts[k] - verts[i]))
    return float(area)


def _box_signed_distance(
    local_points: np.ndarray, hx: float, hy: float, hz: float
) -> np.ndarray:
    """Returns the signed distance from points to an axis-aligned box.

    Positive values are outside the box. Negative values are inside, so
    ``radius - sdf`` is a simple penetration proxy for a particle sphere of
    radius ``radius`` against the box surface.
    """

    q = np.abs(local_points) - np.array([hx, hy, hz], dtype=np.float32)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside
