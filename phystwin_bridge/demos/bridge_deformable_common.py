#!/usr/bin/env python3
"""Shared bridge-side helpers for deformable-object demos.

These utilities were previously imported from individual demo files such as
``demo_rope_bunny_drop.py``. Keeping them here avoids demo-to-demo dependency
chains and makes the demos easier to evolve independently.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from bridge_bootstrap import newton, newton_import_ir


@wp.kernel
def _apply_drag_correction_ignore_axis(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    n_object: int,
    dt: float,
    damping: float,
    axis_unit: wp.vec3,
):
    tid = wp.tid()
    if tid >= n_object:
        return
    v = particle_qd[tid]
    v_axis = axis_unit * wp.dot(v, axis_unit)
    v_ortho = v - v_axis
    scale = wp.exp(-dt * damping)
    particle_q[tid] = particle_q[tid] - v_ortho * dt * (1.0 - scale)
    particle_qd[tid] = v_axis + v_ortho * scale


def load_ir(path: Path) -> dict[str, np.ndarray]:
    """Load a PhysTwin IR bundle through the bridge importer."""
    return newton_import_ir.load_ir(path.resolve())


def _validate_scaling_args(args: argparse.Namespace) -> None:
    if args.auto_set_weight is not None and float(args.auto_set_weight) <= 0.0:
        raise ValueError(f"--auto-set-weight must be > 0, got {args.auto_set_weight}")
    if args.auto_set_weight is not None and not np.isclose(float(args.object_mass), 1.0):
        raise ValueError(
            "--auto-set-weight cannot be used together with a non-default --object-mass. "
            "Use only one source of deformable mass control."
        )
    if args.auto_set_weight is not None and args.mass_spring_scale is not None:
        raise ValueError(
            "--auto-set-weight cannot be used together with --mass-spring-scale. "
            "Use only one source of deformable mass scaling."
        )
    if args.auto_set_weight is not None and not np.isclose(float(args.spring_ke_scale), 1.0):
        raise ValueError(
            "--auto-set-weight cannot be used together with --spring-ke-scale. "
            "The auto-computed weight_scale already controls spring_ke."
        )
    if args.auto_set_weight is not None and not np.isclose(float(args.spring_kd_scale), 1.0):
        raise ValueError(
            "--auto-set-weight cannot be used together with --spring-kd-scale. "
            "The auto-computed weight_scale already controls spring_kd."
        )
    if args.mass_spring_scale is None:
        return
    scale = float(args.mass_spring_scale)
    if scale <= 0.0:
        raise ValueError(f"--mass-spring-scale must be > 0, got {scale}")
    if args.object_mass is not None and not np.isclose(float(args.object_mass), 1.0):
        raise ValueError(
            "--mass-spring-scale cannot be used together with a non-default --object-mass. "
            "Use either the unified scale or an absolute per-particle mass override."
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
    if args.mass_spring_scale is not None:
        mass *= float(args.mass_spring_scale)
    elif args.object_mass is not None:
        mass.fill(float(args.object_mass))
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


def _maybe_autoset_mass_spring_scale(
    args: argparse.Namespace, raw_ir: dict[str, np.ndarray]
) -> None:
    if args.auto_set_weight is None or args.mass_spring_scale is not None:
        return
    n_obj = int(np.asarray(raw_ir["num_object_points"]).ravel()[0])
    total_mass = float(np.asarray(raw_ir["mass"], dtype=np.float32)[:n_obj].sum())
    if total_mass <= 0.0:
        raise ValueError(f"IR total object mass must be > 0, got {total_mass}")
    args.mass_spring_scale = float(args.auto_set_weight) / total_mass


def _assign_shape_material_triplet(
    model: Any, ke: np.ndarray, kd: np.ndarray, kf: np.ndarray
) -> None:
    model.shape_material_ke.assign(np.asarray(ke, dtype=np.float32))
    model.shape_material_kd.assign(np.asarray(kd, dtype=np.float32))
    model.shape_material_kf.assign(np.asarray(kf, dtype=np.float32))


def load_bunny_mesh(asset_name: str, prim_path: str):
    try:
        import newton.examples  # noqa: PLC0415
        import newton.usd  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError("Bunny mesh requires newton.examples, newton.usd, and pxr") from exc

    asset_path = Path(newton.examples.get_asset(asset_name))
    stage = Usd.Stage.Open(str(asset_path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path} in {asset_path}")
    mesh = newton.usd.get_mesh(prim)
    points = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if indices.size % 3 != 0:
        raise ValueError(f"Bunny mesh index length must be multiple of 3, got {indices.size}")
    tri_indices = indices.reshape(-1, 3)
    e01 = tri_indices[:, [0, 1]]
    e12 = tri_indices[:, [1, 2]]
    e20 = tri_indices[:, [2, 0]]
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0).astype(np.int32, copy=False)
    return mesh, points, tri_indices, all_edges, str(asset_path)


def quat_to_rotmat(
    q_xyzw: list[float] | tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = max((x * x + y * y + z * z + w * w) ** 0.5, 1.0e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


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
    original_mass = np.asarray(ir_demo["mass"], dtype=np.float32).copy()[:n_obj]
    weight_scale = _mass_ratio_scale(original_mass, mass)

    ir_demo["x0"] = x0
    ir_demo["v0"] = v0
    ir_demo["mass"] = mass
    ir_demo["weight_scale"] = float(weight_scale)
    ir_demo["original_total_object_mass"] = float(original_mass.sum())
    ir_demo["current_total_object_mass"] = float(mass.sum())
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


__all__ = [
    "_apply_drag_correction_ignore_axis",
    "_assign_shape_material_triplet",
    "_copy_object_only_ir",
    "_effective_spring_scales",
    "_maybe_autoset_mass_spring_scale",
    "_validate_scaling_args",
    "load_bunny_mesh",
    "load_ir",
    "newton",
    "newton_import_ir",
    "quat_to_rotmat",
]
