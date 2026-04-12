#!/usr/bin/env python3
"""Runtime setup helpers for cloth+bunny rollouts and realtime examples."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from bridge_bootstrap import newton, newton_import_ir


REQUIRED_RUNTIME_DEVICE = "cuda:0"


@dataclass(slots=True)
class ClothBunnyExampleAssets:
    """Prepared realtime-playground runtime assets."""

    device: str
    ir_obj: dict[str, Any]
    model: Any
    meta: dict[str, Any]
    n_obj: int
    cfg: Any
    solver: Any
    particle_grid: Any
    search_radius: float
    drag: float
    gravity_axis: np.ndarray | None
    physical_particle_radius: np.ndarray
    visual_particle_radius: np.ndarray


def resolve_workspace_path(path: Path, workspace_root: Path) -> Path:
    """Resolve a repo path relative to cwd first, then the workspace root."""

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (workspace_root / raw).resolve()


def _maybe_autoset_alpha(args: argparse.Namespace, raw_ir: dict[str, np.ndarray]) -> None:
    if args.mass_spring_scale is not None:
        return
    n_obj = int(np.asarray(raw_ir["num_object_points"]).ravel()[0])
    total_mass = float(np.asarray(raw_ir["mass"], dtype=np.float32)[:n_obj].sum())
    if total_mass <= 0.0:
        raise ValueError(f"IR total object mass must be > 0, got {total_mass}")
    args.mass_spring_scale = float(args.target_total_mass) / total_mass


def prepare_ir_for_runtime(module, args: argparse.Namespace) -> dict[str, Any]:
    """Load and scale the runtime IR for the selected mode."""

    raw_ir = module.load_ir(args.ir)
    _maybe_autoset_alpha(args, raw_ir)
    if args.shape_contact_scale is None:
        args.shape_contact_scale = float(args.mass_spring_scale)
    ir_obj = module._copy_object_only_ir(raw_ir, args)
    if "contact_collision_dist" in ir_obj:
        scaled_dist = float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]) * float(args.contact_dist_scale)
        if args.mode == "on":
            scaled_dist *= float(args.on_contact_dist_scale)
        ir_obj["contact_collision_dist"] = np.asarray(scaled_dist, dtype=np.float32)
    return ir_obj


def require_runtime_device(
    module,
    args: argparse.Namespace,
    *,
    required_runtime_device: str = REQUIRED_RUNTIME_DEVICE,
) -> str:
    """Enforce the pinned local runtime device for the realtime viewer."""

    requested = str(args.runtime_device or args.device or module.path_defaults.default_device())
    resolved = module.newton_import_ir.resolve_device(requested)
    if resolved != required_runtime_device:
        raise RuntimeError(
            "This realtime viewer is pinned to the local RTX 4090 path and must run "
            f"on {required_runtime_device}. Requested device resolved to {resolved!r} "
            f"from {requested!r}. Re-run with --runtime-device {required_runtime_device}."
        )
    return resolved


def _apply_on_mode_overrides(args: argparse.Namespace) -> None:
    if args.mode != "on":
        return

    args.rigid_shape = "box"
    args.add_box = bool(getattr(args, "add_bunny", True))
    if args.prefix == "cloth_bunny_playground":
        args.prefix = "cloth_box_playground"

    freeze_profile_cfg = bool(args.profile_only) and bool(args.freeze_profile_config)
    if freeze_profile_cfg:
        return

    if float(args.sim_dt) == 5.0e-5:
        args.sim_dt = 1.0 / 600.0
    if int(args.substeps) == 667:
        args.substeps = 10
    if float(args.rigid_mass) == 0.5:
        args.rigid_mass = 4000.0
    if float(args.body_ke) == 5.0e4:
        args.body_ke = 1.0e4
    if float(args.body_kd) == 5.0e2:
        args.body_kd = 1.0e-2


def build_example_runtime(
    module,
    args: argparse.Namespace,
    *,
    workspace_root: Path,
    required_runtime_device: str = REQUIRED_RUNTIME_DEVICE,
) -> ClothBunnyExampleAssets:
    """Build the prepared runtime state for the realtime cloth+bunny example."""

    args.ir = resolve_workspace_path(args.ir, workspace_root)
    args.out_dir = resolve_workspace_path(args.out_dir, workspace_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _apply_on_mode_overrides(args)
    module._validate_scaling_args(args)

    device = require_runtime_device(
        module,
        args,
        required_runtime_device=required_runtime_device,
    )
    ir_obj = prepare_ir_for_runtime(module, args)
    model, meta, n_obj = module.build_model(ir_obj, args, device)

    add_rigid_shape = bool(getattr(args, "add_box", getattr(args, "add_bunny", True)))
    shape_contacts_enabled = bool(args.shape_contacts) and add_rigid_shape
    cfg = module.newton_import_ir.SimConfig(
        ir_path=args.ir.resolve(),
        out_dir=args.out_dir,
        output_prefix=args.prefix,
        spring_ke_scale=float(args.spring_ke_scale),
        spring_kd_scale=float(args.spring_kd_scale),
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=True,
        disable_particle_contact_kernel=(args.mode == "off"),
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

    particle_grid = model.particle_grid
    search_radius = 0.0
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        search_radius = max(
            search_radius, float(getattr(module.newton_import_ir, "EPSILON", 1.0e-6))
        )

    drag = 0.0
    if args.apply_drag and "drag_damping" in ir_obj:
        drag = float(newton_import_ir.ir_scalar(ir_obj, "drag_damping")) * float(args.drag_damping_scale)

    _, gravity_vec = module.newton_import_ir.resolve_gravity(cfg, ir_obj)
    gravity_axis = None
    gravity_norm = float(np.linalg.norm(gravity_vec))
    if bool(args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
        gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

    physical_particle_radius = model.particle_radius.numpy().astype(np.float32, copy=False).copy()
    visual_particle_radius = np.minimum(
        physical_particle_radius * float(args.particle_radius_vis_scale),
        float(args.particle_radius_vis_min),
    ).astype(np.float32)

    return ClothBunnyExampleAssets(
        device=device,
        ir_obj=ir_obj,
        model=model,
        meta=meta,
        n_obj=n_obj,
        cfg=cfg,
        solver=solver,
        particle_grid=particle_grid,
        search_radius=search_radius,
        drag=drag,
        gravity_axis=gravity_axis,
        physical_particle_radius=physical_particle_radius,
        visual_particle_radius=visual_particle_radius,
    )


def simulate_rollout(
    model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    """Run the canonical cloth+bunny simulation runtime."""

    from .offline import simulate

    return simulate(model, ir_obj, meta, args, n_obj, device)


__all__ = [
    "ClothBunnyExampleAssets",
    "REQUIRED_RUNTIME_DEVICE",
    "build_example_runtime",
    "prepare_ir_for_runtime",
    "require_runtime_device",
    "resolve_workspace_path",
    "simulate_rollout",
]
