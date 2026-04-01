#!/usr/bin/env python3
"""Shared bridge-side PhysTwin contact stack for strict cloth parity scenes.

This module intentionally stays outside ``Newton/newton`` and owns the
bridge-side `phystwin` mode semantics:

- PhysTwin-style force -> velocity update (gravity + drag in the same step)
- PhysTwin-style pairwise self-collision
- PhysTwin-style implicit z=0 ground-plane integration

v1 is intentionally strict: it supports the PhysTwin-native cloth case only and
does not silently mix Newton-only rigid/shape contacts into `phystwin` mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import warp as wp


DEMOS_DIR = Path(__file__).resolve().parents[2] / "demos"
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from self_contact_bridge_kernels import (  # noqa: E402
    apply_self_collision_phystwin_velocity,
    build_filtered_self_contact_tables,
    integrate_ground_collision_phystwin,
    update_vel_from_force_phystwin,
)
from semiimplicit_bridge_kernels import (  # noqa: E402
    eval_bending_forces,
    eval_body_joint_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


PHYSTWIN_MODE = "phystwin"
EPSILON = 1.0e-12


@dataclass
class PhysTwinContactContext:
    """Reusable bridge-side context for the strict `phystwin` mode."""

    n_object: int
    particle_grid: wp.HashGrid
    search_radius: float
    neighbor_table: Any
    neighbor_count: Any
    qd_after_force: Any
    qd_after_collision: Any
    collision_dist: float
    collide_elas: float
    collide_fric: float
    ground_collide_elas: float
    ground_collide_fric: float
    reverse_factor: float
    gravity_mag: float
    drag_damping: float


def is_strict_phystwin_mode(cfg: Any) -> bool:
    return str(getattr(cfg, "self_contact_mode", "")).lower() == PHYSTWIN_MODE


def validate_strict_phystwin_mode(
    cfg: Any,
    ir: dict[str, np.ndarray],
    *,
    particle_contacts_enabled: bool,
) -> None:
    """Validate that the current scene is supported by strict bridge `phystwin`.

    v1 is intentionally narrow and only supports the PhysTwin-native cloth case:
    pairwise self-collision plus the implicit z=0 ground plane.
    """

    if not is_strict_phystwin_mode(cfg):
        return

    if not particle_contacts_enabled:
        raise ValueError(
            "Strict bridge `phystwin` mode requires PhysTwin self-collision to be enabled. "
            "Set the IR/self-collision flag ON and then explicitly select "
            "`--self-contact-mode phystwin`."
        )

    if not bool(getattr(cfg, "add_ground_plane", False)):
        raise ValueError(
            "Strict bridge `phystwin` mode currently targets the PhysTwin cloth parity case "
            "and requires the implicit z=0 ground plane."
        )

    if bool(getattr(cfg, "shape_contacts", False)):
        raise ValueError(
            "Strict bridge `phystwin` mode does not support Newton-only rigid/shape contacts. "
            "Use `off/native/custom` for box/rigid support scenes."
        )

    if bool(getattr(cfg, "rigid_probe", False)):
        raise ValueError(
            "Strict bridge `phystwin` mode does not support rigid-probe scenes."
        )

    if "contact_collision_dist" not in ir:
        raise ValueError("Strict bridge `phystwin` mode requires contact_collision_dist in IR.")
    if "contact_collide_object_elas" not in ir or "contact_collide_object_fric" not in ir:
        raise ValueError(
            "Strict bridge `phystwin` mode requires contact_collide_object_elas and "
            "contact_collide_object_fric in IR."
        )
    if "contact_collide_elas" not in ir or "contact_collide_fric" not in ir:
        raise ValueError(
            "Strict bridge `phystwin` mode requires contact_collide_elas and "
            "contact_collide_fric in IR."
        )


def build_strict_phystwin_contact_context(
    model: Any,
    ir: dict[str, np.ndarray],
    cfg: Any,
    *,
    device: str,
) -> tuple[PhysTwinContactContext, dict[str, Any]]:
    """Create reusable bridge-side buffers + metadata for strict `phystwin` mode."""

    n_object = int(np.asarray(ir["num_object_points"]).reshape(-1)[0])
    edges = np.asarray(ir.get("spring_edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)

    neighbor_table, neighbor_count, _, exclusion_summary = build_filtered_self_contact_tables(
        edges,
        n_particles=int(model.particle_count),
        hops=0,
        device=device,
    )

    collision_dist = float(np.asarray(ir["contact_collision_dist"]).reshape(-1)[0])
    collide_elas = float(np.asarray(ir["contact_collide_object_elas"]).reshape(-1)[0])
    collide_fric = float(np.asarray(ir["contact_collide_object_fric"]).reshape(-1)[0])
    ground_elas_raw = float(np.asarray(ir["contact_collide_elas"]).reshape(-1)[0])
    ground_fric_raw = float(np.asarray(ir["contact_collide_fric"]).reshape(-1)[0])
    ground_collide_elas = float(
        np.clip(
            ground_elas_raw * float(getattr(cfg, "ground_restitution_scale", 1.0)),
            0.0,
            1.0,
        )
    )
    ground_collide_fric = float(
        np.clip(
            ground_fric_raw * float(getattr(cfg, "ground_mu_scale", 1.0)),
            0.0,
            2.0,
        )
    )
    reverse_z = bool(np.asarray(ir.get("reverse_z", np.asarray(True))).reshape(-1)[0])
    reverse_factor = -1.0 if reverse_z else 1.0

    gravity_arg = getattr(cfg, "gravity", None)
    gravity_mag = abs(float(gravity_arg)) if gravity_arg is not None else abs(float(getattr(cfg, "gravity_mag", 9.8)))

    drag_damping = 0.0
    if bool(getattr(cfg, "apply_drag", True)) and "drag_damping" in ir:
        drag_damping = float(np.asarray(ir["drag_damping"]).reshape(-1)[0]) * float(
            getattr(cfg, "drag_damping_scale", 1.0)
        )

    with wp.ScopedDevice(device):
        particle_grid = wp.HashGrid(128, 128, 128)
        particle_grid.reserve(model.particle_count)

    search_radius = max(
        float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion),
        collision_dist * 5.0,
        EPSILON,
    )

    ctx = PhysTwinContactContext(
        n_object=n_object,
        particle_grid=particle_grid,
        search_radius=search_radius,
        neighbor_table=neighbor_table,
        neighbor_count=neighbor_count,
        qd_after_force=wp.empty(model.particle_count, dtype=wp.vec3, device=device),
        qd_after_collision=wp.empty(model.particle_count, dtype=wp.vec3, device=device),
        collision_dist=collision_dist,
        collide_elas=collide_elas,
        collide_fric=collide_fric,
        ground_collide_elas=ground_collide_elas,
        ground_collide_fric=ground_collide_fric,
        reverse_factor=reverse_factor,
        gravity_mag=gravity_mag,
        drag_damping=drag_damping,
    )

    check_updates = {
        "custom_self_contact_hops": 0,
        "effective_custom_self_contact_hops": 0,
        "excluded_neighbor_min": float(exclusion_summary["excluded_neighbor_min"]),
        "excluded_neighbor_mean": float(exclusion_summary["excluded_neighbor_mean"]),
        "excluded_neighbor_median": float(exclusion_summary["excluded_neighbor_median"]),
        "excluded_neighbor_max": float(exclusion_summary["excluded_neighbor_max"]),
        "excluded_pair_count": float(exclusion_summary["excluded_pair_count"]),
        "particle_contacts_enabled": False,
        "self_contact_mode": PHYSTWIN_MODE,
        "collision_radius_source": "shape_preserved_for_bridge_phystwin",
        "phystwin_self_contact_semantics": (
            "bridge-side PhysTwin pairwise self-collision with zero excluded pairs"
        ),
        "phystwin_ground_contact_semantics": (
            "bridge-side PhysTwin implicit z=0 ground-plane integration"
        ),
        "phystwin_collision_dist": float(collision_dist),
        "phystwin_collide_elas": float(collide_elas),
        "phystwin_collide_fric": float(collide_fric),
        "particle_contact_param_source": "phystwin_bridge_exact",
        "ground_contact_param_source": "phystwin_ground_integrator",
        "ground_contact_context": "implicit_z0_plane",
        "ground_contact_elas_supported": True,
        "ground_contact_elas_mapping": "exact_phystwin",
        "ground_contact_elas_raw": float(ground_elas_raw),
        "ground_contact_fric_raw": float(ground_fric_raw),
        "ground_contact_elas_effective": float(ground_collide_elas),
        "ground_contact_map_mu": float(ground_collide_fric),
        "ground_plane_added": True,
        "ground_mu": float(ground_collide_fric),
        "ground_restitution": float(ground_collide_elas),
        "ground_contact_final_ke": None,
        "ground_contact_final_kd": None,
        "ground_contact_final_kf": None,
        "shape_contact_scale_applied": False,
        "shape_contacts_requested_but_unsupported": bool(getattr(cfg, "shape_contacts", False)),
    }
    return ctx, check_updates


def step_strict_phystwin_contact_stack(
    model: Any,
    state_in: Any,
    state_out: Any,
    control: Any,
    ctx: PhysTwinContactContext,
    *,
    sim_dt: float,
    joint_attach_ke: float,
    joint_attach_kd: float,
) -> None:
    """Run one strict PhysTwin-style substep for the cloth parity scene."""

    with wp.ScopedDevice(model.device):
        ctx.particle_grid.build(state_in.particle_q, radius=ctx.search_radius)

    particle_f = state_in.particle_f if state_in.particle_count else None
    body_f = state_in.body_f if state_in.body_count else None
    body_f_work = body_f
    if body_f is not None and model.joint_count and control.joint_f is not None:
        body_f_work = wp.clone(body_f)

    eval_spring_forces(model, state_in, particle_f)
    eval_triangle_forces(model, state_in, control, particle_f)
    eval_bending_forces(model, state_in, particle_f)
    eval_tetrahedra_forces(model, state_in, control, particle_f)
    if body_f_work is not None:
        eval_body_joint_forces(
            model,
            state_in,
            control,
            body_f_work,
            joint_attach_ke,
            joint_attach_kd,
        )

    update_vel_from_force_phystwin(
        model,
        state_in,
        n_object=ctx.n_object,
        dt=float(sim_dt),
        drag_damping=float(ctx.drag_damping),
        gravity_mag=float(ctx.gravity_mag),
        reverse_factor=float(ctx.reverse_factor),
        particle_qd_out=ctx.qd_after_force,
    )
    apply_self_collision_phystwin_velocity(
        device=model.device,
        particle_count=model.particle_count,
        particle_x=state_in.particle_q,
        particle_qd=ctx.qd_after_force,
        particle_mass=model.particle_mass,
        particle_flags=model.particle_flags,
        grid=ctx.particle_grid,
        neighbor_table=ctx.neighbor_table,
        neighbor_count=ctx.neighbor_count,
        collision_dist=float(ctx.collision_dist),
        collide_elas=float(ctx.collide_elas),
        collide_fric=float(ctx.collide_fric),
        particle_qd_out=ctx.qd_after_collision,
    )
    integrate_ground_collision_phystwin(
        model,
        particle_q=state_in.particle_q,
        particle_qd=ctx.qd_after_collision,
        n_object=ctx.n_object,
        collide_elas=float(ctx.ground_collide_elas),
        collide_fric=float(ctx.ground_collide_fric),
        dt=float(sim_dt),
        reverse_factor=float(ctx.reverse_factor),
        particle_q_out=state_out.particle_q,
        particle_qd_out=state_out.particle_qd,
    )
