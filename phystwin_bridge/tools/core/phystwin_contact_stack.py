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
    apply_velocity_update_from_force,
    eval_spring_forces_deterministic,
    apply_self_collision_phystwin_velocity_from_table,
    apply_self_collision_phystwin_velocity,
    build_potential_self_collision_table,
    build_filtered_self_contact_tables,
    integrate_ground_collision_phystwin_after_gravity,
    integrate_ground_collision_phystwin,
    sort_self_collision_table_rows,
    update_vel_from_force_phystwin,
)
from semiimplicit_bridge_kernels import (  # noqa: E402
    eval_bending_forces,
    eval_body_contact_forces,
    eval_body_joint_forces,
    eval_particle_body_contact_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


PHYSTWIN_MODE = "phystwin"
GROUND_CONTACT_LAW_NATIVE = "native"
GROUND_CONTACT_LAW_PHYSTWIN = "phystwin"
EPSILON = 1.0e-12


@wp.kernel
def _copy_object_positions(
    particle_q: wp.array(dtype=wp.vec3),
    n_object: int,
    object_q: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= n_object:
        return
    object_q[tid] = particle_q[tid]


@wp.kernel
def _copy_vec3_buffer(
    src: wp.array(dtype=wp.vec3),
    count: int,
    dst: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= count:
        return
    dst[tid] = src[tid]


@dataclass
class PhysTwinContactContext:
    """Reusable bridge-side context for the strict `phystwin` mode."""

    n_object: int
    particle_grid: wp.HashGrid
    object_q: Any
    search_radius: float
    neighbor_table: Any
    neighbor_count: Any
    qd_after_force: Any
    qd_after_collision: Any
    spring_incident_indices: Any
    spring_incident_count: Any
    spring_incident_capacity: int
    freeze_collision_table: bool
    collision_table_capacity: int
    collision_indices: Any
    collision_number: Any
    collision_candidate_total: Any
    collision_candidate_truncated: Any
    collision_dist: float
    collide_elas: float
    collide_fric: float
    ground_collide_elas: float
    ground_collide_fric: float
    reverse_factor: float
    gravity_mag: float
    drag_damping: float


def is_strict_phystwin_mode(cfg: Any) -> bool:
    return (
        str(getattr(cfg, "self_contact_mode", "")).lower() == PHYSTWIN_MODE
        or str(getattr(cfg, "ground_contact_law", "")).lower() == GROUND_CONTACT_LAW_PHYSTWIN
    )


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

    self_contact_mode = str(getattr(cfg, "self_contact_mode", "")).lower()
    ground_contact_law = str(getattr(cfg, "ground_contact_law", "")).lower()
    uses_phystwin_self_collision = self_contact_mode == PHYSTWIN_MODE
    uses_phystwin_ground = ground_contact_law == GROUND_CONTACT_LAW_PHYSTWIN or (
        ground_contact_law == "" and uses_phystwin_self_collision
    )

    ir_self_collision = bool(np.asarray(ir.get("self_collision", np.asarray(False))).reshape(-1)[0])
    particle_contacts_override = getattr(cfg, "particle_contacts", None)
    strict_self_collision_enabled = ir_self_collision if particle_contacts_override is None else bool(particle_contacts_override)

    if uses_phystwin_self_collision and (not particle_contacts_enabled or not strict_self_collision_enabled):
        raise ValueError(
            "Strict bridge `phystwin` mode requires PhysTwin self-collision to be enabled. "
            "Set the IR/self-collision flag ON and then explicitly select "
            "`--self-contact-mode phystwin`."
        )

    if uses_phystwin_ground and not bool(getattr(cfg, "add_ground_plane", False)):
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

    if uses_phystwin_self_collision and "contact_collision_dist" not in ir:
        raise ValueError("Strict bridge `phystwin` mode requires contact_collision_dist in IR.")
    if uses_phystwin_self_collision and (
        "contact_collide_object_elas" not in ir or "contact_collide_object_fric" not in ir
    ):
        raise ValueError(
            "Strict bridge `phystwin` mode requires contact_collide_object_elas and "
            "contact_collide_object_fric in IR."
        )
    if uses_phystwin_ground and ("contact_collide_elas" not in ir or "contact_collide_fric" not in ir):
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
        n_particles=n_object,
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
        particle_grid.reserve(n_object)

    freeze_collision_table = bool(getattr(cfg, "phystwin_freeze_collision_table", True))
    collision_table_capacity = int(getattr(cfg, "phystwin_collision_table_capacity", 500))
    if collision_table_capacity <= 0:
        raise ValueError(
            f"phystwin_collision_table_capacity must be positive, got {collision_table_capacity}"
        )
    collision_indices = None
    collision_number = None
    collision_candidate_total = wp.zeros(n_object, dtype=wp.int32, device=device)
    collision_candidate_truncated = wp.zeros(n_object, dtype=wp.int32, device=device)
    if freeze_collision_table:
        collision_indices_np = np.full(
            (n_object, collision_table_capacity),
            -1,
            dtype=np.int32,
        )
        collision_indices = wp.array(collision_indices_np, dtype=wp.int32, device=device)
        collision_number = wp.zeros(n_object, dtype=wp.int32, device=device)

    search_radius = max(collision_dist * 5.0, EPSILON)

    spring_indices = model.spring_indices.numpy().astype(np.int32, copy=False).reshape(-1, 2)
    incident_lists: list[list[int]] = [[] for _ in range(model.particle_count)]
    for sid, (i_raw, j_raw) in enumerate(spring_indices):
        i = int(i_raw)
        j = int(j_raw)
        if i >= 0:
            incident_lists[i].append(int(sid))
        if j >= 0:
            incident_lists[j].append(int(sid))
    incident_count_np = np.asarray([len(row) for row in incident_lists], dtype=np.int32)
    incident_capacity = max(1, int(incident_count_np.max(initial=0)))
    incident_indices_np = np.full((model.particle_count, incident_capacity), -1, dtype=np.int32)
    for pid, row in enumerate(incident_lists):
        if row:
            incident_indices_np[pid, : len(row)] = np.asarray(sorted(row), dtype=np.int32)

    ctx = PhysTwinContactContext(
        n_object=n_object,
        particle_grid=particle_grid,
        object_q=wp.empty(n_object, dtype=wp.vec3, device=device),
        search_radius=search_radius,
        neighbor_table=neighbor_table,
        neighbor_count=neighbor_count,
        qd_after_force=wp.empty(model.particle_count, dtype=wp.vec3, device=device),
        qd_after_collision=wp.empty(model.particle_count, dtype=wp.vec3, device=device),
        spring_incident_indices=wp.array(incident_indices_np, dtype=wp.int32, device=device),
        spring_incident_count=wp.array(incident_count_np, dtype=wp.int32, device=device),
        spring_incident_capacity=incident_capacity,
        freeze_collision_table=freeze_collision_table,
        collision_table_capacity=collision_table_capacity,
        collision_indices=collision_indices,
        collision_number=collision_number,
        collision_candidate_total=collision_candidate_total,
        collision_candidate_truncated=collision_candidate_truncated,
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
        "phystwin_collision_candidate_mode": (
            "frame_frozen_table" if freeze_collision_table else "dynamic_hash_query"
        ),
        "phystwin_collision_table_capacity": int(collision_table_capacity),
        "phystwin_collision_runtime_object_only": True,
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


def prepare_strict_phystwin_contact_frame(
    model: Any,
    state_in: Any,
    ctx: PhysTwinContactContext,
    *,
    enable_self_collision: bool = True,
) -> None:
    if not enable_self_collision:
        return
    wp.launch(
        kernel=_copy_object_positions,
        dim=ctx.n_object,
        inputs=[state_in.particle_q, int(ctx.n_object)],
        outputs=[ctx.object_q],
        device=model.device,
    )
    if not ctx.freeze_collision_table:
        return
    with wp.ScopedDevice(model.device):
        ctx.particle_grid.build(ctx.object_q, radius=ctx.search_radius)
    assert ctx.collision_indices is not None
    assert ctx.collision_number is not None
    build_potential_self_collision_table(
        device=model.device,
        particle_count=ctx.n_object,
        grid=ctx.particle_grid,
        particle_x=ctx.object_q,
        particle_flags=model.particle_flags,
        n_object=ctx.n_object,
        collision_dist=ctx.collision_dist,
        collision_table_capacity=ctx.collision_table_capacity,
        collision_indices=ctx.collision_indices,
        collision_number=ctx.collision_number,
        collision_candidate_total=ctx.collision_candidate_total,
        collision_candidate_truncated=ctx.collision_candidate_truncated,
    )
    sort_self_collision_table_rows(
        device=model.device,
        particle_count=ctx.n_object,
        collision_indices=ctx.collision_indices,
        collision_number=ctx.collision_number,
        collision_table_capacity=ctx.collision_table_capacity,
    )


def step_strict_phystwin_contact_stack(
    model: Any,
    state_in: Any,
    state_out: Any,
    control: Any,
    ctx: PhysTwinContactContext,
    *,
    solver: Any | None = None,
    contacts: Any | None = None,
    sim_dt: float,
    joint_attach_ke: float,
    joint_attach_kd: float,
    friction_smoothing: float = 1.0,
    angular_damping: float = 0.0,
    enable_self_collision: bool = True,
    ground_contact_law: str = GROUND_CONTACT_LAW_PHYSTWIN,
    law_isolation_mode: bool = False,
) -> None:
    """Run one strict PhysTwin-style substep for the cloth parity scene."""

    if enable_self_collision and not ctx.freeze_collision_table:
        wp.launch(
            kernel=_copy_object_positions,
            dim=ctx.n_object,
            inputs=[state_in.particle_q, int(ctx.n_object)],
            outputs=[ctx.object_q],
            device=model.device,
        )
        with wp.ScopedDevice(model.device):
            ctx.particle_grid.build(ctx.object_q, radius=ctx.search_radius)

    particle_f = state_in.particle_f if state_in.particle_count else None
    body_f = state_in.body_f if state_in.body_count else None
    body_f_work = body_f
    if body_f is not None and model.joint_count and control.joint_f is not None:
        body_f_work = wp.clone(body_f)

    eval_spring_forces_deterministic(
        model,
        state_in,
        incident_spring_indices=ctx.spring_incident_indices,
        incident_spring_count=ctx.spring_incident_count,
        incident_capacity=ctx.spring_incident_capacity,
        particle_f=particle_f,
    )
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

    if law_isolation_mode:
        apply_velocity_update_from_force(model, state_in, dt=float(sim_dt))
        wp.launch(
            kernel=_copy_vec3_buffer,
            dim=model.particle_count,
            inputs=[state_in.particle_qd, int(model.particle_count)],
            outputs=[ctx.qd_after_force],
            device=model.device,
        )

        if enable_self_collision:
            if ctx.freeze_collision_table:
                assert ctx.collision_indices is not None
                assert ctx.collision_number is not None
                apply_self_collision_phystwin_velocity_from_table(
                    device=model.device,
                    particle_count=model.particle_count,
                    particle_x=state_in.particle_q,
                    particle_qd=state_in.particle_qd,
                    particle_mass=model.particle_mass,
                    particle_flags=model.particle_flags,
                    n_object=ctx.n_object,
                    collision_indices=ctx.collision_indices,
                    collision_number=ctx.collision_number,
                    collision_table_capacity=ctx.collision_table_capacity,
                    collision_dist=float(ctx.collision_dist),
                    collide_elas=float(ctx.collide_elas),
                    collide_fric=float(ctx.collide_fric),
                    particle_qd_out=ctx.qd_after_collision,
                )
            else:
                apply_self_collision_phystwin_velocity(
                    device=model.device,
                    particle_count=model.particle_count,
                    particle_x=state_in.particle_q,
                    particle_qd=state_in.particle_qd,
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
        else:
            wp.launch(
                kernel=_copy_vec3_buffer,
                dim=model.particle_count,
                inputs=[state_in.particle_qd, int(model.particle_count)],
                outputs=[ctx.qd_after_collision],
                device=model.device,
            )

        if ground_contact_law == GROUND_CONTACT_LAW_PHYSTWIN:
            integrate_ground_collision_phystwin_after_gravity(
                model,
                particle_q=state_in.particle_q,
                particle_qd=ctx.qd_after_collision,
                n_object=ctx.n_object,
                collide_elas=float(ctx.ground_collide_elas),
                collide_fric=float(ctx.ground_collide_fric),
                dt=float(sim_dt),
                gravity_mag=float(ctx.gravity_mag),
                reverse_factor=float(ctx.reverse_factor),
                particle_q_out=state_out.particle_q,
                particle_qd_out=state_out.particle_qd,
            )
            return
        if ground_contact_law != GROUND_CONTACT_LAW_NATIVE:
            raise ValueError(f"Unsupported ground_contact_law={ground_contact_law!r}")
        if solver is None:
            raise RuntimeError("Native ground-contact law requires a solver instance.")
        if contacts is None:
            raise RuntimeError("Native ground-contact law requires built contacts.")

        wp.launch(
            kernel=_copy_vec3_buffer,
            dim=model.particle_count,
            inputs=[ctx.qd_after_collision, int(model.particle_count)],
            outputs=[state_in.particle_qd],
            device=model.device,
        )
        state_in.clear_forces()

        eval_body_contact_forces(
            model,
            state_in,
            contacts,
            friction_smoothing=float(friction_smoothing),
            body_f_out=body_f_work,
        )
        eval_particle_body_contact_forces(
            model,
            state_in,
            contacts,
            particle_f,
            body_f_work,
            body_f_in_world_frame=False,
        )

        solver.integrate_particles(model, state_in, state_out, float(sim_dt))
        if model.body_count:
            if body_f_work is body_f:
                solver.integrate_bodies(model, state_in, state_out, float(sim_dt), float(angular_damping))
            else:
                body_f_prev = state_in.body_f
                state_in.body_f = body_f_work
                solver.integrate_bodies(model, state_in, state_out, float(sim_dt), float(angular_damping))
                state_in.body_f = body_f_prev
        return

    if ground_contact_law == GROUND_CONTACT_LAW_PHYSTWIN:
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
        if enable_self_collision:
            if ctx.freeze_collision_table:
                assert ctx.collision_indices is not None
                assert ctx.collision_number is not None
                apply_self_collision_phystwin_velocity_from_table(
                    device=model.device,
                    particle_count=model.particle_count,
                    particle_x=state_in.particle_q,
                    particle_qd=ctx.qd_after_force,
                    particle_mass=model.particle_mass,
                    particle_flags=model.particle_flags,
                    n_object=ctx.n_object,
                    collision_indices=ctx.collision_indices,
                    collision_number=ctx.collision_number,
                    collision_table_capacity=ctx.collision_table_capacity,
                    collision_dist=float(ctx.collision_dist),
                    collide_elas=float(ctx.collide_elas),
                    collide_fric=float(ctx.collide_fric),
                    particle_qd_out=ctx.qd_after_collision,
                )
            else:
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
        else:
            wp.launch(
                kernel=_copy_vec3_buffer,
                dim=model.particle_count,
                inputs=[ctx.qd_after_force, int(model.particle_count)],
                outputs=[ctx.qd_after_collision],
                device=model.device,
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
        return

    if ground_contact_law != GROUND_CONTACT_LAW_NATIVE:
        raise ValueError(f"Unsupported ground_contact_law={ground_contact_law!r}")
    if solver is None:
        raise RuntimeError("Native ground-contact law requires a solver instance.")
    if contacts is None:
        raise RuntimeError("Native ground-contact law requires built contacts.")

    apply_velocity_update_from_force(model, state_in, dt=float(sim_dt))
    if enable_self_collision:
        if ctx.freeze_collision_table:
            assert ctx.collision_indices is not None
            assert ctx.collision_number is not None
            apply_self_collision_phystwin_velocity_from_table(
                device=model.device,
                particle_count=model.particle_count,
                particle_x=state_in.particle_q,
                particle_qd=state_in.particle_qd,
                particle_mass=model.particle_mass,
                particle_flags=model.particle_flags,
                n_object=ctx.n_object,
                collision_indices=ctx.collision_indices,
                collision_number=ctx.collision_number,
                collision_table_capacity=ctx.collision_table_capacity,
                collision_dist=float(ctx.collision_dist),
                collide_elas=float(ctx.collide_elas),
                collide_fric=float(ctx.collide_fric),
                particle_qd_out=ctx.qd_after_collision,
            )
        else:
            apply_self_collision_phystwin_velocity(
                device=model.device,
                particle_count=model.particle_count,
                particle_x=state_in.particle_q,
                particle_qd=state_in.particle_qd,
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
        wp.launch(
            kernel=_copy_vec3_buffer,
            dim=model.particle_count,
            inputs=[ctx.qd_after_collision, int(model.particle_count)],
            outputs=[state_in.particle_qd],
            device=model.device,
        )
    state_in.clear_forces()

    body_f = state_in.body_f if state_in.body_count else None
    body_f_work = body_f
    if body_f is not None and model.joint_count and control.joint_f is not None:
        body_f_work = wp.clone(body_f)

    eval_body_contact_forces(
        model,
        state_in,
        contacts,
        friction_smoothing=float(friction_smoothing),
        body_f_out=body_f_work,
    )
    eval_particle_body_contact_forces(
        model,
        state_in,
        contacts,
        particle_f,
        body_f_work,
        body_f_in_world_frame=False,
    )

    solver.integrate_particles(model, state_in, state_out, float(sim_dt))
    if model.body_count:
        if body_f_work is body_f:
            solver.integrate_bodies(model, state_in, state_out, float(sim_dt), float(angular_damping))
        else:
            body_f_prev = state_in.body_f
            state_in.body_f = body_f_work
            solver.integrate_bodies(model, state_in, state_out, float(sim_dt), float(angular_damping))
            state_in.body_f = body_f_prev
