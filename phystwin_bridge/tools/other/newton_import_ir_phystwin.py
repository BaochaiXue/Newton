#!/usr/bin/env python3
"""Bridge-side importer wrapper with exact PhysTwin self-collision semantics."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
DEMOS_DIR = BRIDGE_ROOT / "demos"
for path in (CORE_DIR, DEMOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bridge_bootstrap import ensure_bridge_runtime_paths  # noqa: E402

ensure_bridge_runtime_paths()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


newton_import_ir = _load_module(
    "bridge_phystwin_self_collision_newton_import_ir",
    CORE_DIR / "newton_import_ir.py",
)

from self_contact_bridge_kernels import (  # noqa: E402
    apply_velocity_update_from_force,
    build_filtered_self_contact_tables,
    eval_filtered_self_contact_phystwin_velocity,
)
from semiimplicit_bridge_kernels import (  # noqa: E402
    eval_bending_forces,
    eval_body_contact_forces,
    eval_body_joint_forces,
    eval_particle_body_contact_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_contact_forces,
    eval_triangle_forces,
)


def simulate_with_phystwin_self_collision(
    cfg: Any,
    ir: dict[str, np.ndarray],
    model_result: Any,
) -> tuple[Any, dict[str, Any]]:
    model = model_result.model
    device = model.device
    solver = newton_import_ir.newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    collision_pipeline_enabled = newton_import_ir._use_collision_pipeline(cfg, ir)
    contacts = model.contacts() if collision_pipeline_enabled else None

    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int64)
    ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
    n_obj = int(np.asarray(ir["num_object_points"]).reshape(-1)[0])
    has_controllers = ctrl_idx.size > 0

    ctrl_idx_wp = ctrl_target_wp = ctrl_vel_wp = None
    ctrl_vel_zero = None
    if has_controllers:
        ctrl_idx_wp = wp.array(ctrl_idx.astype(np.int32), dtype=wp.int32, device=device)
        ctrl_target_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
        ctrl_vel_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
        ctrl_vel_zero = np.zeros((ctrl_idx.size, 3), dtype=np.float32)

    sim_dt = cfg.sim_dt if cfg.sim_dt is not None else newton_import_ir.ir_scalar(ir, "sim_dt")
    substeps = cfg.substeps_per_frame or int(newton_import_ir.ir_scalar(ir, "sim_substeps"))
    substeps = max(1, int(substeps))
    n_frames = max(1, min(int(cfg.num_frames), int(ctrl_traj.shape[0])))

    drag = 0.0
    if cfg.apply_drag and "drag_damping" in ir:
        drag = float(newton_import_ir.ir_scalar(ir, "drag_damping")) * float(cfg.drag_damping_scale)

    if not newton_import_ir._resolve_particle_contacts(cfg, ir):
        raise ValueError("newton_import_ir_phystwin.py requires particle self-collision to be enabled.")

    edges = np.asarray(ir.get("spring_edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
    neighbor_table, neighbor_count, _, exclusion_summary = build_filtered_self_contact_tables(
        edges,
        n_particles=int(model.particle_count),
        hops=0,
        device=str(device),
    )

    phystwin_collision_dist = float(np.asarray(ir["contact_collision_dist"]).reshape(-1)[0])
    phystwin_collide_elas = float(np.asarray(ir["contact_collide_object_elas"]).reshape(-1)[0])
    phystwin_collide_fric = float(np.asarray(ir["contact_collide_object_fric"]).reshape(-1)[0])
    phystwin_qd_tmp = wp.empty_like(state_in.particle_qd)

    with wp.ScopedDevice(device):
        particle_grid = wp.HashGrid(128, 128, 128)
        particle_grid.reserve(model.particle_count)
    search_radius = max(
        float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion),
        phystwin_collision_dist * 5.0,
        float(newton_import_ir.EPSILON),
    )

    rollout_all: list[np.ndarray] = []
    rollout_obj: list[np.ndarray] = []

    q0 = state_in.particle_q.numpy().astype(np.float32)
    rollout_all.append(q0)
    rollout_obj.append(q0[:n_obj])

    t0 = time.perf_counter()
    for frame in range(1, n_frames):
        for sub in range(substeps):
            state_in.clear_forces()

            if has_controllers:
                target = newton_import_ir.interpolate_controller(
                    ctrl_traj, frame, sub, substeps, cfg.interpolate_controls
                )
                ctrl_target_wp.assign(target.astype(np.float32, copy=False))
                ctrl_vel_wp.assign(ctrl_vel_zero)
                wp.launch(
                    newton_import_ir._write_kinematic_state,
                    dim=ctrl_idx.size,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        ctrl_idx_wp,
                        ctrl_target_wp,
                        ctrl_vel_wp,
                    ],
                    device=device,
                )

            with wp.ScopedDevice(device):
                particle_grid.build(state_in.particle_q, radius=search_radius)

            if collision_pipeline_enabled:
                assert contacts is not None
                model.collide(state_in, contacts)

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
                model,
                state_in,
                control,
                body_f_work,
                solver.joint_attach_ke,
                solver.joint_attach_kd,
            )

            apply_velocity_update_from_force(model, state_in, dt=float(sim_dt))
            particle_f.zero_()
            eval_filtered_self_contact_phystwin_velocity(
                model,
                state_in,
                particle_grid,
                neighbor_table,
                neighbor_count,
                collision_dist=float(phystwin_collision_dist),
                collide_elas=float(phystwin_collide_elas),
                collide_fric=float(phystwin_collide_fric),
                particle_qd_out=phystwin_qd_tmp,
            )
            state_in.particle_qd.assign(phystwin_qd_tmp)

            if solver.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            if contacts is not None:
                eval_body_contact_forces(
                    model,
                    state_in,
                    contacts,
                    friction_smoothing=solver.friction_smoothing,
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
            if body_f_work is body_f:
                solver.integrate_bodies(model, state_in, state_out, float(sim_dt), solver.angular_damping)
            else:
                body_f_prev = state_in.body_f
                state_in.body_f = body_f_work
                solver.integrate_bodies(model, state_in, state_out, float(sim_dt), solver.angular_damping)
                state_in.body_f = body_f_prev
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=n_obj,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        n_obj,
                        float(sim_dt),
                        float(drag),
                    ],
                    device=device,
                )

        q = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q)
        rollout_obj.append(q[:n_obj])

    sim_result = newton_import_ir.SimResult(
        particle_q_all=np.stack(rollout_all),
        particle_q_object=np.stack(rollout_obj),
        wall_time_sec=time.perf_counter() - t0,
    )
    meta = {
        "excluded_pair_count": int(exclusion_summary["excluded_pair_count"]),
        "effective_custom_self_contact_hops": 0,
        "phystwin_collision_dist": float(phystwin_collision_dist),
        "phystwin_collide_elas": float(phystwin_collide_elas),
        "phystwin_collide_fric": float(phystwin_collide_fric),
    }
    return sim_result, meta


def main() -> int:
    args = newton_import_ir.parse_args()
    cfg = newton_import_ir.SimConfig.from_args(args)
    cfg.disable_particle_contact_kernel = True

    wp.init()
    ir = newton_import_ir.load_ir(cfg.ir_path)
    model_result = newton_import_ir.build_model(ir, cfg, cfg.device)
    sim_result, meta = simulate_with_phystwin_self_collision(cfg, ir, model_result)
    summary = newton_import_ir.save_results(cfg, ir, model_result, sim_result)
    summary.setdefault("contacts", {})["self_contact_mode"] = "phystwin"
    summary.setdefault("validation", {})["phystwin_self_contact_semantics"] = (
        "bridge-side unique-mask self-collision with zero excluded pairs"
    )
    summary["validation"]["excluded_pair_count"] = int(meta["excluded_pair_count"])
    summary["validation"]["effective_custom_self_contact_hops"] = 0
    summary["validation"]["disable_particle_contact_kernel_forced"] = True
    summary["validation"]["phystwin_collision_dist"] = float(meta["phystwin_collision_dist"])
    summary["validation"]["phystwin_collide_elas"] = float(meta["phystwin_collide_elas"])
    summary["validation"]["phystwin_collide_fric"] = float(meta["phystwin_collide_fric"])

    json_path = cfg.out_dir / f"{cfg.output_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
