#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

import newton_import_ir  # noqa: E402
from phystwin_contact_stack import (  # noqa: E402
    GROUND_CONTACT_LAW_NATIVE,
    GROUND_CONTACT_LAW_PHYSTWIN,
    prepare_strict_phystwin_contact_frame,
    step_strict_phystwin_contact_stack,
)
from semiimplicit_bridge_kernels import (  # noqa: E402
    eval_bending_forces,
    eval_body_joint_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from self_contact_bridge_kernels import apply_self_collision_phystwin_velocity_from_table  # noqa: E402


DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"
DEFAULT_MATRIX_ROOT = (
    BRIDGE_ROOT
    / "results"
    / "ground_contact_self_collision_rmse_matrix_20260404_140154_e11491a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why case 3 vs case 4 separate in the cloth+ground 2x2 matrix."
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instrument-until-frame", type=int, default=121)
    parser.add_argument("--target-frames", default="42,120")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--self-delta-tol", type=float, default=1.0e-9)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_curve(matrix_root: Path, case_label: str) -> np.ndarray:
    npz = np.load(matrix_root / case_label / f"{case_label}.npz")
    return np.asarray(npz["rmse_per_frame"], dtype=np.float32)


def _find_first_positive(diff: np.ndarray) -> int | None:
    for idx, value in enumerate(diff):
        if float(value) > 0.0:
            return idx
    return None


def _find_first_persistent_positive(diff: np.ndarray, window: int = 10) -> int | None:
    if diff.size < window:
        return None
    for idx in range(diff.size - window + 1):
        if np.all(diff[idx : idx + window] > 0.0):
            return idx
    return None


def _fmt_float(x: float) -> str:
    return f"{float(x):.9g}"


def _build_cfg(ir_path: Path, out_dir: Path, device: str, ground_contact_law: str) -> newton_import_ir.SimConfig:
    return newton_import_ir.SimConfig(
        ir_path=ir_path.resolve(),
        out_dir=out_dir.resolve(),
        output_prefix=f"diag_{ground_contact_law}",
        self_contact_mode=newton_import_ir.SELF_CONTACT_MODE_PHYSTWIN,
        ground_contact_law=ground_contact_law,
        shape_contacts=False,
        add_ground_plane=True,
        disable_particle_contact_kernel=True,
        strict_physics_checks=True,
        apply_drag=True,
        ground_restitution_mode=newton_import_ir.GROUND_RESTITUTION_MODE_APPROXIMATE_NATIVE,
        device=device,
        num_frames=121,
    )


def _controller_buffers(ir: dict, device: str):
    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int64).ravel()
    ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
    ctrl_idx_wp = wp.array(ctrl_idx.astype(np.int32), dtype=wp.int32, device=device)
    ctrl_target_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
    ctrl_vel_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
    ctrl_vel_zero = np.zeros((ctrl_idx.size, 3), dtype=np.float32)
    return ctrl_idx, ctrl_traj, ctrl_idx_wp, ctrl_target_wp, ctrl_vel_wp, ctrl_vel_zero


def _set_controllers(
    *,
    frame: int,
    substep: int,
    substeps: int,
    cfg: newton_import_ir.SimConfig,
    state_in,
    ctrl_traj: np.ndarray,
    ctrl_idx_wp,
    ctrl_target_wp,
    ctrl_vel_wp,
    ctrl_vel_zero: np.ndarray,
    device: str,
):
    target = newton_import_ir.interpolate_controller(
        ctrl_traj,
        frame,
        substep,
        substeps,
        cfg.interpolate_controls,
    )
    ctrl_target_wp.assign(target.astype(np.float32, copy=False))
    ctrl_vel_wp.assign(ctrl_vel_zero)
    wp.launch(
        newton_import_ir._write_kinematic_state,
        dim=ctrl_idx_wp.shape[0],
        inputs=[
            state_in.particle_q,
            state_in.particle_qd,
            ctrl_idx_wp,
            ctrl_target_wp,
            ctrl_vel_wp,
        ],
        device=device,
    )


def _force_eval(model, state_in, control, solver):
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
            solver.joint_attach_ke,
            solver.joint_attach_kd,
        )


def _native_force_qd_numpy(model, state_in, dt: float) -> np.ndarray:
    qd = state_in.particle_qd.numpy().astype(np.float32)
    f = state_in.particle_f.numpy().astype(np.float32)
    inv_m = model.particle_inv_mass.numpy().astype(np.float32)
    flags = model.particle_flags.numpy().astype(np.int32)
    active = (flags & int(newton_import_ir.newton.ParticleFlags.ACTIVE)) != 0
    out = qd.copy()
    out[active] = out[active] + f[active] * inv_m[active, None] * np.float32(dt)
    return out


def _phystwin_force_qd_numpy(model, state_in, *, n_object: int, dt: float, drag_damping: float, gravity_mag: float, reverse_factor: float) -> np.ndarray:
    qd = state_in.particle_qd.numpy().astype(np.float32)
    f = state_in.particle_f.numpy().astype(np.float32)
    mass = model.particle_mass.numpy().astype(np.float32)
    flags = model.particle_flags.numpy().astype(np.int32)
    out = qd.copy()
    drag_factor = np.exp(-np.float32(dt) * np.float32(drag_damping))
    gravity_vec = np.array([0.0, 0.0, -gravity_mag * reverse_factor], dtype=np.float32)
    for tid in range(n_object):
        if (flags[tid] & int(newton_import_ir.newton.ParticleFlags.ACTIVE)) == 0:
            continue
        if mass[tid] <= 0.0:
            continue
        all_force = f[tid] + mass[tid] * gravity_vec
        v1 = qd[tid] + all_force / mass[tid] * np.float32(dt)
        out[tid] = v1 * drag_factor
    return out


def _assign_vec3(dst, arr: np.ndarray) -> None:
    dst.assign(np.asarray(arr, dtype=np.float32))


def _self_collision_from_table(ctx, model, state_in, qd_in: np.ndarray) -> np.ndarray:
    _assign_vec3(ctx.qd_after_force, qd_in)
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
    return ctx.qd_after_collision.numpy().astype(np.float32)


def _ground_active_phystwin(x: np.ndarray, v: np.ndarray, *, reverse_factor: float, dt: float) -> int:
    x_z = x[:, 2]
    v_z = v[:, 2]
    next_x_z = (x_z + v_z * dt) * reverse_factor
    active = np.logical_and(next_x_z < 0.0, v_z * reverse_factor < -1.0e-4)
    return int(np.count_nonzero(active))


def _ground_active_native(contacts, n_object: int) -> tuple[int, int]:
    if contacts is None or contacts.soft_contact_count is None:
        return 0, 0
    count = int(np.asarray(contacts.soft_contact_count.numpy()).reshape(-1)[0])
    if count <= 0:
        return 0, 0
    particle_ids = contacts.soft_contact_particle.numpy()[:count].astype(np.int32)
    valid = particle_ids[(particle_ids >= 0) & (particle_ids < n_object)]
    return count, int(np.unique(valid).size)


def _norm_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    per = np.linalg.norm(diff, axis=1)
    return {
        "abs_max": float(np.max(per)) if per.size else 0.0,
        "abs_mean": float(np.mean(per)) if per.size else 0.0,
    }


def _build_model_and_runtime(ir_path: Path, out_dir: Path, device: str, ground_contact_law: str):
    cfg = _build_cfg(ir_path, out_dir, device, ground_contact_law)
    ir = newton_import_ir.load_ir(ir_path.resolve())
    model_result = newton_import_ir.build_model(ir, cfg, device)
    model = model_result.model
    solver = newton_import_ir.newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )
    return cfg, ir, model_result, model, solver, model.state(), model.state(), model.control()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wp.init()

    target_frames = [int(x.strip()) for x in args.target_frames.split(",") if x.strip()]
    target_set = set()
    for frame in target_frames:
        for offset in range(-int(args.window), int(args.window) + 1):
            if frame + offset >= 1:
                target_set.add(frame + offset)

    matrix_root = args.matrix_root.resolve()
    c3_curve = _load_curve(matrix_root, "case_3_self_phystwin_ground_native")
    c4_curve = _load_curve(matrix_root, "case_4_self_phystwin_ground_phystwin")
    diff = c4_curve - c3_curve
    first_positive = _find_first_positive(diff)
    first_persistent = _find_first_persistent_positive(diff, window=10)

    cfg3, ir, mr3, model3, solver3, state3_in, state3_out, control3 = _build_model_and_runtime(
        args.ir, out_dir / "tmp_case3_runtime", args.device, GROUND_CONTACT_LAW_NATIVE
    )
    cfg4, _, mr4, model4, solver4, state4_in, state4_out, control4 = _build_model_and_runtime(
        args.ir, out_dir / "tmp_case4_runtime", args.device, GROUND_CONTACT_LAW_PHYSTWIN
    )

    n_obj = int(np.asarray(ir["num_object_points"]).reshape(-1)[0])
    sim_dt = cfg3.sim_dt if cfg3.sim_dt is not None else newton_import_ir.ir_scalar(ir, "sim_dt")
    substeps = cfg3.substeps_per_frame or int(newton_import_ir.ir_scalar(ir, "sim_substeps"))
    drag_damping = newton_import_ir.ir_scalar(ir, "drag_damping") * cfg3.drag_damping_scale
    reverse_z = bool(np.asarray(ir.get("reverse_z", np.asarray(True))).reshape(-1)[0])
    reverse_factor = -1.0 if reverse_z else 1.0
    gravity_mag = abs(float(cfg3.gravity_mag))

    ctrl_idx, ctrl_traj, ctrl_idx_wp3, ctrl_target_wp3, ctrl_vel_wp3, ctrl_vel_zero3 = _controller_buffers(ir, args.device)
    _, _, ctrl_idx_wp4, ctrl_target_wp4, ctrl_vel_wp4, ctrl_vel_zero4 = _controller_buffers(ir, args.device)

    collision_pipeline_enabled3 = newton_import_ir._use_collision_pipeline(cfg3, ir)
    contacts3 = model3.contacts() if collision_pipeline_enabled3 else None

    records: list[dict[str, object]] = []

    for frame in range(1, int(args.instrument_until_frame) + 1):
        prepare_strict_phystwin_contact_frame(model3, state3_in, mr3.phystwin_contact_context, enable_self_collision=True)
        prepare_strict_phystwin_contact_frame(model4, state4_in, mr4.phystwin_contact_context, enable_self_collision=True)

        for sub in range(substeps):
            state3_in.clear_forces()
            state4_in.clear_forces()

            _set_controllers(
                frame=frame,
                substep=sub,
                substeps=substeps,
                cfg=cfg3,
                state_in=state3_in,
                ctrl_traj=ctrl_traj,
                ctrl_idx_wp=ctrl_idx_wp3,
                ctrl_target_wp=ctrl_target_wp3,
                ctrl_vel_wp=ctrl_vel_wp3,
                ctrl_vel_zero=ctrl_vel_zero3,
                device=args.device,
            )
            _set_controllers(
                frame=frame,
                substep=sub,
                substeps=substeps,
                cfg=cfg4,
                state_in=state4_in,
                ctrl_traj=ctrl_traj,
                ctrl_idx_wp=ctrl_idx_wp4,
                ctrl_target_wp=ctrl_target_wp4,
                ctrl_vel_wp=ctrl_vel_wp4,
                ctrl_vel_zero=ctrl_vel_zero4,
                device=args.device,
            )

            if collision_pipeline_enabled3:
                model3.collide(state3_in, contacts3)

            if frame in target_set:
                _force_eval(model3, state3_in, control3, solver3)
                _force_eval(model4, state4_in, control4, solver4)

                c3_force_native = _native_force_qd_numpy(model3, state3_in, sim_dt)
                c3_force_phystwin = _phystwin_force_qd_numpy(
                    model3,
                    state3_in,
                    n_object=n_obj,
                    dt=sim_dt,
                    drag_damping=drag_damping,
                    gravity_mag=gravity_mag,
                    reverse_factor=reverse_factor,
                )
                c4_force_phystwin = _phystwin_force_qd_numpy(
                    model4,
                    state4_in,
                    n_object=n_obj,
                    dt=sim_dt,
                    drag_damping=drag_damping,
                    gravity_mag=gravity_mag,
                    reverse_factor=reverse_factor,
                )
                c3_post_self = _self_collision_from_table(mr3.phystwin_contact_context, model3, state3_in, c3_force_native)
                c4_post_self = _self_collision_from_table(mr4.phystwin_contact_context, model4, state4_in, c4_force_phystwin)

                c3_self_delta = np.linalg.norm(c3_post_self[:n_obj] - c3_force_native[:n_obj], axis=1)
                c4_self_delta = np.linalg.norm(c4_post_self[:n_obj] - c4_force_phystwin[:n_obj], axis=1)
                c3_candidate_total = int(np.sum(mr3.phystwin_contact_context.collision_number.numpy()))
                c4_candidate_total = int(np.sum(mr4.phystwin_contact_context.collision_number.numpy()))
                c3_ground_contact_count, c3_ground_active_particles = _ground_active_native(contacts3, n_obj)
                c4_ground_active_particles = _ground_active_phystwin(
                    state4_in.particle_q.numpy().astype(np.float32)[:n_obj],
                    c4_post_self[:n_obj],
                    reverse_factor=reverse_factor,
                    dt=sim_dt,
                )

                state3_in.clear_forces()
                state4_in.clear_forces()

                _set_controllers(
                    frame=frame,
                    substep=sub,
                    substeps=substeps,
                    cfg=cfg3,
                    state_in=state3_in,
                    ctrl_traj=ctrl_traj,
                    ctrl_idx_wp=ctrl_idx_wp3,
                    ctrl_target_wp=ctrl_target_wp3,
                    ctrl_vel_wp=ctrl_vel_wp3,
                    ctrl_vel_zero=ctrl_vel_zero3,
                    device=args.device,
                )
                _set_controllers(
                    frame=frame,
                    substep=sub,
                    substeps=substeps,
                    cfg=cfg4,
                    state_in=state4_in,
                    ctrl_traj=ctrl_traj,
                    ctrl_idx_wp=ctrl_idx_wp4,
                    ctrl_target_wp=ctrl_target_wp4,
                    ctrl_vel_wp=ctrl_vel_wp4,
                    ctrl_vel_zero=ctrl_vel_zero4,
                    device=args.device,
                )
                if collision_pipeline_enabled3:
                    model3.collide(state3_in, contacts3)

                step_strict_phystwin_contact_stack(
                    model3,
                    state3_in,
                    state3_out,
                    control3,
                    mr3.phystwin_contact_context,
                    solver=solver3,
                    contacts=contacts3,
                    sim_dt=sim_dt,
                    joint_attach_ke=solver3.joint_attach_ke,
                    joint_attach_kd=solver3.joint_attach_kd,
                    friction_smoothing=solver3.friction_smoothing,
                    angular_damping=solver3.angular_damping,
                    enable_self_collision=True,
                    ground_contact_law=GROUND_CONTACT_LAW_NATIVE,
                )
                step_strict_phystwin_contact_stack(
                    model4,
                    state4_in,
                    state4_out,
                    control4,
                    mr4.phystwin_contact_context,
                    solver=solver4,
                    contacts=None,
                    sim_dt=sim_dt,
                    joint_attach_ke=solver4.joint_attach_ke,
                    joint_attach_kd=solver4.joint_attach_kd,
                    friction_smoothing=solver4.friction_smoothing,
                    angular_damping=solver4.angular_damping,
                    enable_self_collision=True,
                    ground_contact_law=GROUND_CONTACT_LAW_PHYSTWIN,
                )

                rec = {
                    "frame": int(frame),
                    "substep": int(sub),
                    "case3_candidate_total": c3_candidate_total,
                    "case4_candidate_total": c4_candidate_total,
                    "case3_self_active_nodes": int(np.count_nonzero(c3_self_delta > args.self_delta_tol)),
                    "case4_self_active_nodes": int(np.count_nonzero(c4_self_delta > args.self_delta_tol)),
                    "case3_ground_contact_count": c3_ground_contact_count,
                    "case3_ground_active_particles": c3_ground_active_particles,
                    "case4_ground_active_particles": c4_ground_active_particles,
                    "pre_step_q_diff": _norm_stats(
                        state3_in.particle_q.numpy().astype(np.float32)[:n_obj],
                        state4_in.particle_q.numpy().astype(np.float32)[:n_obj],
                    ),
                    "pre_step_qd_diff": _norm_stats(
                        state3_in.particle_qd.numpy().astype(np.float32)[:n_obj],
                        state4_in.particle_qd.numpy().astype(np.float32)[:n_obj],
                    ),
                    "post_force_timing_diff": _norm_stats(
                        c3_force_phystwin[:n_obj],
                        c3_force_native[:n_obj],
                    ),
                    "case3_vs_case4_post_self_diff": _norm_stats(
                        c3_post_self[:n_obj],
                        c4_post_self[:n_obj],
                    ),
                    "post_step_q_diff": _norm_stats(
                        state3_out.particle_q.numpy().astype(np.float32)[:n_obj],
                        state4_out.particle_q.numpy().astype(np.float32)[:n_obj],
                    ),
                    "post_step_qd_diff": _norm_stats(
                        state3_out.particle_qd.numpy().astype(np.float32)[:n_obj],
                        state4_out.particle_qd.numpy().astype(np.float32)[:n_obj],
                    ),
                }
                records.append(rec)
            else:
                step_strict_phystwin_contact_stack(
                    model3,
                    state3_in,
                    state3_out,
                    control3,
                    mr3.phystwin_contact_context,
                    solver=solver3,
                    contacts=contacts3,
                    sim_dt=sim_dt,
                    joint_attach_ke=solver3.joint_attach_ke,
                    joint_attach_kd=solver3.joint_attach_kd,
                    friction_smoothing=solver3.friction_smoothing,
                    angular_damping=solver3.angular_damping,
                    enable_self_collision=True,
                    ground_contact_law=GROUND_CONTACT_LAW_NATIVE,
                )
                step_strict_phystwin_contact_stack(
                    model4,
                    state4_in,
                    state4_out,
                    control4,
                    mr4.phystwin_contact_context,
                    solver=solver4,
                    contacts=None,
                    sim_dt=sim_dt,
                    joint_attach_ke=solver4.joint_attach_ke,
                    joint_attach_kd=solver4.joint_attach_kd,
                    friction_smoothing=solver4.friction_smoothing,
                    angular_damping=solver4.angular_damping,
                    enable_self_collision=True,
                    ground_contact_law=GROUND_CONTACT_LAW_PHYSTWIN,
                )

            state3_in, state3_out = state3_out, state3_in
            state4_in, state4_out = state4_out, state4_in

            if cfg3.apply_drag and frame <= int(args.instrument_until_frame):
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=n_obj,
                    inputs=[state3_in.particle_q, state3_in.particle_qd, n_obj, sim_dt, drag_damping],
                    device=args.device,
                )

    payload = {
        "matrix_root": str(matrix_root),
        "first_positive_frame_from_matrix": first_positive,
        "first_persistent_positive_frame_from_matrix": first_persistent,
        "target_frames": sorted(target_set),
        "records": records,
    }
    json_path = out_dir / "step_diagnostics.json"
    md_path = out_dir / "step_diagnostics.md"
    _write_text(json_path, json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Step Diagnostics",
        "",
        f"- first positive frame from matrix root: `{first_positive}`",
        f"- first persistent positive frame from matrix root: `{first_persistent}`",
        "",
        "## Highlights",
        "",
    ]
    for rec in records[: min(12, len(records))]:
        lines.append(
            f"- frame `{rec['frame']}` substep `{rec['substep']}`:"
            f" post_force_timing_diff.abs_max=`{_fmt_float(rec['post_force_timing_diff']['abs_max'])}`"
            f", case3_vs_case4_post_self_diff.abs_max=`{_fmt_float(rec['case3_vs_case4_post_self_diff']['abs_max'])}`"
            f", post_step_q_diff.abs_max=`{_fmt_float(rec['post_step_q_diff']['abs_max'])}`"
            f", case3_ground_active_particles=`{rec['case3_ground_active_particles']}`"
            f", case4_ground_active_particles=`{rec['case4_ground_active_particles']}`"
        )
    lines.append("")
    _write_text(md_path, "\n".join(lines))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
