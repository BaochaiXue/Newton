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
from semiimplicit_bridge_kernels import eval_spring_forces  # noqa: E402


DEFAULT_IR = BRIDGE_ROOT / "ir" / "blue_cloth_double_lift_around" / "phystwin_ir_v2_bf_strict.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether controller-driven spring forces differ between "
            "PhysTwin's separate control_x/control_v semantics and the current "
            "bridge zero-mass controller-particle representation."
        )
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--short-frames", type=int, default=3)
    parser.add_argument("--sample-stride", type=int, default=64)
    return parser.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _phys_force_on_objects(
    x_all: np.ndarray,
    v_all: np.ndarray,
    *,
    n_object: int,
    spring_edges: np.ndarray,
    spring_rest: np.ndarray,
    spring_y: np.ndarray,
    spring_kd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obj_force = np.zeros((n_object, 3), dtype=np.float64)
    ctrl_force = np.zeros((n_object, 3), dtype=np.float64)
    for spring_idx, (i_raw, j_raw) in enumerate(spring_edges):
        i = int(i_raw)
        j = int(j_raw)
        xi = x_all[i]
        xj = x_all[j]
        vi = v_all[i]
        vj = v_all[j]
        dis = xj - xi
        dis_len = float(np.linalg.norm(dis))
        if dis_len < 1.0e-12:
            continue
        d = dis / max(dis_len, 1.0e-6)
        v_rel = float(np.dot(vj - vi, d))
        overall_force = d * (
            float(spring_y[spring_idx]) * (dis_len / float(spring_rest[spring_idx]) - 1.0)
            + float(spring_kd[spring_idx]) * v_rel
        )
        if i < n_object:
            obj_force[i] += overall_force
            if j >= n_object:
                ctrl_force[i] += overall_force
        if j < n_object:
            obj_force[j] -= overall_force
            if i >= n_object:
                ctrl_force[j] -= overall_force
    return obj_force.astype(np.float32), ctrl_force.astype(np.float32)


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    diff = np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    abs_per_particle = np.linalg.norm(diff, axis=1)
    ref_scale = np.maximum(np.linalg.norm(ref, axis=1), 1.0e-12)
    rel_per_particle = abs_per_particle / ref_scale
    return {
        "force_abs_max": float(np.max(abs_per_particle)) if abs_per_particle.size else 0.0,
        "force_abs_mean": float(np.mean(abs_per_particle)) if abs_per_particle.size else 0.0,
        "force_rel_max": float(np.max(rel_per_particle)) if rel_per_particle.size else 0.0,
        "force_rel_mean": float(np.mean(rel_per_particle)) if rel_per_particle.size else 0.0,
    }


def _build_cfg(ir_path: Path, out_dir: Path, device: str) -> newton_import_ir.SimConfig:
    return newton_import_ir.SimConfig(
        ir_path=ir_path.resolve(),
        out_dir=out_dir.resolve(),
        output_prefix="controller_spring_diag",
        self_contact_mode=newton_import_ir.SELF_CONTACT_MODE_OFF,
        shape_contacts=False,
        add_ground_plane=False,
        disable_particle_contact_kernel=True,
        strict_physics_checks=True,
        apply_drag=False,
        device=device,
    )


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    ir = newton_import_ir.load_ir(args.ir.resolve())
    cfg = _build_cfg(args.ir, out_dir, args.device)
    model_result = newton_import_ir.build_model(ir, cfg, args.device)
    model = model_result.model
    solver = newton_import_ir.newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int64).ravel()
    ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
    n_object = int(np.asarray(ir["num_object_points"]).reshape(-1)[0])
    sim_dt = cfg.sim_dt if cfg.sim_dt is not None else newton_import_ir.ir_scalar(ir, "sim_dt")
    substeps = cfg.substeps_per_frame or int(newton_import_ir.ir_scalar(ir, "sim_substeps"))
    short_frames = max(2, min(int(args.short_frames), int(ctrl_traj.shape[0])))
    sample_stride = max(1, int(args.sample_stride))

    ctrl_idx_wp = wp.array(ctrl_idx.astype(np.int32), dtype=wp.int32, device=model.device)
    ctrl_target_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=model.device)
    ctrl_vel_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=model.device)
    ctrl_vel_zero = np.zeros((ctrl_idx.size, 3), dtype=np.float32)

    spring_edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    spring_rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).reshape(-1)
    spring_y = np.asarray(ir["spring_y"], dtype=np.float32).reshape(-1)
    spring_kd = np.asarray(ir["spring_kd"], dtype=np.float32).reshape(-1)
    controller_touch_mask = (spring_edges[:, 0] >= n_object) | (spring_edges[:, 1] >= n_object)

    samples: list[dict[str, float | int]] = []
    one_step_metrics: dict[str, float] | None = None

    for frame in range(1, short_frames):
        for sub in range(substeps):
            target = newton_import_ir.interpolate_controller(
                ctrl_traj,
                frame,
                sub,
                substeps,
                cfg.interpolate_controls,
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
                device=model.device,
            )

            if sub % sample_stride == 0:
                state_in.clear_forces()
                eval_spring_forces(model, state_in, state_in.particle_f)
                kernel_force = state_in.particle_f.numpy().astype(np.float32)[:n_object]
                x_all = state_in.particle_q.numpy().astype(np.float32)
                v_all = state_in.particle_qd.numpy().astype(np.float32)
                phys_all, phys_ctrl = _phys_force_on_objects(
                    x_all,
                    v_all,
                    n_object=n_object,
                    spring_edges=spring_edges,
                    spring_rest=spring_rest,
                    spring_y=spring_y,
                    spring_kd=spring_kd,
                )
                m = _metrics(phys_all, kernel_force)
                ctrl_force_norm = np.linalg.norm(phys_ctrl, axis=1)
                sample = {
                    "frame": int(frame),
                    "substep": int(sub),
                    **m,
                    "controller_force_abs_max": float(np.max(ctrl_force_norm)) if ctrl_force_norm.size else 0.0,
                    "controller_force_abs_mean": float(np.mean(ctrl_force_norm)) if ctrl_force_norm.size else 0.0,
                }
                samples.append(sample)
                if one_step_metrics is None:
                    one_step_metrics = dict(sample)

            solver.step(state_in, state_out, control, None, float(sim_dt))
            state_in, state_out = state_out, state_in

    if one_step_metrics is None:
        raise RuntimeError("No sampled substeps were recorded for the controller-spring diagnostic.")

    abs_max = max(float(s["force_abs_max"]) for s in samples)
    abs_mean = float(np.mean([float(s["force_abs_mean"]) for s in samples]))
    rel_max = max(float(s["force_rel_max"]) for s in samples)
    rel_mean = float(np.mean([float(s["force_rel_mean"]) for s in samples]))
    controller_force_peak = max(float(s["controller_force_abs_max"]) for s in samples)

    payload = {
        "ir_path": str(args.ir.resolve()),
        "short_frames": int(short_frames),
        "sample_stride": int(sample_stride),
        "n_object": int(n_object),
        "n_controller": int(ctrl_idx.size),
        "spring_count": int(spring_edges.shape[0]),
        "controller_touching_spring_count": int(np.count_nonzero(controller_touch_mask)),
        "controller_velocity_mode": "zero",
        "one_step": one_step_metrics,
        "short_rollout": {
            "samples_recorded": int(len(samples)),
            "force_abs_max": abs_max,
            "force_abs_mean": abs_mean,
            "force_rel_max": rel_max,
            "force_rel_mean": rel_mean,
            "controller_force_abs_peak": controller_force_peak,
            "pass": bool(abs_max <= 1.0e-5 and rel_max <= 1.0e-5),
        },
        "reference_semantics": "PhysTwin separate control_x/control_v spring law versus Newton embedded zero-mass controller particles",
    }

    json_path = out_dir / "controller_spring_diagnostic.json"
    md_path = out_dir / "controller_spring_diagnostic.md"
    _write(json_path, json.dumps(payload, indent=2) + "\n")
    _write(
        md_path,
        "\n".join(
            [
                "# Controller-Spring Diagnostic",
                "",
                f"- springs touching controllers: `{payload['controller_touching_spring_count']}`",
                f"- one-step force_abs_max: `{payload['one_step']['force_abs_max']}`",
                f"- one-step force_rel_max: `{payload['one_step']['force_rel_max']}`",
                f"- short-rollout force_abs_max: `{payload['short_rollout']['force_abs_max']}`",
                f"- short-rollout force_rel_max: `{payload['short_rollout']['force_rel_max']}`",
                f"- pass: `{payload['short_rollout']['pass']}`",
            ]
        )
        + "\n",
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["short_rollout"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
