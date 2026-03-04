#!/usr/bin/env python3
"""Mass/momentum probe for PhysTwin->Newton bridge.

Purpose
-------
Run a controlled collision experiment between:
- imported spring-mass particles from an IR file (object particles mass forced to 1.0), and
- a native Newton rigid body (box),
to check whether interaction behavior is physically plausible under a shared solver.

This script is intentionally independent from the default bridge pipeline so it
does not alter production parity behavior.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_core_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_defaults = _load_core_module("path_defaults", CORE_DIR / "path_defaults.py")
newton_import_ir = _load_core_module("newton_import_ir", CORE_DIR / "newton_import_ir.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a mass/momentum interaction probe: imported spring-mass particles "
            "(object mass forced to 1) vs a native Newton rigid body."
        )
    )
    parser.add_argument("--ir", type=Path, required=True, help="IR .npz file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--output-prefix", default="mass_probe")
    parser.add_argument("--device", default=path_defaults.default_device())

    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--substeps-per-frame", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=None)

    parser.add_argument("--object-mass", type=float, default=1.0)
    parser.add_argument("--drop-controller-springs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-controllers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controller-radius", type=float, default=1e-5)

    parser.add_argument("--rigid-mass", type=float, default=5.0)
    parser.add_argument("--rigid-inertia-diag", type=float, default=0.05)
    parser.add_argument("--body-hx", type=float, default=0.02)
    parser.add_argument("--body-hy", type=float, default=0.02)
    parser.add_argument("--body-hz", type=float, default=0.02)
    parser.add_argument(
        "--body-offset",
        type=float,
        nargs=3,
        default=(-0.18, 0.0, 0.02),
        metavar=("DX", "DY", "DZ"),
        help="Rigid body initial offset relative to object center.",
    )
    parser.add_argument(
        "--body-velocity",
        type=float,
        nargs=3,
        default=(1.2, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
        help="Rigid body initial linear velocity.",
    )
    parser.add_argument("--body-mu", type=float, default=0.4)
    parser.add_argument("--body-ke", type=float, default=5e4)
    parser.add_argument("--body-kd", type=float, default=5e2)

    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument("--friction-smoothing", type=float, default=1.0)
    parser.add_argument("--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _to_transform_xyzw(t: np.ndarray) -> np.ndarray:
    """Extract translation [x,y,z] from Warp transform array."""
    if t.ndim != 2 or t.shape[1] < 3:
        raise ValueError(f"Unexpected body_q shape: {t.shape}")
    return t[:, :3]


def build_probe_model(ir: dict, args: argparse.Namespace, device: str):
    up_axis = newton.Axis.from_any("Z")
    builder = newton.ModelBuilder(up_axis=up_axis, gravity=0.0)

    x0 = np.asarray(ir["x0"], dtype=np.float32)
    v0 = np.asarray(ir["v0"], dtype=np.float32)
    mass = np.asarray(ir["mass"], dtype=np.float32).copy()
    n_total = x0.shape[0]
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    if n_obj <= 0 or n_obj > n_total:
        raise ValueError(f"Invalid num_object_points={n_obj} for total={n_total}")

    # Force object particle masses to 1.0 (experiment requirement).
    mass[:n_obj] = float(args.object_mass)

    radius, _, _ = newton_import_ir.resolve_collision_radius(ir, n_total)
    if "contact_collision_dist" in ir:
        collision_dist = newton_import_ir.ir_scalar(ir, "contact_collision_dist")
        radius[:n_obj] = max(collision_dist * 0.5, newton_import_ir.EPSILON)
        if "controller_idx" in ir:
            ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int64).ravel()
            if ctrl_idx.size:
                radius[ctrl_idx] = float(args.controller_radius)

    flags = np.full(n_total, int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
    builder.add_particles(
        pos=[tuple(row.tolist()) for row in x0],
        vel=[tuple(row.tolist()) for row in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
        flags=flags.astype(int).tolist(),
    )

    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).ravel()
    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel()
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel()
    if not (edges.shape[0] == rest.shape[0] == ke.shape[0] == kd.shape[0]):
        raise ValueError("Spring arrays have inconsistent lengths.")

    if args.drop_controller_springs:
        keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
        edges = edges[keep]
        rest = rest[keep]
        ke = ke[keep]
        kd = kd[keep]

    for idx in range(edges.shape[0]):
        i = int(edges[idx, 0])
        j = int(edges[idx, 1])
        builder.add_spring(i=i, j=j, ke=float(ke[idx]), kd=float(kd[idx]), control=0.0)
        builder.spring_rest_length[-1] = float(rest[idx])

    obj_center = x0[:n_obj].mean(axis=0)
    body_pos = obj_center + np.asarray(args.body_offset, dtype=np.float32)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*body_pos.tolist()), wp.quat_identity()),
        mass=float(args.rigid_mass),
        inertia=wp.mat33(
            float(args.rigid_inertia_diag),
            0.0,
            0.0,
            0.0,
            float(args.rigid_inertia_diag),
            0.0,
            0.0,
            0.0,
            float(args.rigid_inertia_diag),
        ),
        lock_inertia=True,
        label="probe_box",
    )
    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.body_mu)
    rigid_cfg.ke = float(args.body_ke)
    rigid_cfg.kd = float(args.body_kd)
    builder.add_shape_box(
        body=body,
        hx=float(args.body_hx),
        hy=float(args.body_hy),
        hz=float(args.body_hz),
        cfg=rigid_cfg,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model, n_obj, mass


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = newton_import_ir.resolve_device(args.device)
    ir = newton_import_ir.load_ir(args.ir.resolve())

    model, n_obj, particle_mass = build_probe_model(ir, args, device)
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=float(args.angular_damping),
        friction_smoothing=float(args.friction_smoothing),
        enable_tri_contact=bool(args.enable_tri_contact),
    )
    contacts = model.contacts()
    control = model.control()
    state_in = model.state()
    state_out = model.state()

    ctrl_idx = np.asarray(ir.get("controller_idx", np.zeros((0,), dtype=np.int32)), dtype=np.int32).ravel()
    ctrl_targets = None
    ctrl_idx_wp = None
    ctrl_target_wp = None
    ctrl_vel_wp = None
    if args.freeze_controllers and ctrl_idx.size:
        ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
        ctrl_targets = ctrl_traj[0]
        ctrl_idx_wp = wp.array(ctrl_idx, dtype=wp.int32, device=device)
        ctrl_target_wp = wp.array(ctrl_targets, dtype=wp.vec3, device=device)
        ctrl_vel_wp = wp.zeros(ctrl_idx.size, dtype=wp.vec3, device=device)

    # Set rigid body's initial linear velocity.
    body_qd = state_in.body_qd.numpy()
    if body_qd.shape[0] < 1:
        raise RuntimeError("No rigid body present in probe model.")
    body_qd[0, 0:3] = np.asarray(args.body_velocity, dtype=np.float32)
    body_qd[0, 3:6] = 0.0
    state_in.body_qd.assign(body_qd.astype(np.float32))

    sim_dt = (
        float(args.sim_dt)
        if args.sim_dt is not None
        else float(newton_import_ir.ir_scalar(ir, "sim_dt"))
    )
    substeps = (
        int(args.substeps_per_frame)
        if args.substeps_per_frame is not None
        else int(newton_import_ir.ir_scalar(ir, "sim_substeps"))
    )
    substeps = max(1, substeps)
    frames = max(2, int(args.frames))

    body_pos_hist: list[np.ndarray] = []
    body_vel_hist: list[np.ndarray] = []
    obj_com_hist: list[np.ndarray] = []
    p_obj_hist: list[np.ndarray] = []
    p_body_hist: list[np.ndarray] = []
    p_total_hist: list[np.ndarray] = []

    t0 = time.perf_counter()
    for _frame in range(frames):
        # Record frame-start state.
        q = state_in.particle_q.numpy().astype(np.float32)
        qd = state_in.particle_qd.numpy().astype(np.float32)
        bq = _to_transform_xyzw(state_in.body_q.numpy().astype(np.float32))[0]
        bqd = state_in.body_qd.numpy().astype(np.float32)[0, 0:3]

        obj_q = q[:n_obj]
        obj_qd = qd[:n_obj]
        obj_mass = particle_mass[:n_obj].astype(np.float32)
        obj_com = (obj_q * obj_mass[:, None]).sum(axis=0) / max(float(obj_mass.sum()), 1e-12)
        p_obj = (obj_qd * obj_mass[:, None]).sum(axis=0)
        p_body = bqd * float(args.rigid_mass)
        p_total = p_obj + p_body

        body_pos_hist.append(bq.copy())
        body_vel_hist.append(bqd.copy())
        obj_com_hist.append(obj_com.copy())
        p_obj_hist.append(p_obj.copy())
        p_body_hist.append(p_body.copy())
        p_total_hist.append(p_total.copy())

        for _sub in range(substeps):
            state_in.clear_forces()

            if ctrl_idx_wp is not None:
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

            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

    wall_time = time.perf_counter() - t0

    body_pos = np.asarray(body_pos_hist, dtype=np.float32)
    body_vel = np.asarray(body_vel_hist, dtype=np.float32)
    obj_com = np.asarray(obj_com_hist, dtype=np.float32)
    p_obj = np.asarray(p_obj_hist, dtype=np.float32)
    p_body = np.asarray(p_body_hist, dtype=np.float32)
    p_total = np.asarray(p_total_hist, dtype=np.float32)

    p0 = p_total[0]
    p1 = p_total[-1]
    p_delta = p1 - p0
    p_delta_norm = float(np.linalg.norm(p_delta))
    p0_norm = float(np.linalg.norm(p0))
    p_rel = float(p_delta_norm / max(p0_norm, 1e-8))
    body_speed0 = float(np.linalg.norm(body_vel[0]))
    body_speed_min = float(np.min(np.linalg.norm(body_vel, axis=1)))
    body_speed_end = float(np.linalg.norm(body_vel[-1]))
    min_body_obj_dist = float(np.min(np.linalg.norm(body_pos - obj_com, axis=1)))

    out_npz = args.out_dir / f"{args.output_prefix}.npz"
    out_csv = args.out_dir / f"{args.output_prefix}_timeseries.csv"
    out_json = args.out_dir / f"{args.output_prefix}_summary.json"

    np.savez_compressed(
        out_npz,
        body_pos=body_pos,
        body_vel=body_vel,
        object_com=obj_com,
        p_obj=p_obj,
        p_body=p_body,
        p_total=p_total,
        sim_dt=np.float32(sim_dt),
        substeps_per_frame=np.int32(substeps),
    )

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "body_x",
                "body_y",
                "body_z",
                "obj_x",
                "obj_y",
                "obj_z",
                "body_vx",
                "body_vy",
                "body_vz",
                "p_total_x",
                "p_total_y",
                "p_total_z",
            ]
        )
        for i in range(body_pos.shape[0]):
            writer.writerow(
                [
                    i,
                    *body_pos[i].tolist(),
                    *obj_com[i].tolist(),
                    *body_vel[i].tolist(),
                    *p_total[i].tolist(),
                ]
            )

    summary = {
        "ir_path": str(args.ir.resolve()),
        "device": str(device),
        "frames": int(frames),
        "substeps_per_frame": int(substeps),
        "sim_dt": float(sim_dt),
        "object_mass_set_to": float(args.object_mass),
        "rigid_mass": float(args.rigid_mass),
        "drop_controller_springs": bool(args.drop_controller_springs),
        "freeze_controllers": bool(args.freeze_controllers),
        "initial_total_momentum_norm": p0_norm,
        "final_total_momentum_norm": float(np.linalg.norm(p1)),
        "momentum_delta_norm": p_delta_norm,
        "momentum_delta_relative": p_rel,
        "body_speed_initial": body_speed0,
        "body_speed_min": body_speed_min,
        "body_speed_final": body_speed_end,
        "min_body_object_com_distance": min_body_obj_dist,
        "wall_time_sec": wall_time,
        "outputs": {
            "npz": str(out_npz),
            "csv": str(out_csv),
        },
    }

    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
