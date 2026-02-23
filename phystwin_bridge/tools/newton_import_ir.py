#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate a PhysTwin IR in Newton and run a short rollout."
    )
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--solver", choices=["xpbd", "semi_implicit"], default="xpbd")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=None,
        help="Override IR substeps for faster short rollouts.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=None,
        help="Override IR sim_dt.",
    )
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--enable-contacts", action="store_true")
    parser.add_argument(
        "--inference",
        type=Path,
        default=None,
        help="Optional PhysTwin baseline inference.pkl for error metrics.",
    )
    parser.add_argument("--output-prefix", default="newton_rollout_short")
    return parser.parse_args()


def _as_scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(-1)[0])


def _load_ir(ir_path: Path) -> dict[str, np.ndarray]:
    with np.load(ir_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_device(device_str: str) -> str:
    try:
        wp.get_device(device_str)
        return device_str
    except Exception:
        return "cpu"


def _build_model(ir: dict[str, np.ndarray], up_axis: str, gravity: float, device: str):
    axis = newton.Axis.from_any(up_axis)
    builder = newton.ModelBuilder(up_axis=axis, gravity=gravity)

    x0 = ir["x0"]
    v0 = ir["v0"]
    mass = ir["mass"]
    radius = ir["radius"]

    builder.add_particles(
        pos=[tuple(row.tolist()) for row in x0],
        vel=[tuple(row.tolist()) for row in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
    )

    spring_edges = ir["spring_edges"]
    spring_ke = ir["spring_ke"]
    spring_kd = ir["spring_kd"]
    for spring_idx in range(spring_edges.shape[0]):
        i = int(spring_edges[spring_idx, 0])
        j = int(spring_edges[spring_idx, 1])
        builder.add_spring(
            i=i,
            j=j,
            ke=float(spring_ke[spring_idx]),
            kd=float(spring_kd[spring_idx]),
            control=0.0,
        )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, gravity))
    model.particle_mu = _as_scalar(ir["contact_collide_fric"])
    return model


def _make_solver(model: newton.Model, solver_name: str, iterations: int):
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model)
    return newton.solvers.SolverXPBD(model, iterations=iterations)


def _load_inference(path: Path | None):
    if path is None or not path.exists():
        return None
    import pickle

    with path.open("rb") as handle:
        inference = pickle.load(handle)
    return np.asarray(inference, dtype=np.float32)


def main() -> int:
    args = parse_args()
    ir_path = args.ir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ir = _load_ir(ir_path)
    resolved_device = _resolve_device(args.device)
    wp.init()

    model = _build_model(
        ir=ir,
        up_axis=args.up_axis,
        gravity=args.gravity,
        device=resolved_device,
    )
    solver = _make_solver(model, args.solver, args.solver_iterations)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts() if args.enable_contacts else None

    controller_idx = ir["controller_idx"].astype(np.int64)
    controller_traj = ir["controller_traj"].astype(np.float32)
    num_object_points = int(ir["num_object_points"])

    ir_frames = controller_traj.shape[0]
    frames_to_run = max(1, min(args.num_frames, ir_frames))

    sim_dt = float(args.sim_dt) if args.sim_dt is not None else _as_scalar(ir["sim_dt"])
    substeps_per_frame = (
        int(args.substeps_per_frame)
        if args.substeps_per_frame is not None
        else int(np.asarray(ir["sim_substeps"]).reshape(-1)[0])
    )
    substeps_per_frame = max(1, substeps_per_frame)

    rollout_all: list[np.ndarray] = []
    rollout_object: list[np.ndarray] = []

    wall_start = time.perf_counter()
    for frame_idx in range(frames_to_run):
        target_ctrl = controller_traj[frame_idx]
        for _ in range(substeps_per_frame):
            if controller_idx.size > 0:
                q = state_in.particle_q.numpy()
                q[controller_idx] = target_ctrl
                state_in.particle_q.assign(q)

                qd = state_in.particle_qd.numpy()
                qd[controller_idx] = 0.0
                state_in.particle_qd.assign(qd)

            if contacts is not None:
                model.collide(state_in, contacts)

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

        q_frame = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q_frame)
        rollout_object.append(q_frame[:num_object_points])
    wall_elapsed = time.perf_counter() - wall_start

    rollout_all_np = np.stack(rollout_all, axis=0)
    rollout_object_np = np.stack(rollout_object, axis=0)

    inference_path = args.inference
    if inference_path is None:
        candidate = ir_path.parent.parent.parent / "inputs" / "cases" / str(ir["case_name"]) / "inference.pkl"
        inference_path = candidate if candidate.exists() else None
    inference = _load_inference(inference_path)

    rmse_per_frame = None
    mean_rmse = None
    max_rmse = None
    compared_frames = 0
    if inference is not None and inference.ndim == 3 and inference.shape[1] == num_object_points:
        compared_frames = min(frames_to_run, inference.shape[0])
        err = rollout_object_np[:compared_frames] - inference[:compared_frames]
        rmse_per_frame = np.sqrt(np.mean(err * err, axis=(1, 2))).astype(np.float32)
        mean_rmse = float(rmse_per_frame.mean())
        max_rmse = float(rmse_per_frame.max())

    output_prefix = args.output_prefix
    npz_path = out_dir / f"{output_prefix}.npz"
    json_path = out_dir / f"{output_prefix}.json"

    np.savez_compressed(
        npz_path,
        particle_q_all=rollout_all_np,
        particle_q_object=rollout_object_np,
        sim_dt=np.asarray(sim_dt, dtype=np.float32),
        substeps_per_frame=np.asarray(substeps_per_frame, dtype=np.int32),
        frames_run=np.asarray(frames_to_run, dtype=np.int32),
        rmse_per_frame=(
            rmse_per_frame if rmse_per_frame is not None else np.zeros((0,), dtype=np.float32)
        ),
    )

    summary = {
        "ir_path": str(ir_path),
        "output_npz": str(npz_path),
        "device_requested": args.device,
        "device_used": resolved_device,
        "solver": args.solver,
        "solver_iterations": int(args.solver_iterations),
        "frames_run": int(frames_to_run),
        "substeps_per_frame": int(substeps_per_frame),
        "sim_dt": float(sim_dt),
        "wall_time_sec": float(wall_elapsed),
        "particles_total": int(rollout_all_np.shape[1]),
        "particles_object": int(num_object_points),
        "inference_used": str(inference_path) if inference is not None else None,
        "compared_frames": int(compared_frames),
        "rmse_mean": mean_rmse,
        "rmse_max": max_rmse,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
