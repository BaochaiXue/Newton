#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def _parse_scalar(value: str):
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in raw for char in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def load_config(config_path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        config: dict = {}
        with config_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                config[key.strip()] = _parse_scalar(value)
        return config


def to_scalar(value, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() == 0:
            return float(default)
        return float(tensor[0].item())
    if isinstance(value, np.ndarray):
        array = value.reshape(-1)
        if array.size == 0:
            return float(default)
        return float(array[0])
    return float(value)


def build_structure_points(final_data: dict) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(final_data["object_points"], dtype=np.float32)
    surface_points = np.asarray(final_data["surface_points"], dtype=np.float32)
    interior_points = np.asarray(final_data["interior_points"], dtype=np.float32)
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float32)

    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(f"Unexpected object_points shape: {object_points.shape}")
    if controller_points.ndim != 3 or controller_points.shape[-1] != 3:
        raise ValueError(f"Unexpected controller_points shape: {controller_points.shape}")

    structure_points = np.concatenate(
        [object_points[0], surface_points, interior_points], axis=0
    ).astype(np.float32)
    return structure_points, controller_points


def build_springs_open3d(
    structure_points: np.ndarray,
    controller_points_0: np.ndarray | None,
    object_radius: float,
    object_max_neighbours: int,
    controller_radius: float,
    controller_max_neighbours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import open3d as o3d  # type: ignore

    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(structure_points)
    pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

    points = np.asarray(object_pcd.points)
    spring_flags = np.zeros((len(points), len(points)), dtype=np.uint8)
    edges: list[list[int]] = []
    rest_lengths: list[float] = []

    for i in range(len(points)):
        _, idx, _ = pcd_tree.search_hybrid_vector_3d(
            points[i], object_radius, object_max_neighbours
        )
        idx = idx[1:]
        for j in idx:
            rest_length = float(np.linalg.norm(points[i] - points[j]))
            if spring_flags[i, j] == 0 and spring_flags[j, i] == 0 and rest_length > 1e-4:
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                edges.append([i, j])
                rest_lengths.append(rest_length)

    num_object_springs = len(edges)

    if controller_points_0 is not None and controller_points_0.shape[0] > 0:
        num_object_points = len(points)
        points = np.concatenate([points, controller_points_0], axis=0)
        for i in range(controller_points_0.shape[0]):
            _, idx, _ = pcd_tree.search_hybrid_vector_3d(
                controller_points_0[i], controller_radius, controller_max_neighbours
            )
            for j in idx:
                edges.append([num_object_points + i, j])
                rest_lengths.append(
                    float(np.linalg.norm(controller_points_0[i] - points[j]))
                )

    return (
        points.astype(np.float32),
        np.asarray(edges, dtype=np.int32),
        np.asarray(rest_lengths, dtype=np.float32),
        num_object_springs,
    )


def build_springs_bruteforce(
    structure_points: np.ndarray,
    controller_points_0: np.ndarray | None,
    object_radius: float,
    object_max_neighbours: int,
    controller_radius: float,
    controller_max_neighbours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    points_object = structure_points.astype(np.float32)
    object_count = points_object.shape[0]
    object_radius_sq = float(object_radius) ** 2
    controller_radius_sq = float(controller_radius) ** 2

    spring_flags = np.zeros((object_count, object_count), dtype=np.uint8)
    edges: list[list[int]] = []
    rest_lengths: list[float] = []

    for i in range(object_count):
        deltas = points_object - points_object[i]
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        candidate_indices = np.flatnonzero(distances_sq <= object_radius_sq)
        candidate_indices = candidate_indices[
            np.argsort(distances_sq[candidate_indices], kind="mergesort")
        ]
        if candidate_indices.size > object_max_neighbours:
            candidate_indices = candidate_indices[:object_max_neighbours]
        for j in candidate_indices:
            if j == i:
                continue
            rest_length = float(np.sqrt(distances_sq[j]))
            if spring_flags[i, j] == 0 and spring_flags[j, i] == 0 and rest_length > 1e-4:
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                edges.append([i, int(j)])
                rest_lengths.append(rest_length)

    num_object_springs = len(edges)
    points_all = points_object

    if controller_points_0 is not None and controller_points_0.shape[0] > 0:
        points_all = np.concatenate([points_object, controller_points_0], axis=0)
        for i in range(controller_points_0.shape[0]):
            deltas = points_object - controller_points_0[i]
            distances_sq = np.einsum("ij,ij->i", deltas, deltas)
            candidate_indices = np.flatnonzero(distances_sq <= controller_radius_sq)
            candidate_indices = candidate_indices[
                np.argsort(distances_sq[candidate_indices], kind="mergesort")
            ]
            if candidate_indices.size > controller_max_neighbours:
                candidate_indices = candidate_indices[:controller_max_neighbours]
            for j in candidate_indices:
                edges.append([object_count + i, int(j)])
                rest_lengths.append(float(np.sqrt(distances_sq[j])))

    return (
        points_all.astype(np.float32),
        np.asarray(edges, dtype=np.int32),
        np.asarray(rest_lengths, dtype=np.float32),
        num_object_springs,
    )


def build_springs(
    structure_points: np.ndarray,
    controller_points_0: np.ndarray | None,
    object_radius: float,
    object_max_neighbours: int,
    controller_radius: float,
    controller_max_neighbours: int,
    topology_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    if topology_method == "open3d":
        points, edges, rest, num_object_springs = build_springs_open3d(
            structure_points=structure_points,
            controller_points_0=controller_points_0,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
        )
        return points, edges, rest, num_object_springs, "open3d"

    if topology_method == "bruteforce":
        points, edges, rest, num_object_springs = build_springs_bruteforce(
            structure_points=structure_points,
            controller_points_0=controller_points_0,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
        )
        return points, edges, rest, num_object_springs, "bruteforce"

    try:
        points, edges, rest, num_object_springs = build_springs_bruteforce(
            structure_points=structure_points,
            controller_points_0=controller_points_0,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
        )
        return points, edges, rest, num_object_springs, "bruteforce"
    except Exception:
        points, edges, rest, num_object_springs = build_springs_open3d(
            structure_points=structure_points,
            controller_points_0=controller_points_0,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
        )
        return points, edges, rest, num_object_springs, "open3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PhysTwin case to a Newton-oriented IR v2 .npz bundle."
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--topology-method",
        choices=["auto", "open3d", "bruteforce"],
        default="bruteforce",
        help=(
            "Spring topology reconstruction method. "
            "Use 'bruteforce' for determinism (recommended for matching checkpoint spring_Y ordering)."
        ),
    )
    parser.add_argument(
        "--strict-phystwin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable strict PhysTwin-aligned defaults and checks. "
            "When enabled, spring_ke defaults to spring_Y/rest_length."
        ),
    )
    parser.add_argument(
        "--spring-ke-mode",
        choices=["raw", "y_over_rest"],
        default=None,
        help=(
            "Mapping from checkpoint spring_Y to Newton spring_ke. "
            "If unset, defaults to y_over_rest in strict mode, otherwise raw."
        ),
    )
    parser.add_argument(
        "--rest-length-eps",
        type=float,
        default=1e-8,
        help="Numerical floor used when mapping spring_ke from spring_Y/rest_length.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    config_path = args.config.resolve()
    output_path = args.out.resolve()

    required_files = [
        case_dir / "final_data.pkl",
        case_dir / "optimal_params.pkl",
        case_dir / "best.pth",
        config_path,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    with (case_dir / "final_data.pkl").open("rb") as handle:
        final_data = pickle.load(handle)
    with (case_dir / "optimal_params.pkl").open("rb") as handle:
        optimal_params = pickle.load(handle)
    checkpoint = torch.load(case_dir / "best.pth", map_location="cpu")

    config = load_config(config_path)
    optimal_params = dict(optimal_params)
    if "global_spring_Y" in optimal_params:
        optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
    config.update(optimal_params)

    topology_object_radius = float(config.get("object_radius", 0.02))
    object_max_neighbours = int(config.get("object_max_neighbours", 30))
    topology_controller_radius = float(config.get("controller_radius", 0.04))
    controller_max_neighbours = int(config.get("controller_max_neighbours", 50))
    collision_dist = float(config.get("collision_dist", topology_object_radius))
    collision_object_radius = float(
        config.get("collision_object_radius", collision_dist)
    )
    collision_controller_radius = float(
        config.get("collision_controller_radius", collision_dist)
    )

    structure_points, controller_traj = build_structure_points(final_data)
    controller_points_0 = controller_traj[0] if controller_traj.shape[1] > 0 else None

    points_all, edges, rest_lengths, num_object_springs, used_topology_method = build_springs(
        structure_points=structure_points,
        controller_points_0=controller_points_0,
        object_radius=topology_object_radius,
        object_max_neighbours=object_max_neighbours,
        controller_radius=topology_controller_radius,
        controller_max_neighbours=controller_max_neighbours,
        topology_method=args.topology_method,
    )

    spring_y = checkpoint["spring_Y"]
    if isinstance(spring_y, torch.Tensor):
        spring_y = spring_y.detach().cpu().numpy()
    spring_y = np.asarray(spring_y, dtype=np.float32).reshape(-1)
    if spring_y.shape[0] != edges.shape[0]:
        raise ValueError(
            f"Spring count mismatch: checkpoint={spring_y.shape[0]}, reconstructed={edges.shape[0]}"
        )
    if rest_lengths.shape[0] != edges.shape[0]:
        raise ValueError(
            "Rest-length count mismatch: "
            f"rest_lengths={rest_lengths.shape[0]}, reconstructed={edges.shape[0]}"
        )
    if not np.all(np.isfinite(rest_lengths)):
        raise ValueError("Non-finite spring rest lengths detected.")
    if np.any(rest_lengths <= 0.0):
        min_rest = float(np.min(rest_lengths))
        raise ValueError(f"Non-positive rest lengths detected (min={min_rest}).")

    spring_ke_mode = args.spring_ke_mode
    if spring_ke_mode is None:
        spring_ke_mode = "y_over_rest" if args.strict_phystwin else "raw"

    if spring_ke_mode == "raw":
        spring_ke = spring_y.copy()
    elif spring_ke_mode == "y_over_rest":
        safe_rest = np.maximum(rest_lengths, float(args.rest_length_eps))
        spring_ke = spring_y / safe_rest
    else:
        raise ValueError(f"Unsupported spring_ke mode: {spring_ke_mode}")

    if not np.all(np.isfinite(spring_ke)):
        raise ValueError("Non-finite spring_ke values generated.")

    checkpoint_num_object_springs = int(checkpoint.get("num_object_springs", num_object_springs))
    if checkpoint_num_object_springs != num_object_springs:
        raise ValueError(
            "Object spring count mismatch: "
            f"checkpoint={checkpoint_num_object_springs}, reconstructed={num_object_springs}"
        )

    num_object_points = structure_points.shape[0]
    num_particles = points_all.shape[0]
    num_control_points = num_particles - num_object_points

    is_controller = np.zeros((num_particles,), dtype=np.bool_)
    if num_control_points > 0:
        is_controller[num_object_points:] = True
    controller_idx = np.arange(num_object_points, num_particles, dtype=np.int32)

    mass = np.ones((num_particles,), dtype=np.float32)
    mass[is_controller] = 0.0

    topology_radius = np.full((num_particles,), topology_object_radius, dtype=np.float32)
    if num_control_points > 0:
        topology_radius[num_object_points:] = topology_controller_radius

    collision_radius = np.full(
        (num_particles,), collision_object_radius, dtype=np.float32
    )
    if num_control_points > 0:
        collision_radius[num_object_points:] = collision_controller_radius

    spring_kd = np.full(
        (edges.shape[0],),
        float(config.get("dashpot_damping", 0.0)),
        dtype=np.float32,
    )
    spring_y_min = float(config.get("spring_Y_min", 0.0))
    spring_y_max = float(config.get("spring_Y_max", 1.0e5))
    drag_damping = float(config.get("drag_damping", 0.0))
    spring_is_object = np.zeros((edges.shape[0],), dtype=np.bool_)
    spring_is_object[:num_object_springs] = True

    sim_dt = float(config.get("dt", 5e-5))
    sim_substeps = int(config.get("num_substeps", 1))
    sim_fps = float(config.get("FPS", 30))
    frame_dt = float(1.0 / sim_fps) if sim_fps > 0 else float(sim_dt * sim_substeps)
    self_collision = bool(config.get("self_collision", False))

    collide_elas = to_scalar(
        checkpoint.get("collide_elas"), config.get("collide_elas", 0.5)
    )
    collide_fric = to_scalar(
        checkpoint.get("collide_fric"), config.get("collide_fric", 0.3)
    )
    collide_object_elas = to_scalar(
        checkpoint.get("collide_object_elas"), config.get("collide_object_elas", 0.7)
    )
    collide_object_fric = to_scalar(
        checkpoint.get("collide_object_fric"), config.get("collide_object_fric", 0.3)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ir_version=np.asarray(2, dtype=np.int32),
        case_name=np.asarray(case_dir.name),
        created_at_utc=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        strict_phystwin_export=np.asarray(bool(args.strict_phystwin)),
        spring_ke_mode=np.asarray(spring_ke_mode),
        rest_length_eps=np.asarray(float(args.rest_length_eps), dtype=np.float32),
        topology_method=np.asarray(used_topology_method),
        x0=points_all.astype(np.float32),
        v0=np.zeros_like(points_all, dtype=np.float32),
        mass=mass,
        # Backward-compat alias: legacy readers treat `radius` as particle contact radius.
        radius=collision_radius,
        topology_radius=topology_radius,
        collision_radius=collision_radius,
        is_controller=is_controller,
        controller_idx=controller_idx,
        controller_traj=controller_traj.astype(np.float32),
        spring_edges=edges.astype(np.int32),
        spring_rest_length=rest_lengths.astype(np.float32),
        spring_y=spring_y.astype(np.float32),
        spring_y_min=np.asarray(spring_y_min, dtype=np.float32),
        spring_y_max=np.asarray(spring_y_max, dtype=np.float32),
        spring_ke=spring_ke.astype(np.float32),
        spring_kd=spring_kd,
        drag_damping=np.asarray(drag_damping, dtype=np.float32),
        spring_is_object=spring_is_object,
        num_object_points=np.asarray(num_object_points, dtype=np.int32),
        num_control_points=np.asarray(num_control_points, dtype=np.int32),
        num_object_springs=np.asarray(num_object_springs, dtype=np.int32),
        # Legacy scalar names preserved for topology reconstruction compatibility.
        object_radius=np.asarray(topology_object_radius, dtype=np.float32),
        object_max_neighbours=np.asarray(object_max_neighbours, dtype=np.int32),
        controller_radius=np.asarray(topology_controller_radius, dtype=np.float32),
        controller_max_neighbours=np.asarray(controller_max_neighbours, dtype=np.int32),
        topology_object_radius=np.asarray(topology_object_radius, dtype=np.float32),
        topology_controller_radius=np.asarray(topology_controller_radius, dtype=np.float32),
        collision_object_radius=np.asarray(collision_object_radius, dtype=np.float32),
        collision_controller_radius=np.asarray(collision_controller_radius, dtype=np.float32),
        sim_dt=np.asarray(sim_dt, dtype=np.float32),
        sim_substeps=np.asarray(sim_substeps, dtype=np.int32),
        sim_fps=np.asarray(sim_fps, dtype=np.float32),
        frame_dt=np.asarray(frame_dt, dtype=np.float32),
        reverse_z=np.asarray(bool(config.get("reverse_z", True))),
        self_collision=np.asarray(self_collision),
        contact_collide_elas=np.asarray(collide_elas, dtype=np.float32),
        contact_collide_fric=np.asarray(collide_fric, dtype=np.float32),
        contact_collide_object_elas=np.asarray(collide_object_elas, dtype=np.float32),
        contact_collide_object_fric=np.asarray(collide_object_fric, dtype=np.float32),
        contact_collision_dist=np.asarray(collision_dist, dtype=np.float32),
    )

    summary = {
        "ir_version": 2,
        "case_name": case_dir.name,
        "output": str(output_path),
        "topology_method": used_topology_method,
        "num_particles": int(num_particles),
        "num_object_points": int(num_object_points),
        "num_control_points": int(num_control_points),
        "num_springs": int(edges.shape[0]),
        "num_object_springs": int(num_object_springs),
        "trajectory_frames": int(controller_traj.shape[0]),
        "controller_points_per_frame": int(controller_traj.shape[1]),
        "strict_phystwin_export": bool(args.strict_phystwin),
        "spring_ke_mode": spring_ke_mode,
        "spring_ke_min": float(np.min(spring_ke)),
        "spring_ke_max": float(np.max(spring_ke)),
        "spring_ke_mean": float(np.mean(spring_ke)),
        "spring_y_min": float(spring_y_min),
        "spring_y_max": float(spring_y_max),
        "drag_damping": float(drag_damping),
        "topology_object_radius": float(topology_object_radius),
        "topology_controller_radius": float(topology_controller_radius),
        "collision_object_radius": float(collision_object_radius),
        "collision_controller_radius": float(collision_controller_radius),
        "sim_dt": sim_dt,
        "sim_substeps": sim_substeps,
        "sim_fps": sim_fps,
        "self_collision": bool(self_collision),
    }
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
