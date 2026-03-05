#!/usr/bin/env python3
"""Export a PhysTwin case into a Newton-oriented IR (Intermediate Representation).

This script is the "PhysTwin -> Newton bridge" entry point.

Inputs (from a PhysTwin case directory):
- `final_data.pkl`: point clouds + controller trajectories (and various metadata)
- `optimal_params.pkl`: CMA-ES results (used to overwrite config defaults)
- `best.pth`: trained checkpoint containing per-spring parameters (`spring_Y`)
- `topology.npz`: standalone topology sidecar (the only topology source)
- a config file (`.yaml`): training-time hyperparameters / simulation timing

Output:
- IR v2 `.npz` bundle (plus a small `.json` summary) consumed by `newton_import_ir.py`.

Design notes:
- Topology source is sidecar-only: we read `spring_edges` and
  `spring_rest_lengths` from `topology.npz` and ignore checkpoint topology
  fields even if they exist.
- This enforces a single source of truth for topology and keeps export behavior
  forward-compatible with lean checkpoints.
- IR schema: we export an explicit per-particle `collision_radius` field and avoid
  ambiguous legacy conventions where a single `radius` could refer to topology vs
  contact radius.
- Spring mapping: PhysTwin's "Y" parameter corresponds to Newton's spring stiffness
  `ke` via `ke = Y / rest_length` when using the PhysTwin force form
  `F = Y * (l/rest - 1) * n_hat`.
"""
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
    """Parse a loose YAML-ish scalar string (fallback when PyYAML isn't available)."""
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
    """Load a config file as a dict.

    Prefers PyYAML when available. Falls back to a very small `key: value` parser
    so the bridge remains usable even without `pyyaml` installed.
    """
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
    """Convert tensor/ndarray/python scalar -> python float."""
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


def resolve_scalar_from_sources(
    key: str,
    checkpoint: dict,
    optimal_params: dict,
    config: dict,
    default: float,
) -> tuple[float, str]:
    """Resolve a scalar with explicit source priority.

    Priority:
    1) checkpoint (`best.pth`)
    2) optimization result (`optimal_params.pkl`)
    3) merged config
    4) hardcoded default
    """
    sources = [
        ("checkpoint", checkpoint.get(key)),
        ("optimal_params", optimal_params.get(key)),
        ("config", config.get(key)),
    ]
    for source_name, raw in sources:
        if raw is None:
            continue
        value = to_scalar(raw, default)
        if np.isfinite(value):
            return float(value), source_name
    return float(default), "default"


def resolve_scalar_checkpoint_strict(
    key: str, checkpoint: dict, default: float
) -> tuple[float, str]:
    """Resolve a scalar strictly from checkpoint (PhysTwin inference parity mode).

    PhysTwin inference loads collision coefficients from the selected checkpoint.
    In strict export mode we mirror that behavior and fail fast if fields are missing.
    """
    if key not in checkpoint:
        raise KeyError(
            f"Strict PhysTwin export requires {key!r} in checkpoint (best.pth). "
            "Refusing to fall back to optimal_params/config/default."
        )
    value = to_scalar(checkpoint[key], default)
    if not np.isfinite(value):
        raise ValueError(f"Checkpoint value for {key!r} is non-finite: {value}")
    return float(value), "checkpoint"


def require_config_keys(config: dict, keys: list[str], context: str) -> None:
    """Fail fast when required config keys are missing."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(
            f"{context}: missing required config keys: {missing}. "
            "Add them to YAML/optimal_params or run with --no-strict-phystwin."
        )


def to_array(value, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert tensor/ndarray/list -> numpy array (optionally casting dtype)."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def build_structure_points(final_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the particle set used by PhysTwin/bridge from `final_data.pkl`.

    Returns:
    - `structure_points`: object + surface + interior points (single frame)
    - `controller_points`: controller trajectory `[frames, n_ctrl, 3]`
    """
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


def load_topology_sidecar(topology_path: Path) -> dict:
    """Load topology arrays from a standalone topology.npz file."""
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology sidecar not found: {topology_path}")
    with np.load(topology_path, allow_pickle=False) as bundle:
        if "spring_edges" not in bundle:
            raise KeyError(f"{topology_path} missing required key: spring_edges")
        rest_key = (
            "spring_rest_lengths"
            if "spring_rest_lengths" in bundle
            else "spring_rest_length"
            if "spring_rest_length" in bundle
            else None
        )
        if rest_key is None:
            raise KeyError(
                f"{topology_path} missing required key: spring_rest_lengths"
            )
        loaded: dict[str, np.ndarray] = {
            "spring_edges": np.asarray(bundle["spring_edges"]),
            "spring_rest_lengths": np.asarray(bundle[rest_key]),
        }
        for key in ["init_vertices", "init_masses", "num_object_springs"]:
            if key in bundle:
                loaded[key] = np.asarray(bundle[key])
    return loaded


def parse_args() -> argparse.Namespace:
    """CLI for exporting an IR bundle."""
    parser = argparse.ArgumentParser(
        description="Export a PhysTwin case to a Newton-oriented IR v2 .npz bundle."
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help=(
            "Topology sidecar path (.npz). "
            "If omitted, defaults to <case-dir>/topology.npz."
        ),
    )
    parser.add_argument(
        "--strict-phystwin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enforce PhysTwin-parity export (no silent fallback for critical fields). "
            "When spring-ke-mode is unset, strict mode defaults to y_over_rest. "
            "Use --no-strict-phystwin to allow explicit native-Newton fallback behavior."
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
    topology_path = (
        args.topology.resolve()
        if args.topology is not None
        else (case_dir / "topology.npz").resolve()
    )

    required_files = [
        case_dir / "final_data.pkl",
        case_dir / "optimal_params.pkl",
        case_dir / "best.pth",
        config_path,
        topology_path,
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
        # Older PhysTwin checkpoints used `global_spring_Y`; normalize to the newer key.
        optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
    config.update(optimal_params)

    if args.strict_phystwin:
        require_config_keys(
            config,
            [
                "object_radius",
                "controller_radius",
                "collision_dist",
                "dashpot_damping",
                "drag_damping",
                "dt",
                "num_substeps",
                "FPS",
                "reverse_z",
                "self_collision",
                "spring_Y_min",
                "spring_Y_max",
            ],
            context="strict PhysTwin export",
        )

    # Topology radii from PhysTwin config (topology data itself is loaded separately).
    topology_object_radius = float(config["object_radius"]) if args.strict_phystwin else float(config.get("object_radius", 0.02))
    topology_controller_radius = float(config["controller_radius"]) if args.strict_phystwin else float(config.get("controller_radius", 0.04))

    # Collision radii: used by Newton importer to decide contact radii (separate from topology radii).
    collision_dist = float(config["collision_dist"]) if args.strict_phystwin else float(config.get("collision_dist", topology_object_radius))
    if "collision_object_radius" in config:
        collision_object_radius = float(config["collision_object_radius"])
        collision_object_radius_source = "config"
    else:
        collision_object_radius = collision_dist
        collision_object_radius_source = "derived_from_collision_dist"
    if "collision_controller_radius" in config:
        collision_controller_radius = float(config["collision_controller_radius"])
        collision_controller_radius_source = "config"
    else:
        collision_controller_radius = collision_dist
        collision_controller_radius_source = "derived_from_collision_dist"

    structure_points, controller_traj = build_structure_points(final_data)
    controller_points_0 = controller_traj[0] if controller_traj.shape[1] > 0 else None

    # Learned spring parameter (PhysTwin): one scalar per spring edge.
    spring_y = checkpoint["spring_Y"]
    if isinstance(spring_y, torch.Tensor):
        spring_y = spring_y.detach().cpu().numpy()
    spring_y = np.asarray(spring_y, dtype=np.float32).reshape(-1)

    # Topology is sidecar-only by design.
    checkpoint_mass: np.ndarray | None = None
    topology_source_file = str(topology_path)
    topo = load_topology_sidecar(topology_path)
    edges = np.asarray(topo["spring_edges"], dtype=np.int32)
    rest_lengths = np.asarray(topo["spring_rest_lengths"], dtype=np.float32).reshape(-1)
    num_object_springs = int(
        np.asarray(topo.get("num_object_springs", edges.shape[0])).reshape(-1)[0]
    )
    if args.strict_phystwin and "init_vertices" not in topo:
        raise KeyError(
            "strict PhysTwin export requires topology.npz to contain init_vertices."
        )
    if args.strict_phystwin and "init_masses" not in topo:
        raise KeyError(
            "strict PhysTwin export requires topology.npz to contain init_masses."
        )

    if "init_vertices" in topo:
        points_all = np.asarray(topo["init_vertices"], dtype=np.float32)
    else:
        points_all = (
            np.concatenate([structure_points, controller_points_0], axis=0).astype(np.float32)
            if controller_points_0 is not None
            else structure_points.astype(np.float32)
        )
    if "init_masses" in topo:
        checkpoint_mass = np.asarray(topo["init_masses"], dtype=np.float32).reshape(-1)
    used_topology_method = "topology_sidecar"

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected spring_edges shape [N, 2], got {edges.shape}")
    if rest_lengths.shape[0] != edges.shape[0]:
        raise ValueError(
            "spring_rest_lengths length mismatch: "
            f"rest={rest_lengths.shape[0]}, edges={edges.shape[0]}"
        )
    if points_all.ndim != 2 or points_all.shape[1] != 3:
        raise ValueError(f"Expected init_vertices shape [N, 3], got {points_all.shape}")
    if spring_y.shape[0] != edges.shape[0]:
        raise ValueError(
            f"Spring count mismatch: checkpoint={spring_y.shape[0]}, topology={edges.shape[0]}"
        )
    if rest_lengths.shape[0] != edges.shape[0]:
        raise ValueError(
            "Rest-length count mismatch: "
            f"rest_lengths={rest_lengths.shape[0]}, topology={edges.shape[0]}"
        )
    if not np.all(np.isfinite(rest_lengths)):
        raise ValueError("Non-finite spring rest lengths detected.")
    if np.any(rest_lengths <= 0.0):
        min_rest = float(np.min(rest_lengths))
        raise ValueError(f"Non-positive rest lengths detected (min={min_rest}).")

    # PhysTwin clamps spring_Y at runtime; export the same effective values so Newton
    # consumes the actual spring field used by the source simulator.
    spring_y_min = (
        float(config["spring_Y_min"])
        if args.strict_phystwin
        else float(config.get("spring_Y_min", 0.0))
    )
    spring_y_max = (
        float(config["spring_Y_max"])
        if args.strict_phystwin
        else float(config.get("spring_Y_max", 1.0e5))
    )
    if not np.isfinite(spring_y_min) or not np.isfinite(spring_y_max):
        raise ValueError(
            f"Non-finite spring_Y clamp bounds: min={spring_y_min}, max={spring_y_max}"
        )
    if spring_y_min > spring_y_max:
        raise ValueError(
            f"Invalid spring_Y clamp bounds: min={spring_y_min} > max={spring_y_max}"
        )

    spring_y_raw = spring_y.astype(np.float32, copy=True)
    spring_y_below_min = spring_y_raw < spring_y_min
    spring_y_above_max = spring_y_raw > spring_y_max
    spring_y = np.clip(spring_y_raw, spring_y_min, spring_y_max).astype(
        np.float32, copy=False
    )

    spring_ke_mode = args.spring_ke_mode
    if spring_ke_mode is None:
        spring_ke_mode = "y_over_rest" if args.strict_phystwin else "raw"

    if spring_ke_mode == "raw":
        # Directly treat PhysTwin's effective spring_Y as Newton spring_ke.
        spring_ke = spring_y.copy()
    elif spring_ke_mode == "y_over_rest":
        # PhysTwin force: F = Y * (l/rest - 1) * n_hat
        # Newton spring:  F = ke * (l - rest) * n_hat
        # => ke = Y / rest
        safe_rest = np.maximum(rest_lengths, float(args.rest_length_eps))
        spring_ke = spring_y / safe_rest
    else:
        raise ValueError(f"Unsupported spring_ke mode: {spring_ke_mode}")

    if not np.all(np.isfinite(spring_ke)):
        raise ValueError("Non-finite spring_ke values generated.")

    num_object_points = structure_points.shape[0]
    num_particles = points_all.shape[0]
    num_control_points = num_particles - num_object_points
    if num_control_points != controller_traj.shape[1]:
        raise ValueError(
            "Controller count mismatch: "
            f"from vertices={num_control_points}, from final_data={controller_traj.shape[1]}"
        )

    controller_idx = np.arange(num_object_points, num_particles, dtype=np.int32)

    if checkpoint_mass is not None:
        if checkpoint_mass.shape[0] != num_particles:
            raise ValueError(
                "Checkpoint init_masses length mismatch: "
                f"mass={checkpoint_mass.shape[0]}, particles={num_particles}"
            )
        mass = checkpoint_mass.astype(np.float32, copy=True)
    else:
        # PhysTwin may store masses separately; when missing we default to 1.0 and rely on
        # downstream code to interpret controller particles as kinematic.
        mass = np.ones((num_particles,), dtype=np.float32)
    # Keep controllers kinematic in Newton even if PhysTwin stores all-ones masses.
    mass[controller_idx] = 0.0

    collision_radius = np.full(
        (num_particles,), collision_object_radius, dtype=np.float32
    )
    if num_control_points > 0:
        collision_radius[num_object_points:] = collision_controller_radius

    # Dashpot damping (Newton spring `kd`). PhysTwin uses a global `dashpot_damping` in config.
    spring_kd = np.full(
        (edges.shape[0],),
        float(config["dashpot_damping"]) if args.strict_phystwin else float(config.get("dashpot_damping", 0.0)),
        dtype=np.float32,
    )
    # Global drag damping used in PhysTwin (optional). Newton importer can emulate this.
    drag_damping = float(config["drag_damping"]) if args.strict_phystwin else float(config.get("drag_damping", 0.0))

    # Timing / stepping parameters (used by importer defaults).
    sim_dt = float(config["dt"]) if args.strict_phystwin else float(config.get("dt", 5e-5))
    sim_substeps = int(config["num_substeps"]) if args.strict_phystwin else int(config.get("num_substeps", 1))
    sim_fps = float(config["FPS"]) if args.strict_phystwin else float(config.get("FPS", 30))
    self_collision = bool(config["self_collision"]) if args.strict_phystwin else bool(config.get("self_collision", False))
    reverse_z = bool(config["reverse_z"]) if args.strict_phystwin else bool(config.get("reverse_z", True))

    if args.strict_phystwin:
        collide_elas, collide_elas_source = resolve_scalar_checkpoint_strict(
            key="collide_elas",
            checkpoint=checkpoint,
            default=0.5,
        )
        collide_fric, collide_fric_source = resolve_scalar_checkpoint_strict(
            key="collide_fric",
            checkpoint=checkpoint,
            default=0.3,
        )
    else:
        collide_elas, collide_elas_source = resolve_scalar_from_sources(
            key="collide_elas",
            checkpoint=checkpoint,
            optimal_params=optimal_params,
            config=config,
            default=0.5,
        )
        collide_fric, collide_fric_source = resolve_scalar_from_sources(
            key="collide_fric",
            checkpoint=checkpoint,
            optimal_params=optimal_params,
            config=config,
            default=0.3,
        )
    # PhysTwin particle-particle collision parameters (impulse model).
    # Keep them as optional IR fields so Newton-side experiments can map them
    # to SemiImplicit particle-contact coefficients when explicitly enabled.
    if args.strict_phystwin:
        collide_object_elas, collide_object_elas_source = resolve_scalar_checkpoint_strict(
            key="collide_object_elas",
            checkpoint=checkpoint,
            default=0.7,
        )
        collide_object_fric, collide_object_fric_source = resolve_scalar_checkpoint_strict(
            key="collide_object_fric",
            checkpoint=checkpoint,
            default=0.3,
        )
    else:
        collide_object_elas, collide_object_elas_source = resolve_scalar_from_sources(
            key="collide_object_elas",
            checkpoint=checkpoint,
            optimal_params=optimal_params,
            config=config,
            default=0.7,
        )
        collide_object_fric, collide_object_fric_source = resolve_scalar_from_sources(
            key="collide_object_fric",
            checkpoint=checkpoint,
            optimal_params=optimal_params,
            config=config,
            default=0.3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        # ---- Metadata / versioning ----
        ir_version=np.asarray(2, dtype=np.int32),
        case_name=np.asarray(case_dir.name),
        created_at_utc=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        strict_phystwin_export=np.asarray(bool(args.strict_phystwin)),
        spring_ke_mode=np.asarray(spring_ke_mode),
        rest_length_eps=np.asarray(float(args.rest_length_eps), dtype=np.float32),
        topology_method=np.asarray(used_topology_method),
        # ---- Particles ----
        x0=points_all.astype(np.float32),
        v0=np.zeros_like(points_all, dtype=np.float32),
        mass=mass,
        collision_radius=collision_radius,
        controller_idx=controller_idx,
        controller_traj=controller_traj.astype(np.float32),
        # ---- Springs ----
        spring_edges=edges.astype(np.int32),
        spring_rest_length=rest_lengths.astype(np.float32),
        spring_y_raw=spring_y_raw.astype(np.float32),
        spring_y=spring_y.astype(np.float32),
        spring_ke=spring_ke.astype(np.float32),
        spring_kd=spring_kd,
        spring_y_clamp_min=np.asarray(spring_y_min, dtype=np.float32),
        spring_y_clamp_max=np.asarray(spring_y_max, dtype=np.float32),
        spring_y_clamp_count=np.asarray(
            int(np.count_nonzero(spring_y_below_min | spring_y_above_max)),
            dtype=np.int32,
        ),
        spring_y_clamp_below_count=np.asarray(
            int(np.count_nonzero(spring_y_below_min)),
            dtype=np.int32,
        ),
        spring_y_clamp_above_count=np.asarray(
            int(np.count_nonzero(spring_y_above_max)),
            dtype=np.int32,
        ),
        drag_damping=np.asarray(drag_damping, dtype=np.float32),
        num_object_points=np.asarray(num_object_points, dtype=np.int32),
        # ---- Simulation timing / contact knobs ----
        sim_dt=np.asarray(sim_dt, dtype=np.float32),
        sim_substeps=np.asarray(sim_substeps, dtype=np.int32),
        reverse_z=np.asarray(reverse_z),
        self_collision=np.asarray(self_collision),
        contact_collide_elas=np.asarray(collide_elas, dtype=np.float32),
        contact_collide_fric=np.asarray(collide_fric, dtype=np.float32),
        contact_collision_dist=np.asarray(collision_dist, dtype=np.float32),
        # Optional PhysTwin particle-particle collision parameters.
        contact_collide_object_elas=np.asarray(collide_object_elas, dtype=np.float32),
        contact_collide_object_fric=np.asarray(collide_object_fric, dtype=np.float32),
    )

    summary = {
        "ir_version": 2,
        "case_name": case_dir.name,
        "output": str(output_path),
        "topology_method": used_topology_method,
        "topology_source_file": topology_source_file,
        "num_particles": int(num_particles),
        "num_object_points": int(num_object_points),
        "num_control_points": int(num_control_points),
        "num_springs": int(edges.shape[0]),
        "num_object_springs": int(num_object_springs),
        "trajectory_frames": int(controller_traj.shape[0]),
        "controller_points_per_frame": int(controller_traj.shape[1]),
        "strict_phystwin_export": bool(args.strict_phystwin),
        "spring_ke_mode": spring_ke_mode,
        "spring_y_raw_min": float(np.min(spring_y_raw)),
        "spring_y_raw_max": float(np.max(spring_y_raw)),
        "spring_y_effective_min": float(np.min(spring_y)),
        "spring_y_effective_max": float(np.max(spring_y)),
        "spring_y_clamp_count": int(
            np.count_nonzero(spring_y_below_min | spring_y_above_max)
        ),
        "spring_y_clamp_below_count": int(np.count_nonzero(spring_y_below_min)),
        "spring_y_clamp_above_count": int(np.count_nonzero(spring_y_above_max)),
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
        "collision_object_radius_source": collision_object_radius_source,
        "collision_controller_radius_source": collision_controller_radius_source,
        "sim_dt": sim_dt,
        "sim_substeps": sim_substeps,
        "sim_fps": sim_fps,
        "reverse_z": bool(reverse_z),
        "self_collision": bool(self_collision),
        "contact_collide_elas_source": collide_elas_source,
        "contact_collide_fric_source": collide_fric_source,
        "contact_collide_object_elas": float(collide_object_elas),
        "contact_collide_object_fric": float(collide_object_fric),
        "contact_collide_object_elas_source": collide_object_elas_source,
        "contact_collide_object_fric_source": collide_object_fric_source,
    }
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
