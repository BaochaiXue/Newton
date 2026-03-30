#!/usr/bin/env python3
"""Verify bridge-side PhysTwin-style self-collision against a reference law.

The reference uses the same unique-mask self-collision semantics as PhysTwin:
every particle gets a distinct mask, so all non-self pairs are eligible.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = BRIDGE_ROOT / "tools" / "core"
DEMOS_DIR = BRIDGE_ROOT / "demos"
for path in (CORE_DIR, DEMOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_core_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_defaults = _load_core_module("verify_self_collision_path_defaults", CORE_DIR / "path_defaults.py")

from bridge_bootstrap import newton  # noqa: E402
from self_contact_bridge_kernels import eval_filtered_self_contact_phystwin_velocity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify bridge-side PhysTwin-style self-collision operator.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=path_defaults.default_device())
    parser.add_argument("--num-cases", type=int, default=10000)
    parser.add_argument("--min-particles", type=int, default=2)
    parser.add_argument("--max-particles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol-max-abs-dv", type=float, default=1.0e-5)
    parser.add_argument("--tol-median-rel-dv", type=float, default=1.0e-4)
    return parser.parse_args()


def _reference_velocity_update(
    q: np.ndarray,
    qd: np.ndarray,
    masses: np.ndarray,
    *,
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    out = qd.copy()
    elas = float(np.clip(collide_elas, 0.0, 1.0))
    fric = float(np.clip(collide_fric, 0.0, 2.0))

    for i in range(q.shape[0]):
        v1 = qd[i]
        m1 = masses[i]
        valid_count = 0.0
        j_sum = np.zeros((3,), dtype=np.float64)
        for j in range(q.shape[0]):
            if j == i:
                continue
            x2 = q[j]
            v2 = qd[j]
            m2 = masses[j]
            dis = x2 - q[i]
            dis_len = float(np.linalg.norm(dis))
            relative_v = v2 - v1
            if dis_len < collision_dist and float(np.dot(dis, relative_v)) < -1.0e-4:
                valid_count += 1.0
                collision_normal = dis / max(dis_len, 1.0e-6)
                v_rel_n = float(np.dot(relative_v, collision_normal)) * collision_normal
                impulse_n = (-(1.0 + elas) * v_rel_n) / (1.0 / m1 + 1.0 / m2)
                v_rel_n_length = float(np.linalg.norm(v_rel_n))
                v_rel_t = relative_v - v_rel_n
                v_rel_t_length = max(float(np.linalg.norm(v_rel_t)), 1.0e-6)
                a = max(
                    0.0,
                    1.0 - fric * (1.0 + elas) * v_rel_n_length / v_rel_t_length,
                )
                impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)
                j_sum += impulse_n + impulse_t
        if valid_count > 0.0:
            out[i] = v1 - (j_sum / valid_count) / m1
    return out.astype(np.float32)


def _bridge_velocity_update(
    q: np.ndarray,
    qd: np.ndarray,
    masses: np.ndarray,
    *,
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
    device: str,
) -> np.ndarray:
    n = int(q.shape[0])
    with wp.ScopedDevice(device):
        q_wp = wp.array(np.asarray(q, dtype=np.float32), dtype=wp.vec3, device=device)
        qd_wp = wp.array(np.asarray(qd, dtype=np.float32), dtype=wp.vec3, device=device)
        mass_wp = wp.array(np.asarray(masses, dtype=np.float32).reshape(-1), dtype=wp.float32, device=device)
        flags_wp = wp.array(
            np.full((n,), int(newton.ParticleFlags.ACTIVE), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        neighbor_table_wp = wp.array(np.full((n, 1), -1, dtype=np.int32), dtype=wp.int32, device=device)
        neighbor_count_wp = wp.array(np.zeros((n,), dtype=np.int32), dtype=wp.int32, device=device)
        qd_out_wp = wp.empty_like(qd_wp)
        grid = wp.HashGrid(128, 128, 128)
        grid.reserve(n)
        grid.build(q_wp, radius=float(collision_dist) * 5.0)

        model = SimpleNamespace(
            particle_count=n,
            particle_mass=mass_wp,
            particle_flags=flags_wp,
            device=device,
        )
        state = SimpleNamespace(
            particle_q=q_wp,
            particle_qd=qd_wp,
        )
        eval_filtered_self_contact_phystwin_velocity(
            model,
            state,
            grid,
            neighbor_table_wp,
            neighbor_count_wp,
            collision_dist=float(collision_dist),
            collide_elas=float(collide_elas),
            collide_fric=float(collide_fric),
            particle_qd_out=qd_out_wp,
        )
        wp.synchronize_device(device)
        return np.asarray(qd_out_wp.numpy(), dtype=np.float32)


def _generate_case(rng: np.random.Generator, min_particles: int, max_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n = int(rng.integers(min_particles, max_particles + 1))
    collision_dist = float(rng.uniform(0.015, 0.05))
    q = rng.normal(loc=0.0, scale=collision_dist * 0.3, size=(n, 3)).astype(np.float32)
    qd = rng.normal(loc=0.0, scale=1.0, size=(n, 3)).astype(np.float32)
    masses = rng.uniform(0.05, 2.0, size=(n,)).astype(np.float32)
    collide_elas = float(rng.uniform(0.0, 1.0))
    collide_fric = float(rng.uniform(0.0, 2.0))
    return q, qd, masses, collision_dist, collide_elas, collide_fric


def main() -> int:
    args = parse_args()
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = args.device
    rng = np.random.default_rng(int(args.seed))

    max_abs_dv = 0.0
    rel_errors: list[float] = []
    cases_with_contact = 0
    worst_case: dict[str, Any] | None = None

    for case_idx in range(int(args.num_cases)):
        q, qd, masses, collision_dist, collide_elas, collide_fric = _generate_case(
            rng,
            int(args.min_particles),
            int(args.max_particles),
        )
        ref = _reference_velocity_update(
            q,
            qd,
            masses,
            collision_dist=collision_dist,
            collide_elas=collide_elas,
            collide_fric=collide_fric,
        )
        bridge = _bridge_velocity_update(
            q,
            qd,
            masses,
            collision_dist=collision_dist,
            collide_elas=collide_elas,
            collide_fric=collide_fric,
            device=device,
        )
        diff = bridge.astype(np.float64) - ref.astype(np.float64)
        case_max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_abs_dv = max(max_abs_dv, case_max_abs)

        ref_norm = np.linalg.norm(ref.astype(np.float64), axis=1)
        diff_norm = np.linalg.norm(diff, axis=1)
        case_rel = diff_norm / np.maximum(ref_norm, 1.0e-8)
        rel_errors.extend(case_rel.tolist())
        if np.any(np.linalg.norm(ref - qd, axis=1) > 1.0e-7):
            cases_with_contact += 1
        if worst_case is None or case_max_abs > float(worst_case["max_abs_dv"]):
            worst_case = {
                "case_index": int(case_idx),
                "particle_count": int(q.shape[0]),
                "collision_dist": float(collision_dist),
                "collide_elas": float(collide_elas),
                "collide_fric": float(collide_fric),
                "max_abs_dv": float(case_max_abs),
                "median_rel_dv_case": float(np.median(case_rel)) if case_rel.size else 0.0,
            }

    median_rel = float(np.median(np.asarray(rel_errors, dtype=np.float64))) if rel_errors else 0.0
    payload = {
        "num_cases": int(args.num_cases),
        "min_particles": int(args.min_particles),
        "max_particles": int(args.max_particles),
        "seed": int(args.seed),
        "device": str(device),
        "cases_with_contact": int(cases_with_contact),
        "max_abs_dv": float(max_abs_dv),
        "median_rel_dv": float(median_rel),
        "tol_max_abs_dv": float(args.tol_max_abs_dv),
        "tol_median_rel_dv": float(args.tol_median_rel_dv),
        "pass_max_abs_dv": bool(max_abs_dv <= float(args.tol_max_abs_dv)),
        "pass_median_rel_dv": bool(median_rel <= float(args.tol_median_rel_dv)),
        "pass": bool(
            max_abs_dv <= float(args.tol_max_abs_dv)
            and median_rel <= float(args.tol_median_rel_dv)
        ),
        "reference_semantics": "PhysTwin object_collision law under unique-mask self-collision semantics",
        "worst_case": worst_case,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Equivalence JSON: {out_path}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
