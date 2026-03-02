#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(-1)[0])


def _load_ir(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _force_on_i_phystwin(
    xi: np.ndarray,
    xj: np.ndarray,
    vi: np.ndarray,
    vj: np.ndarray,
    rest: np.ndarray,
    spring_y: np.ndarray,
    spring_kd: np.ndarray,
) -> np.ndarray:
    d = xj - xi
    l = np.linalg.norm(d, axis=1)
    l_safe = np.maximum(l, 1.0e-12)
    dir_ij = d / l_safe[:, None]
    v_rel = np.sum((vj - vi) * dir_ij, axis=1)
    return dir_ij * (spring_y * (l_safe / rest - 1.0) + spring_kd * v_rel)[:, None]


def _force_on_i_newton(
    xi: np.ndarray,
    xj: np.ndarray,
    vi: np.ndarray,
    vj: np.ndarray,
    rest: np.ndarray,
    spring_ke: np.ndarray,
    spring_kd: np.ndarray,
) -> np.ndarray:
    d = xj - xi
    l = np.linalg.norm(d, axis=1)
    l_safe = np.maximum(l, 1.0e-12)
    dir_ij = d / l_safe[:, None]
    v_rel = np.sum((vj - vi) * dir_ij, axis=1)
    return dir_ij * (spring_ke * (l_safe - rest) + spring_kd * v_rel)[:, None]


def _verify_single_ir(path: Path, samples: int, seed: int) -> dict[str, object]:
    ir = _load_ir(path)

    required = ["x0", "spring_edges", "spring_rest_length", "spring_kd", "spring_ke"]
    missing = [k for k in required if k not in ir]
    if missing:
        raise KeyError(f"IR missing keys {missing}: {path}")
    if "spring_y" not in ir:
        raise KeyError(f"IR missing spring_y (needed for mapping check): {path}")

    x0 = np.asarray(ir["x0"], dtype=np.float64)
    edges = np.asarray(ir["spring_edges"], dtype=np.int64)
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float64).reshape(-1)
    spring_y = np.asarray(ir["spring_y"], dtype=np.float64).reshape(-1)
    spring_ke = np.asarray(ir["spring_ke"], dtype=np.float64).reshape(-1)
    spring_kd = np.asarray(ir["spring_kd"], dtype=np.float64).reshape(-1)

    n_springs = int(edges.shape[0])
    if not (rest.shape[0] == spring_y.shape[0] == spring_ke.shape[0] == spring_kd.shape[0] == n_springs):
        raise ValueError(
            f"Spring array length mismatch for {path}: "
            f"edges={n_springs}, rest={rest.shape[0]}, y={spring_y.shape[0]}, "
            f"ke={spring_ke.shape[0]}, kd={spring_kd.shape[0]}"
        )
    if np.any(rest <= 0.0):
        raise ValueError(f"Non-positive rest length detected in {path}")

    ke_expected = spring_y / rest
    ke_abs_err = np.abs(spring_ke - ke_expected)
    ke_rel_err = ke_abs_err / np.maximum(np.abs(ke_expected), 1.0e-12)

    rng = np.random.default_rng(seed)
    pick = min(int(samples), n_springs)
    idx = rng.choice(n_springs, size=pick, replace=False)
    i = edges[idx, 0]
    j = edges[idx, 1]

    xi = x0[i].copy()
    xj = x0[j].copy()
    vi = rng.normal(0.0, 0.25, size=xi.shape)
    vj = rng.normal(0.0, 0.25, size=xj.shape)

    # Perturb positions so we test both stretch/compression, not only rest pose.
    xi = xi + rng.normal(0.0, 0.005, size=xi.shape)
    xj = xj + rng.normal(0.0, 0.005, size=xj.shape)

    f_phys = _force_on_i_phystwin(
        xi=xi,
        xj=xj,
        vi=vi,
        vj=vj,
        rest=rest[idx],
        spring_y=spring_y[idx],
        spring_kd=spring_kd[idx],
    )
    f_newton = _force_on_i_newton(
        xi=xi,
        xj=xj,
        vi=vi,
        vj=vj,
        rest=rest[idx],
        spring_ke=spring_ke[idx],
        spring_kd=spring_kd[idx],
    )

    f_diff = np.linalg.norm(f_phys - f_newton, axis=1)
    f_scale = np.maximum(np.linalg.norm(f_phys, axis=1), 1.0e-12)
    f_rel_err = f_diff / f_scale

    case_name = str(np.asarray(ir.get("case_name", np.asarray(path.stem))).reshape(-1)[0])
    ir_version = int(np.asarray(ir.get("ir_version", np.asarray(1))).reshape(-1)[0])
    spring_ke_mode = str(np.asarray(ir.get("spring_ke_mode", np.asarray("unknown"))).reshape(-1)[0])

    return {
        "ir_path": str(path.resolve()),
        "case_name": case_name,
        "ir_version": ir_version,
        "spring_ke_mode": spring_ke_mode,
        "springs_total": n_springs,
        "springs_sampled": pick,
        "mapping_check": {
            "ke_rel_error_mean": float(np.mean(ke_rel_err)),
            "ke_rel_error_max": float(np.max(ke_rel_err)),
            "ke_abs_error_mean": float(np.mean(ke_abs_err)),
            "ke_abs_error_max": float(np.max(ke_abs_err)),
        },
        "force_check": {
            "force_rel_error_mean": float(np.mean(f_rel_err)),
            "force_rel_error_max": float(np.max(f_rel_err)),
            "force_abs_error_mean": float(np.mean(f_diff)),
            "force_abs_error_max": float(np.max(f_diff)),
        },
        "pass": bool(float(np.max(ke_rel_err)) <= 1.0e-4 and float(np.max(f_rel_err)) <= 1.0e-4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify PhysTwin spring formula mapping against Newton spring force form. "
            "Checks ke ~= spring_y/rest and randomized force equivalence."
        )
    )
    parser.add_argument("--ir", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: list[dict[str, object]] = []
    all_passed = True
    for ir_path in args.ir:
        report = _verify_single_ir(path=ir_path.resolve(), samples=args.samples, seed=args.seed)
        reports.append(report)
        all_passed = all_passed and bool(report["pass"])

    print(json.dumps({"passed": all_passed, "reports": reports}, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
