#!/usr/bin/env python3
"""Profiling helpers for the cloth+bunny realtime playground."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_OP_NAMES = (
    "particle_grid_build",
    "model_collide",
    "eval_spring_forces",
    "eval_triangle_forces",
    "eval_bending_forces",
    "eval_tetrahedra_forces",
    "eval_body_joint_forces",
    "eval_particle_contact_forces",
    "eval_triangle_contact_forces",
    "eval_body_contact_forces",
    "eval_particle_body_contact_forces",
    "integrate_particles",
    "integrate_bodies",
    "drag_correction",
    "solver_step",
    "total_substep",
    "total_step",
)

PROFILE_GROUPS = {
    "internal_force": (
        "eval_spring_forces",
        "eval_triangle_forces",
        "eval_bending_forces",
        "eval_tetrahedra_forces",
        "eval_body_joint_forces",
    ),
    "collision_contact": (
        "particle_grid_build",
        "model_collide",
        "eval_particle_contact_forces",
        "eval_triangle_contact_forces",
        "eval_body_contact_forces",
        "eval_particle_body_contact_forces",
    ),
    "integration": (
        "integrate_particles",
        "integrate_bodies",
    ),
}


def summarize_profile_runs(
    runs: list[dict[str, Any]],
    *,
    op_names: tuple[str, ...] = PROFILE_OP_NAMES,
) -> dict[str, Any]:
    """Summarize repeated profiling episodes into one aggregate payload."""

    summary: dict[str, Any] = {}
    for name in op_names:
        run_means = []
        all_calls = []
        counts = []
        for run in runs:
            samples = run["samples"].get(name, [])
            counts.append(len(samples))
            if samples:
                run_means.append(float(np.mean(samples)))
                all_calls.extend(samples)
        summary[name] = {
            "call_count_total": int(len(all_calls)),
            "call_count_per_run": counts,
            "run_mean_ms": run_means,
            "mean_of_run_means_ms": float(np.mean(run_means)) if run_means else 0.0,
            "std_of_run_means_ms": float(np.std(run_means)) if len(run_means) > 1 else 0.0,
            "mean_over_all_calls_ms": float(np.mean(all_calls)) if all_calls else 0.0,
            "std_over_all_calls_ms": float(np.std(all_calls)) if len(all_calls) > 1 else 0.0,
        }
    return summary


def rank_profile_ops(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Sort the amortized profiling ops by mean wall time."""

    excluded = {"solver_step", "total_substep", "total_step"}
    ranking = []
    for name, stats in aggregate.items():
        if name in excluded:
            continue
        ranking.append(
            {
                "op_name": str(name),
                "mean_of_run_means_ms": float(stats["mean_of_run_means_ms"]),
                "call_count_total": int(stats["call_count_total"]),
            }
        )
    ranking.sort(key=lambda item: item["mean_of_run_means_ms"], reverse=True)
    return ranking


def compute_group_shares(
    aggregate: dict[str, Any],
    *,
    groups: dict[str, tuple[str, ...]] = PROFILE_GROUPS,
) -> dict[str, float]:
    """Compute grouped shares against the profiled substep total."""

    total = float(aggregate.get("total_substep", {}).get("mean_of_run_means_ms", 0.0))
    if total <= 1.0e-12:
        return {name: 0.0 for name in groups}
    shares = {}
    for group_name, op_names in groups.items():
        subtotal = 0.0
        for op_name in op_names:
            subtotal += float(aggregate.get(op_name, {}).get("mean_of_run_means_ms", 0.0))
        shares[group_name] = subtotal / total
    return shares


def write_profile_outputs(args, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write the profiling payload to JSON and CSV outputs."""

    json_path = (
        args.profile_json.resolve()
        if args.profile_json is not None
        else (args.out_dir / f"{args.prefix}_profile.json").resolve()
    )
    csv_path = (
        args.profile_csv.resolve()
        if args.profile_csv is not None
        else (args.out_dir / f"{args.prefix}_profile.csv").resolve()
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "op_name",
                "call_count_total",
                "mean_of_run_means_ms",
                "std_of_run_means_ms",
                "mean_over_all_calls_ms",
                "std_over_all_calls_ms",
            ]
        )
        for name, stats in payload["aggregate"].items():
            writer.writerow(
                [
                    name,
                    stats["call_count_total"],
                    stats["mean_of_run_means_ms"],
                    stats["std_of_run_means_ms"],
                    stats["mean_over_all_calls_ms"],
                    stats["std_over_all_calls_ms"],
                ]
            )
    return json_path, csv_path


__all__ = [
    "PROFILE_GROUPS",
    "PROFILE_OP_NAMES",
    "compute_group_shares",
    "rank_profile_ops",
    "summarize_profile_runs",
    "write_profile_outputs",
]
