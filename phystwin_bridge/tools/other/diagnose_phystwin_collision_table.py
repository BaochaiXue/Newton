#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
DEMOS_DIR = BRIDGE_ROOT / "demos"
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from bridge_bootstrap import ensure_bridge_runtime_paths  # noqa: E402

ensure_bridge_runtime_paths()

from self_contact_bridge_kernels import build_potential_self_collision_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict phystwin dynamic-query versus frozen-table runs and "
            "record per-frame candidate/truncation statistics."
        )
    )
    parser.add_argument("--dynamic-report", type=Path, required=True)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_table_stats(
    rollout_npz: Path,
    *,
    collision_dist: float,
    capacity: int,
    device: str,
) -> list[dict[str, float | int]]:
    payload = np.load(rollout_npz)
    q_obj = np.asarray(payload["particle_q_object"], dtype=np.float32)
    n_frames, n_object = int(q_obj.shape[0]), int(q_obj.shape[1])

    with wp.ScopedDevice(device):
        grid = wp.HashGrid(128, 128, 128)
        grid.reserve(n_object)
    object_q_wp = wp.empty(n_object, dtype=wp.vec3, device=device)
    flags_np = np.full(n_object, 1, dtype=np.int32)
    flags_wp = wp.array(flags_np, dtype=wp.int32, device=device)
    collision_indices = wp.array(
        np.full((n_object, int(capacity)), -1, dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    collision_number = wp.zeros(n_object, dtype=wp.int32, device=device)
    candidate_total = wp.zeros(n_object, dtype=wp.int32, device=device)
    candidate_truncated = wp.zeros(n_object, dtype=wp.int32, device=device)

    rows: list[dict[str, float | int]] = []
    for frame_idx in range(n_frames):
        object_q_wp.assign(q_obj[frame_idx].astype(np.float32, copy=False))
        with wp.ScopedDevice(device):
            grid.build(object_q_wp, radius=float(collision_dist) * 5.0)
        build_potential_self_collision_table(
            device=device,
            particle_count=n_object,
            grid=grid,
            particle_x=object_q_wp,
            particle_flags=flags_wp,
            n_object=n_object,
            collision_dist=float(collision_dist),
            collision_table_capacity=int(capacity),
            collision_indices=collision_indices,
            collision_number=collision_number,
            collision_candidate_total=candidate_total,
            collision_candidate_truncated=candidate_truncated,
        )
        count_np = collision_number.numpy().astype(np.int64)
        total_np = candidate_total.numpy().astype(np.int64)
        trunc_np = candidate_truncated.numpy().astype(np.int64)
        rows.append(
            {
                "frame": int(frame_idx),
                "candidate_total_sum": int(total_np.sum()),
                "candidate_stored_sum": int(count_np.sum()),
                "candidate_truncated_sum": int(trunc_np.sum()),
                "candidate_total_max": int(total_np.max(initial=0)),
                "candidate_stored_max": int(count_np.max(initial=0)),
                "candidate_truncated_max": int(trunc_np.max(initial=0)),
            }
        )
    return rows


def _rollout_metrics(report: dict) -> dict[str, float]:
    checks = report["checks"]
    return {
        "rmse_mean": float(checks["rmse_mean"]["value"]),
        "rmse_max": float(checks["rmse_max"]["value"]),
        "first30_rmse": float(checks["first30_rmse"]["value"]),
        "last30_rmse": float(checks["last30_rmse"]["value"]),
    }


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    dynamic_report = _load_json(args.dynamic_report.resolve())
    frozen_report = _load_json(args.frozen_report.resolve())

    dyn_cfg = dynamic_report["rollout_summary"]["config"]
    frz_cfg = frozen_report["rollout_summary"]["config"]
    collision_dist = float(frozen_report["rollout_summary"]["validation"]["phystwin_collision_dist"])
    dyn_capacity = int(dyn_cfg.get("phystwin_collision_table_capacity", 500))
    frz_capacity = int(frz_cfg.get("phystwin_collision_table_capacity", 500))

    dynamic_rows = _frame_table_stats(
        Path(dynamic_report["rollout_npz"]),
        collision_dist=collision_dist,
        capacity=dyn_capacity,
        device=args.device,
    )
    frozen_rows = _frame_table_stats(
        Path(frozen_report["rollout_npz"]),
        collision_dist=collision_dist,
        capacity=frz_capacity,
        device=args.device,
    )

    dynamic_metrics = _rollout_metrics(dynamic_report)
    frozen_metrics = _rollout_metrics(frozen_report)
    rmse_delta = {
        key: float(frozen_metrics[key] - dynamic_metrics[key]) for key in dynamic_metrics
    }

    summary = {
        "dynamic_report": str(args.dynamic_report.resolve()),
        "frozen_report": str(args.frozen_report.resolve()),
        "collision_dist": collision_dist,
        "dynamic_candidate_mode": dynamic_report["rollout_summary"]["validation"]["phystwin_collision_candidate_mode"],
        "frozen_candidate_mode": frozen_report["rollout_summary"]["validation"]["phystwin_collision_candidate_mode"],
        "dynamic_metrics": dynamic_metrics,
        "frozen_metrics": frozen_metrics,
        "rmse_delta_frozen_minus_dynamic": rmse_delta,
        "dynamic_peak_candidate_total_sum": max(row["candidate_total_sum"] for row in dynamic_rows),
        "frozen_peak_candidate_total_sum": max(row["candidate_total_sum"] for row in frozen_rows),
        "dynamic_peak_candidate_truncated_sum": max(row["candidate_truncated_sum"] for row in dynamic_rows),
        "frozen_peak_candidate_truncated_sum": max(row["candidate_truncated_sum"] for row in frozen_rows),
        "frozen_better_or_equal_60": bool(
            frozen_metrics["rmse_mean"] <= dynamic_metrics["rmse_mean"]
            and frozen_metrics["first30_rmse"] <= dynamic_metrics["first30_rmse"]
        ),
    }

    _write(out_dir / "collision_table_diagnostic.json", json.dumps(summary, indent=2) + "\n")
    with (out_dir / "collision_table_per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "frame",
                "candidate_total_sum",
                "candidate_stored_sum",
                "candidate_truncated_sum",
                "candidate_total_max",
                "candidate_stored_max",
                "candidate_truncated_max",
            ],
        )
        writer.writeheader()
        for row in dynamic_rows:
            writer.writerow({"mode": "dynamic", **row})
        for row in frozen_rows:
            writer.writerow({"mode": "frozen", **row})

    _write(
        out_dir / "collision_table_diagnostic.md",
        "\n".join(
            [
                "# Collision Table Diagnostic",
                "",
                f"- dynamic rmse_mean: `{dynamic_metrics['rmse_mean']}`",
                f"- frozen rmse_mean: `{frozen_metrics['rmse_mean']}`",
                f"- rmse_mean delta (frozen - dynamic): `{rmse_delta['rmse_mean']}`",
                f"- dynamic peak truncated candidates/frame: `{summary['dynamic_peak_candidate_truncated_sum']}`",
                f"- frozen peak truncated candidates/frame: `{summary['frozen_peak_candidate_truncated_sum']}`",
                f"- frozen better or equal on rmse_mean + first30: `{summary['frozen_better_or_equal_60']}`",
            ]
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
