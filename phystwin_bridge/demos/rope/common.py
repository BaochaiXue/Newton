#!/usr/bin/env python3
"""Shared helper functions for rope-centric bridge demos."""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from bridge_bootstrap import newton_import_ir


def rope_endpoints(
    edges: np.ndarray,
    n_obj: int,
    points_world: np.ndarray | None = None,
) -> np.ndarray:
    """Return a pair of rope endpoint particle indices.

    If there are more than two degree-1 candidates, the pair with the largest
    geometric distance is chosen.
    """
    degree = np.zeros((n_obj,), dtype=np.int32)
    for i, j in np.asarray(edges, dtype=np.int32):
        if 0 <= i < n_obj and 0 <= j < n_obj:
            degree[i] += 1
            degree[j] += 1
    endpoints = np.flatnonzero(degree == 1)
    if endpoints.size >= 2:
        if endpoints.size == 2:
            return endpoints.astype(np.int32, copy=False)
        pts = None if points_world is None else np.asarray(points_world, dtype=np.float32)
        best_pair = (int(endpoints[0]), int(endpoints[1]))
        best_dist = -1.0
        for i_idx in range(endpoints.size):
            for j_idx in range(i_idx + 1, endpoints.size):
                i = int(endpoints[i_idx])
                j = int(endpoints[j_idx])
                if pts is None:
                    dist = float(abs(j_idx - i_idx))
                else:
                    dist = float(np.linalg.norm(pts[i] - pts[j]))
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        return np.asarray(best_pair, dtype=np.int32)
    return np.asarray([0, max(0, n_obj - 1)], dtype=np.int32)


def anchor_particle_indices(
    shifted_q: np.ndarray,
    endpoint_indices: np.ndarray,
    count_per_end: int,
) -> np.ndarray:
    """Choose the nearest particles around each endpoint to act as anchors."""
    count_per_end = max(1, int(count_per_end))
    selected: list[int] = []
    for endpoint in endpoint_indices.tolist():
        endpoint_pos = shifted_q[int(endpoint)]
        d = np.linalg.norm(shifted_q - endpoint_pos[None, :], axis=1)
        nearest = np.argsort(d)[:count_per_end]
        selected.extend(int(v) for v in nearest.tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def resolve_particle_contact_settings(
    ir_obj: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    """Resolve particle-contact settings from CLI overrides and IR defaults."""
    particle_contacts = (
        bool(args.particle_contacts)
        if args.particle_contacts is not None
        else bool(newton_import_ir.ir_bool(ir_obj, "self_collision", default=False))
    )
    particle_contact_kernel = (
        bool(args.particle_contact_kernel)
        if args.particle_contact_kernel is not None
        else particle_contacts
    )
    if not particle_contacts:
        particle_contact_kernel = False
    return particle_contacts, particle_contact_kernel


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float32)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float32)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


__all__ = [
    "anchor_particle_indices",
    "quat_conjugate",
    "quat_multiply",
    "resolve_particle_contact_settings",
    "rope_endpoints",
]
