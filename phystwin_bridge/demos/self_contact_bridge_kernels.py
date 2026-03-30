#!/usr/bin/env python3
"""Custom self-contact helpers for bridge-side ablation experiments.

This module intentionally lives outside ``Newton/newton`` so we can study
pair-selection effects without modifying Newton core.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from bridge_bootstrap import newton
from bridge_shared import _pair_penalty_contact_force


def _build_graph_hop_neighbors(
    spring_edges: np.ndarray,
    n_particles: int,
    hops: int,
) -> list[list[int]]:
    if hops <= 0:
        return [[] for _ in range(n_particles)]

    adjacency: list[set[int]] = [set() for _ in range(n_particles)]
    for a_raw, b_raw in np.asarray(spring_edges, dtype=np.int32):
        a = int(a_raw)
        b = int(b_raw)
        if a < 0 or b < 0 or a >= n_particles or b >= n_particles or a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    neighbors: list[list[int]] = []
    for seed in range(n_particles):
        visited = {seed}
        frontier = set(adjacency[seed])
        accum = set(frontier)
        for _ in range(1, max(1, hops)):
            nxt: set[int] = set()
            for node in frontier:
                nxt.update(adjacency[node])
            nxt.difference_update(visited)
            visited.update(frontier)
            nxt.discard(seed)
            accum.update(nxt)
            frontier = nxt
            if not frontier:
                break
        neighbors.append(sorted(accum))
    return neighbors


def build_filtered_self_contact_tables(
    spring_edges: np.ndarray,
    *,
    n_particles: int,
    hops: int,
    device: str,
) -> tuple[wp.array, wp.array, set[tuple[int, int]], dict[str, float]]:
    """Builds exclusion tables for custom self-contact filtering.

    Args:
        spring_edges: Cloth spring graph edges.
        n_particles: Number of cloth particles.
        hops: Graph-hop exclusion radius. ``1`` excludes direct spring neighbors.
        device: Warp device.

    Returns:
        ``(neighbor_table_wp, neighbor_count_wp, excluded_pairs_cpu, summary)``
    """

    hop_lists = _build_graph_hop_neighbors(spring_edges, n_particles, hops=max(0, int(hops)))
    counts = np.asarray([len(row) for row in hop_lists], dtype=np.int32)
    width = max(1, int(counts.max(initial=0)))
    table = np.full((n_particles, width), -1, dtype=np.int32)
    excluded_pairs: set[tuple[int, int]] = set()
    for i, row in enumerate(hop_lists):
        if row:
            table[i, : len(row)] = np.asarray(row, dtype=np.int32)
            excluded_pairs.update((i, int(j)) if i < int(j) else (int(j), i) for j in row if i != int(j))

    neighbor_table_wp = wp.array(table, dtype=wp.int32, device=device)
    neighbor_count_wp = wp.array(counts, dtype=wp.int32, device=device)
    summary = {
        "filter_hops": float(hops),
        "excluded_neighbor_min": float(counts.min(initial=0)),
        "excluded_neighbor_mean": float(counts.mean()) if counts.size else 0.0,
        "excluded_neighbor_median": float(np.median(counts)) if counts.size else 0.0,
        "excluded_neighbor_max": float(counts.max(initial=0)),
        "excluded_pair_count": float(len(excluded_pairs)),
    }
    return neighbor_table_wp, neighbor_count_wp, excluded_pairs, summary


@wp.func
def _is_excluded_neighbor(
    particle_i: int,
    particle_j: int,
    neighbor_table: wp.array2d(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
) -> bool:
    count = neighbor_count[particle_i]
    for k in range(count):
        if neighbor_table[particle_i, k] == particle_j:
            return True
    return False


@wp.kernel
def _eval_filtered_self_contact(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    neighbor_table: wp.array2d(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    f = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        if _is_excluded_neighbor(i, index, neighbor_table, neighbor_count):
            continue

        n = x - particle_x[index]
        d = wp.length(n)
        if d < 1.0e-8:
            continue
        err = d - radius - particle_radius[index]
        if err <= k_cohesion:
            n = n / d
            vrel = v - particle_v[index]
            f += _pair_penalty_contact_force(
                n, vrel, err, k_contact, k_damp, k_friction, k_mu
            )

    particle_f[i] = particle_f[i] + f


def eval_filtered_self_contact_forces(
    model: Any,
    state: Any,
    particle_f: Any,
    grid: wp.HashGrid | None,
    neighbor_table: Any,
    neighbor_count: Any,
) -> None:
    if grid is None:
        return
    wp.launch(
        kernel=_eval_filtered_self_contact,
        dim=model.particle_count,
        inputs=[
            grid.id,
            state.particle_q,
            state.particle_qd,
            model.particle_radius,
            model.particle_flags,
            neighbor_table,
            neighbor_count,
            float(model.particle_ke),
            float(model.particle_kd),
            float(model.particle_kf),
            float(model.particle_mu),
            float(model.particle_cohesion),
            float(model.particle_max_radius),
        ],
        outputs=[particle_f],
        device=model.device,
    )


@wp.kernel
def _apply_force_velocity_update(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        return
    particle_qd[tid] = particle_qd[tid] + particle_f[tid] * particle_inv_mass[tid] * dt


def apply_velocity_update_from_force(
    model: Any,
    state: Any,
    *,
    dt: float,
) -> None:
    wp.launch(
        kernel=_apply_force_velocity_update,
        dim=model.particle_count,
        inputs=[
            state.particle_qd,
            state.particle_f,
            model.particle_inv_mass,
            model.particle_flags,
            float(dt),
        ],
        device=model.device,
    )


@wp.kernel
def _eval_filtered_self_contact_phystwin(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
    neighbor_table: wp.array2d(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    v1 = particle_qd[i]
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        particle_qd_out[i] = v1
        return

    x1 = particle_x[i]
    m1 = particle_mass[i]
    if m1 <= 0.0:
        particle_qd_out[i] = v1
        return

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        if _is_excluded_neighbor(i, index, neighbor_table, neighbor_count):
            continue

        x2 = particle_x[index]
        v2 = particle_qd[index]
        m2 = particle_mass[index]
        if m2 <= 0.0:
            continue

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        if dis_len < collision_dist and wp.dot(dis, relative_v) < -1.0e-4:
            valid_count += 1.0
            collision_normal = dis / wp.max(dis_len, 1.0e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + collide_elas) * v_rel_n) / (1.0 / m1 + 1.0 / m2)
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1.0e-6)
            a = wp.max(
                0.0,
                1.0 - collide_fric * (1.0 + collide_elas) * v_rel_n_length / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)
            J_sum += impulse_n + impulse_t

    if valid_count > 0.0:
        particle_qd_out[i] = v1 - J_sum / valid_count / m1
    else:
        particle_qd_out[i] = v1


def eval_filtered_self_contact_phystwin_velocity(
    model: Any,
    state: Any,
    grid: wp.HashGrid | None,
    neighbor_table: Any,
    neighbor_count: Any,
    *,
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
    particle_qd_out: Any,
) -> None:
    if grid is None:
        return
    wp.launch(
        kernel=_eval_filtered_self_contact_phystwin,
        dim=model.particle_count,
        inputs=[
            grid.id,
            state.particle_q,
            state.particle_qd,
            model.particle_mass,
            model.particle_flags,
            neighbor_table,
            neighbor_count,
            float(collision_dist),
            float(np.clip(collide_elas, 0.0, 1.0)),
            float(np.clip(collide_fric, 0.0, 2.0)),
        ],
        outputs=[particle_qd_out],
        device=model.device,
    )


def compute_nonexcluded_overlap_stats(
    points: np.ndarray,
    radii: np.ndarray,
    excluded_pairs: set[tuple[int, int]],
) -> dict[str, float]:
    """Computes final-frame self-overlap statistics for non-excluded pairs."""

    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64).reshape(-1)
    if pts.shape[0] <= 1:
        return {
            "pair_count": 0.0,
            "max_overlap": 0.0,
            "p95_overlap": 0.0,
        }

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(2.0 * np.max(rad)), output_type="set")
    overlaps: list[float] = []
    for a_raw, b_raw in pairs:
        a = int(a_raw)
        b = int(b_raw)
        pair = (a, b) if a < b else (b, a)
        if pair in excluded_pairs:
            continue
        d = float(np.linalg.norm(pts[a] - pts[b]))
        overlap = float(rad[a] + rad[b] - d)
        if overlap > 0.0:
            overlaps.append(overlap)

    if not overlaps:
        return {
            "pair_count": 0.0,
            "max_overlap": 0.0,
            "p95_overlap": 0.0,
        }

    arr = np.asarray(overlaps, dtype=np.float64)
    return {
        "pair_count": float(arr.size),
        "max_overlap": float(arr.max()),
        "p95_overlap": float(np.quantile(arr, 0.95)),
    }
