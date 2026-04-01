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
def _update_vel_from_force_phystwin(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
    n_object: int,
    dt: float,
    drag_damping: float,
    gravity_mag: float,
    reverse_factor: float,
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = particle_qd[tid]
    if tid >= n_object:
        particle_qd_out[tid] = v0
        return
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        particle_qd_out[tid] = v0
        return

    m0 = particle_mass[tid]
    if m0 <= 0.0:
        particle_qd_out[tid] = v0
        return

    f0 = particle_f[tid]
    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -gravity_mag) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    particle_qd_out[tid] = v1 * drag_damping_factor


def update_vel_from_force_phystwin(
    model: Any,
    state: Any,
    *,
    n_object: int,
    dt: float,
    drag_damping: float,
    gravity_mag: float,
    reverse_factor: float,
    particle_qd_out: Any,
) -> None:
    wp.launch(
        kernel=_update_vel_from_force_phystwin,
        dim=model.particle_count,
        inputs=[
            state.particle_qd,
            state.particle_f,
            model.particle_mass,
            model.particle_flags,
            int(n_object),
            float(dt),
            float(drag_damping),
            float(gravity_mag),
            float(reverse_factor),
        ],
        outputs=[particle_qd_out],
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


def apply_self_collision_phystwin_velocity(
    *,
    device: Any,
    particle_count: int,
    particle_x: Any,
    particle_qd: Any,
    particle_mass: Any,
    particle_flags: Any,
    grid: wp.HashGrid | None,
    neighbor_table: Any,
    neighbor_count: Any,
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
    particle_qd_out: Any,
) -> None:
    if grid is None:
        particle_qd_out.assign(particle_qd)
        return
    wp.launch(
        kernel=_eval_filtered_self_contact_phystwin,
        dim=particle_count,
        inputs=[
            grid.id,
            particle_x,
            particle_qd,
            particle_mass,
            particle_flags,
            neighbor_table,
            neighbor_count,
            float(collision_dist),
            float(np.clip(collide_elas, 0.0, 1.0)),
            float(np.clip(collide_fric, 0.0, 2.0)),
        ],
        outputs=[particle_qd_out],
        device=device,
    )


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
    apply_self_collision_phystwin_velocity(
        device=model.device,
        particle_count=model.particle_count,
        particle_x=state.particle_q,
        particle_qd=state.particle_qd,
        particle_mass=model.particle_mass,
        particle_flags=model.particle_flags,
        grid=grid,
        neighbor_table=neighbor_table,
        neighbor_count=neighbor_count,
        collision_dist=collision_dist,
        collide_elas=collide_elas,
        collide_fric=collide_fric,
        particle_qd_out=particle_qd_out,
    )


@wp.kernel
def _integrate_ground_collision_phystwin(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    n_object: int,
    collide_elas: float,
    collide_fric: float,
    dt: float,
    reverse_factor: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = particle_q[tid]
    v0 = particle_qd[tid]
    if tid >= n_object:
        particle_q_out[tid] = x0
        particle_qd_out[tid] = v0
        return
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        particle_q_out[tid] = x0
        particle_qd_out[tid] = v0
        return

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor
    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1.0e-4:
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1.0e-6)

        clamp_collide_elas = wp.clamp(collide_elas, low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric, low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    particle_q_out[tid] = x0 + v0 * toi + v1 * (dt - toi)
    particle_qd_out[tid] = v1


def integrate_ground_collision_phystwin(
    model: Any,
    particle_q: Any,
    particle_qd: Any,
    *,
    n_object: int,
    collide_elas: float,
    collide_fric: float,
    dt: float,
    reverse_factor: float,
    particle_q_out: Any,
    particle_qd_out: Any,
) -> None:
    wp.launch(
        kernel=_integrate_ground_collision_phystwin,
        dim=model.particle_count,
        inputs=[
            particle_q,
            particle_qd,
            model.particle_flags,
            int(n_object),
            float(np.clip(collide_elas, 0.0, 1.0)),
            float(np.clip(collide_fric, 0.0, 2.0)),
            float(dt),
            float(reverse_factor),
        ],
        outputs=[particle_q_out, particle_qd_out],
        device=model.device,
    )


def reference_filtered_self_contact_phystwin_velocity(
    particle_x: np.ndarray,
    particle_qd: np.ndarray,
    particle_mass: np.ndarray,
    *,
    collision_dist: float,
    collide_elas: float,
    collide_fric: float,
    excluded_pairs: set[tuple[int, int]] | None = None,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Literal NumPy reference for the PhysTwin-style self-collision operator.

    This is intentionally bridge-side and semantically mirrors the Warp kernel
    above so rollout-level parity can be checked in-scope without touching
    `Newton/newton/`.
    """

    x = np.asarray(particle_x, dtype=np.float32)
    qd = np.asarray(particle_qd, dtype=np.float32)
    mass = np.asarray(particle_mass, dtype=np.float32).reshape(-1)
    n_particles = int(x.shape[0])
    out = qd.copy()
    if n_particles <= 1:
        return out

    active = np.ones((n_particles,), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool).reshape(-1)
    valid_particles = active & np.isfinite(mass) & (mass > 0.0)
    valid_idx = np.flatnonzero(valid_particles)
    if valid_idx.size <= 1:
        return out

    tree = cKDTree(x[valid_idx].astype(np.float64, copy=False))
    pair_local = tree.query_pairs(r=float(collision_dist) * 5.0, output_type="ndarray")
    if pair_local.size == 0:
        return out

    pair_idx = valid_idx[np.asarray(pair_local, dtype=np.int64)]
    if excluded_pairs:
        keep_mask = np.fromiter(
            (
                (int(i), int(j)) not in excluded_pairs
                for i, j in pair_idx
            ),
            dtype=bool,
            count=pair_idx.shape[0],
        )
        pair_idx = pair_idx[keep_mask]
        if pair_idx.size == 0:
            return out

    i_idx = pair_idx[:, 0]
    j_idx = pair_idx[:, 1]
    dis = x[j_idx] - x[i_idx]
    dis_len = np.linalg.norm(dis, axis=1)
    relative_v = qd[j_idx] - qd[i_idx]
    approaching = np.sum(dis * relative_v, axis=1) < -1.0e-4
    close = dis_len < float(collision_dist)
    valid_pairs = approaching & close & np.isfinite(dis_len)
    if not np.any(valid_pairs):
        return out

    i_idx = i_idx[valid_pairs]
    j_idx = j_idx[valid_pairs]
    dis = dis[valid_pairs]
    dis_len = np.maximum(dis_len[valid_pairs], 1.0e-6).astype(np.float32, copy=False)
    relative_v = relative_v[valid_pairs].astype(np.float32, copy=False)

    collision_normal = dis / dis_len[:, None]
    v_rel_n_scalar = np.sum(relative_v * collision_normal, axis=1)
    v_rel_n = v_rel_n_scalar[:, None] * collision_normal
    denom = (1.0 / mass[i_idx] + 1.0 / mass[j_idx]).astype(np.float32, copy=False)

    elas = np.float32(np.clip(collide_elas, 0.0, 1.0))
    fric = np.float32(np.clip(collide_fric, 0.0, 2.0))
    impulse_n = (-(1.0 + elas) * v_rel_n) / denom[:, None]

    v_rel_n_length = np.linalg.norm(v_rel_n, axis=1)
    v_rel_t = relative_v - v_rel_n
    v_rel_t_length = np.maximum(np.linalg.norm(v_rel_t, axis=1), 1.0e-6).astype(np.float32, copy=False)
    a = np.maximum(
        0.0,
        1.0 - fric * (1.0 + elas) * v_rel_n_length / v_rel_t_length,
    ).astype(np.float32, copy=False)
    impulse_t = (a[:, None] - 1.0) * v_rel_t / denom[:, None]
    pair_impulse = (impulse_n + impulse_t).astype(np.float32, copy=False)

    impulse_sum = np.zeros_like(out, dtype=np.float32)
    valid_count = np.zeros((n_particles,), dtype=np.float32)
    np.add.at(impulse_sum, i_idx, pair_impulse)
    np.add.at(impulse_sum, j_idx, -pair_impulse)
    np.add.at(valid_count, i_idx, 1.0)
    np.add.at(valid_count, j_idx, 1.0)

    contact_mask = valid_count > 0.0
    if np.any(contact_mask):
        out[contact_mask] = (
            qd[contact_mask]
            - impulse_sum[contact_mask]
            / valid_count[contact_mask, None]
            / mass[contact_mask, None]
        )
    return out.astype(np.float32, copy=False)


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


def compute_nonexcluded_overlap_curve(
    points_seq: np.ndarray,
    radii: np.ndarray,
    excluded_pairs: set[tuple[int, int]],
) -> dict[str, np.ndarray | float]:
    """Computes rollout-level self-overlap statistics for non-excluded pairs.

    Args:
        points_seq: Particle positions over time with shape ``[frames, N, 3]``.
        radii: Per-particle radii.
        excluded_pairs: Pair set that should be ignored when evaluating self-overlap.

    Returns:
        A dictionary containing per-frame curves plus rollout-peak aggregates.
    """

    pts_seq = np.asarray(points_seq, dtype=np.float64)
    if pts_seq.ndim != 3 or pts_seq.shape[-1] != 3:
        raise ValueError(f"points_seq must be [F, N, 3], got {pts_seq.shape}")

    pair_curve = np.zeros((pts_seq.shape[0],), dtype=np.float64)
    max_curve = np.zeros((pts_seq.shape[0],), dtype=np.float64)
    p95_curve = np.zeros((pts_seq.shape[0],), dtype=np.float64)
    nonfinite_mask = np.zeros((pts_seq.shape[0],), dtype=bool)

    for frame_idx, points in enumerate(pts_seq):
        if not np.isfinite(points).all():
            nonfinite_mask[frame_idx] = True
            continue
        stats = compute_nonexcluded_overlap_stats(points, radii, excluded_pairs)
        pair_curve[frame_idx] = float(stats["pair_count"])
        max_curve[frame_idx] = float(stats["max_overlap"])
        p95_curve[frame_idx] = float(stats["p95_overlap"])

    return {
        "pair_count_curve": pair_curve,
        "max_overlap_curve": max_curve,
        "p95_overlap_curve": p95_curve,
        "nonfinite_frame_mask": nonfinite_mask,
        "peak_pair_count": float(pair_curve.max(initial=0.0)),
        "peak_max_overlap": float(max_curve.max(initial=0.0)),
        "peak_p95_overlap": float(p95_curve.max(initial=0.0)),
        "persistent_overlap_frames": float(np.count_nonzero(max_curve > 0.0)),
        "nonfinite_frame_count": float(np.count_nonzero(nonfinite_mask)),
    }
