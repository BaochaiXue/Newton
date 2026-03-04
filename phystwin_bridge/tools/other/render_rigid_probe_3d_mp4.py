#!/usr/bin/env python3
"""Render a true 3D interaction video for rigid-probe runs.

This renders:
- spring-mass object particles (rope) as 3D scatter
- optional sparse spring graph as faint 3D lines
- rigid box as 3D wireframe

Input:
- scene npz from importer rigid probe:
  `newton_import_ir.py --rigid-probe` -> `<prefix>_scene.npz`
- optional IR npz for spring edges (to draw rope structure lines)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


BOX_EDGES = np.asarray(
    [
        [0, 1],
        [0, 2],
        [0, 4],
        [3, 1],
        [3, 2],
        [3, 7],
        [5, 1],
        [5, 4],
        [5, 7],
        [6, 2],
        [6, 4],
        [6, 7],
    ],
    dtype=np.int32,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 3D rigid-box vs rope interaction MP4.")
    p.add_argument("--scene-npz", type=Path, required=True, help="Rigid-probe scene npz")
    p.add_argument("--out-mp4", type=Path, required=True)
    p.add_argument("--ir-npz", type=Path, default=None, help="Optional IR npz (for spring_edges)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--size", type=float, nargs=2, default=(12.8, 7.2), metavar=("W_IN", "H_IN"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-points", type=int, default=1500)
    p.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Scatter marker size for rope particles.",
    )
    p.add_argument(
        "--spring-stride",
        type=int,
        default=40,
        help="Use every k-th spring for structural lines (if --ir-npz provided).",
    )
    p.add_argument("--elev", type=float, default=20.0, help="Camera elevation in degrees.")
    p.add_argument("--azim", type=float, default=-55.0, help="Camera azimuth in degrees.")
    p.add_argument(
        "--draw-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a visible table plane at z=0 for presentation clarity.",
    )
    p.add_argument(
        "--track-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track camera center using midpoint between body center and object COM.",
    )
    p.add_argument(
        "--track-smooth",
        type=float,
        default=0.9,
        help="EMA smoothing for camera center when tracking (0..1).",
    )
    p.add_argument(
        "--window-scale",
        type=float,
        default=1.35,
        help="Scale factor for dynamic camera half-span.",
    )
    p.add_argument(
        "--min-half-span",
        type=float,
        default=0.12,
        help="Lower bound for dynamic camera half-span in world units.",
    )
    p.add_argument(
        "--display-reverse-z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Visualization-only axis flip: render z'=-z to match conventional "
            "upward Z display when simulation uses reverse-z gravity."
        ),
    )
    return p.parse_args()


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.shape[0] < 4:
        raise ValueError(f"Quaternion length must be 4, got {q.shape}")
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n <= 0.0:
        return np.eye(3, dtype=np.float64)
    x /= n
    y /= n
    z /= n
    w /= n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_corners(pos: np.ndarray, quat_xyzw: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    local = np.asarray(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, +hz],
            [-hx, +hy, -hz],
            [-hx, +hy, +hz],
            [+hx, -hy, -hz],
            [+hx, -hy, +hz],
            [+hx, +hy, -hz],
            [+hx, +hy, +hz],
        ],
        dtype=np.float64,
    )
    rot = quat_xyzw_to_rot(quat_xyzw)
    return local @ rot.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)


def mesh_vertices_world(
    pos: np.ndarray, quat_xyzw: np.ndarray, vertices_local: np.ndarray, scale: float
) -> np.ndarray:
    rot = quat_xyzw_to_rot(quat_xyzw)
    v = np.asarray(vertices_local, dtype=np.float64) * float(scale)
    return v @ rot.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)


def build_segments(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # points: [N,3], edges: [M,2] -> [M,2,3]
    return points[edges]


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.track_smooth < 1.0):
        raise ValueError("--track-smooth must be in [0,1).")
    if args.spring_stride < 1:
        raise ValueError("--spring-stride must be >= 1.")
    if args.max_points < 1:
        raise ValueError("--max-points must be >= 1.")

    scene_npz = args.scene_npz.resolve()
    out_mp4 = args.out_mp4.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with np.load(scene_npz, allow_pickle=False) as data:
        q_obj = np.asarray(data["particle_q_object"], dtype=np.float32)  # [F,N,3]
        body_q = np.asarray(data["body_q"], dtype=np.float32)  # [F,7]
        hx = float(np.asarray(data["rigid_hx"]).ravel()[0])
        hy = float(np.asarray(data["rigid_hy"]).ravel()[0])
        hz = float(np.asarray(data["rigid_hz"]).ravel()[0])
        sim_dt = float(np.asarray(data["sim_dt"]).ravel()[0]) if "sim_dt" in data.files else 1.0 / args.fps
        rigid_shape_kind = "box"
        if "rigid_shape_kind" in data.files:
            rigid_shape_kind = str(np.asarray(data["rigid_shape_kind"]).reshape(-1)[0])
        rigid_mesh_vertices_local = None
        rigid_mesh_edges = None
        rigid_mesh_scale = 1.0
        if rigid_shape_kind == "bunny_mesh":
            if (
                "rigid_mesh_vertices_local" in data.files
                and "rigid_mesh_render_edges" in data.files
            ):
                rigid_mesh_vertices_local = np.asarray(
                    data["rigid_mesh_vertices_local"], dtype=np.float32
                ).reshape(-1, 3)
                rigid_mesh_edges = np.asarray(
                    data["rigid_mesh_render_edges"], dtype=np.int32
                ).reshape(-1, 2)
                if "rigid_mesh_scale" in data.files:
                    rigid_mesh_scale = float(np.asarray(data["rigid_mesh_scale"]).ravel()[0])
            else:
                rigid_shape_kind = "box"

    if q_obj.ndim != 3 or q_obj.shape[-1] != 3:
        raise ValueError(f"Unexpected particle_q_object shape: {q_obj.shape}")
    if body_q.ndim != 2 or body_q.shape[0] != q_obj.shape[0] or body_q.shape[1] < 7:
        raise ValueError(f"Unexpected body_q shape: {body_q.shape}")

    q_obj_raw = q_obj.copy()
    body_pos_raw = body_q[:, 0:3].astype(np.float64)
    if args.display_reverse_z:
        q_obj[:, :, 2] *= -1.0

    frames = int(q_obj.shape[0])
    if args.max_frames is not None:
        frames = max(1, min(frames, int(args.max_frames)))
        q_obj = q_obj[:frames]
        q_obj_raw = q_obj_raw[:frames]
        body_q = body_q[:frames]
        body_pos_raw = body_pos_raw[:frames]

    n_obj = int(q_obj.shape[1])
    if n_obj > args.max_points:
        idx = np.linspace(0, n_obj - 1, num=args.max_points, dtype=np.int64)
        q_vis = q_obj[:, idx, :]
    else:
        idx = None
        q_vis = q_obj

    spring_edges = None
    if args.ir_npz is not None:
        with np.load(args.ir_npz.resolve(), allow_pickle=False) as ir:
            if "spring_edges" in ir and "num_object_points" in ir:
                edges = np.asarray(ir["spring_edges"], dtype=np.int32)
                n_ir_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
                keep = (edges[:, 0] < n_ir_obj) & (edges[:, 1] < n_ir_obj)
                edges = edges[keep]
                if idx is not None:
                    # Remap filtered object indices to downsampled index set.
                    remap = -np.ones((n_ir_obj,), dtype=np.int32)
                    remap[idx] = np.arange(idx.shape[0], dtype=np.int32)
                    a = remap[edges[:, 0]]
                    b = remap[edges[:, 1]]
                    ok = (a >= 0) & (b >= 0)
                    edges = np.stack([a[ok], b[ok]], axis=1)
                if edges.shape[0] > 0:
                    spring_edges = edges[:: args.spring_stride].copy()

    body_pos_vis = body_pos_raw.copy()
    if args.display_reverse_z:
        body_pos_vis[:, 2] *= -1.0

    # Base global span for consistent dynamic zoom floor.
    all_pts = q_vis.reshape(-1, 3)
    xyz_min = np.min(all_pts, axis=0)
    xyz_max = np.max(all_pts, axis=0)
    if rigid_shape_kind == "bunny_mesh" and rigid_mesh_vertices_local is not None:
        rigid_extent_radius = float(
            np.max(np.linalg.norm(rigid_mesh_vertices_local * rigid_mesh_scale, axis=1))
        )
    else:
        rigid_extent_radius = float(np.linalg.norm(np.asarray([hx, hy, hz], dtype=np.float64)))
    xyz_min = np.minimum(xyz_min, np.min(body_pos_vis, axis=0) - rigid_extent_radius)
    xyz_max = np.maximum(xyz_max, np.max(body_pos_vis, axis=0) + rigid_extent_radius)
    base_half_span = 0.5 * np.max(xyz_max - xyz_min)
    base_half_span = max(base_half_span, float(args.min_half_span))

    fig = plt.figure(figsize=(float(args.size[0]), float(args.size[1])), dpi=int(args.dpi))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.96, 0.97, 0.99))

    # Initial artists.
    pts0 = q_vis[0]
    scatter = ax.scatter(
        pts0[:, 0],
        pts0[:, 1],
        pts0[:, 2],
        s=float(args.point_size),
        c="#4477AA",
        alpha=0.90,
        depthshade=False,
    )

    # Rigid shape wireframe.
    if rigid_shape_kind == "bunny_mesh" and rigid_mesh_vertices_local is not None and rigid_mesh_edges is not None:
        v0 = mesh_vertices_world(
            body_q[0, 0:3], body_q[0, 3:7], rigid_mesh_vertices_local, rigid_mesh_scale
        )
        if args.display_reverse_z:
            v0[:, 2] *= -1.0
        rigid_segments = v0[rigid_mesh_edges]
        rigid_lc = Line3DCollection(rigid_segments, colors="#D64137", linewidths=0.65, alpha=0.90)
    else:
        c0 = box_corners(body_q[0, 0:3], body_q[0, 3:7], hx, hy, hz)
        if args.display_reverse_z:
            c0[:, 2] *= -1.0
        rigid_segments = c0[BOX_EDGES]
        rigid_lc = Line3DCollection(rigid_segments, colors="#D64137", linewidths=2.5)
    ax.add_collection3d(rigid_lc)

    rope_lc = None
    if spring_edges is not None and spring_edges.shape[0] > 0:
        rope_segments = build_segments(pts0, spring_edges)
        rope_lc = Line3DCollection(rope_segments, colors="#7FA5D6", linewidths=0.35, alpha=0.30)
        ax.add_collection3d(rope_lc)

    body_marker = ax.scatter(
        [body_pos_vis[0, 0]],
        [body_pos_vis[0, 1]],
        [body_pos_vis[0, 2]],
        s=46.0,
        c="#D64137",
        depthshade=False,
    )
    com0 = np.mean(pts0, axis=0)
    com_marker = ax.scatter(
        [com0[0]],
        [com0[1]],
        [com0[2]],
        s=42.0,
        c="#E0AA20",
        depthshade=False,
    )
    connector, = ax.plot(
        [body_pos_vis[0, 0], com0[0]],
        [body_pos_vis[0, 1], com0[1]],
        [body_pos_vis[0, 2], com0[2]],
        color="#777777",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (display=-sim)" if args.display_reverse_z else "Z")

    if args.draw_ground:
        ground_center = 0.5 * (xyz_min + xyz_max)
        ground_half = max(float(base_half_span) * 1.35, 0.25)
        gx = float(ground_center[0])
        gy = float(ground_center[1])
        gz = 0.0
        table = np.asarray(
            [
                [gx - ground_half, gy - ground_half, gz],
                [gx + ground_half, gy - ground_half, gz],
                [gx + ground_half, gy + ground_half, gz],
                [gx - ground_half, gy + ground_half, gz],
            ],
            dtype=np.float64,
        )
        table_poly = Poly3DCollection(
            [table], facecolors="#D5DCE4", edgecolors="#8A94A0", linewidths=0.9, alpha=0.42
        )
        ax.add_collection3d(table_poly)

    # Camera tracking state.
    cam_center = None
    cam_half = base_half_span

    writer = FFMpegWriter(fps=float(args.fps), codec="libx264", bitrate=7000)
    meta_text = fig.text(0.01, 0.02, "", fontsize=10, color="#222222")

    with writer.saving(fig, str(out_mp4), dpi=int(args.dpi)):
        for f in range(frames):
            pts = q_vis[f]
            bpos_raw = body_pos_raw[f]
            bpos = body_pos_vis[f]
            bquat = body_q[f, 3:7].astype(np.float64)
            com = np.mean(pts, axis=0).astype(np.float64)
            raw_com = np.mean(q_obj_raw[f], axis=0).astype(np.float64)
            dist = float(np.linalg.norm(bpos_raw - raw_com))
            speed = 0.0
            if f > 0 and sim_dt > 0.0:
                speed = float(np.linalg.norm(body_pos_raw[f] - body_pos_raw[f - 1]) / sim_dt)

            # Update particle scatter.
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

            # Update rope structural lines.
            if rope_lc is not None:
                rope_lc.set_segments(build_segments(pts, spring_edges))

            # Update rigid shape.
            if (
                rigid_shape_kind == "bunny_mesh"
                and rigid_mesh_vertices_local is not None
                and rigid_mesh_edges is not None
            ):
                v = mesh_vertices_world(
                    bpos_raw, bquat, rigid_mesh_vertices_local, rigid_mesh_scale
                )
                if args.display_reverse_z:
                    v[:, 2] *= -1.0
                rigid_lc.set_segments(v[rigid_mesh_edges])
            else:
                c = box_corners(bpos_raw, bquat, hx, hy, hz)
                if args.display_reverse_z:
                    c[:, 2] *= -1.0
                rigid_lc.set_segments(c[BOX_EDGES])

            # Update body/com markers and connector.
            body_marker._offsets3d = ([bpos[0]], [bpos[1]], [bpos[2]])
            com_marker._offsets3d = ([com[0]], [com[1]], [com[2]])
            connector.set_data_3d([bpos[0], com[0]], [bpos[1], com[1]], [bpos[2], com[2]])

            # Dynamic tracked camera bounds.
            if args.track_center:
                target_center = 0.5 * (bpos + com)
                local_half = max(
                    float(args.min_half_span),
                    float(args.window_scale) * (np.max(np.quantile(pts, 0.95, axis=0) - np.quantile(pts, 0.05, axis=0)) * 0.5 + dist * 0.7 + max(hx, hy, hz)),
                )
                if cam_center is None:
                    cam_center = target_center
                    cam_half = max(base_half_span, local_half)
                else:
                    alpha = float(args.track_smooth)
                    cam_center = alpha * cam_center + (1.0 - alpha) * target_center
                    cam_half = alpha * cam_half + (1.0 - alpha) * max(base_half_span, local_half)
                ax.set_xlim(cam_center[0] - cam_half, cam_center[0] + cam_half)
                ax.set_ylim(cam_center[1] - cam_half, cam_center[1] + cam_half)
                ax.set_zlim(cam_center[2] - cam_half, cam_center[2] + cam_half)
            else:
                ax.set_xlim(xyz_min[0], xyz_max[0])
                ax.set_ylim(xyz_min[1], xyz_max[1])
                ax.set_zlim(xyz_min[2], xyz_max[2])

            rigid_label = "Bunny Mesh" if rigid_shape_kind == "bunny_mesh" else "Rigid Box"
            title = f"{rigid_label} vs Rope Collision (Newton SemiImplicit)"
            if args.display_reverse_z:
                title += " | display z=-sim z"
            ax.set_title(title, fontsize=13, pad=10)
            meta_text.set_text(
                f"frame {f+1}/{frames}  |  body_speed={speed:.3f} m/s  |  dist(body, COM)={dist:.4f} m"
            )
            writer.grab_frame()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
