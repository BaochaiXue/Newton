#!/usr/bin/env python3
"""Render rigid-probe scene with Open3D offscreen renderer.

This renderer is presentation-oriented:
- static camera (no camera tracking drift)
- optional display-only z flip (`z_display = -z_sim`) for reverse-z scenes
- optional spring wireframe overlay
- direct MP4 export + optional slow-motion relabel pass
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d


BOX_TRIANGLES = np.asarray(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.int32,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render rigid-probe scene with Open3D.")
    p.add_argument("--scene-npz", type=Path, required=True)
    p.add_argument("--out-mp4", type=Path, required=True)
    p.add_argument("--ir-npz", type=Path, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-points", type=int, default=2500)
    p.add_argument("--spring-stride", type=int, default=3)
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument(
        "--display-reverse-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Visualization-only axis transform: z_display = -z_sim.",
    )
    p.add_argument(
        "--draw-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--slowmo-factor", type=float, default=4.0)
    p.add_argument(
        "--slowmo-label",
        type=str,
        default="Slow Motion x4",
        help="Drawn on video after slow-motion pass.",
    )
    p.add_argument("--camera-scale", type=float, default=1.7)
    p.add_argument("--camera-height-scale", type=float, default=1.0)
    p.add_argument("--min-span", type=float, default=0.25)
    p.add_argument("--camera-eye", type=float, nargs=3, default=None, metavar=("EX", "EY", "EZ"))
    p.add_argument("--camera-lookat", type=float, nargs=3, default=None, metavar=("LX", "LY", "LZ"))
    return p.parse_args()


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    x, y, z, w = map(float, q[:4])
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


def mesh_vertices_world(
    pos: np.ndarray, quat_xyzw: np.ndarray, vertices_local: np.ndarray, scale: float
) -> np.ndarray:
    rot = quat_xyzw_to_rot(quat_xyzw)
    v = np.asarray(vertices_local, dtype=np.float64) * float(scale)
    return v @ rot.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)


def box_vertices_world(pos: np.ndarray, quat_xyzw: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    local = np.asarray(
        [
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, +hy, -hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ],
        dtype=np.float64,
    )
    rot = quat_xyzw_to_rot(quat_xyzw)
    return local @ rot.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)


def maybe_flip_z(points: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return points
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., 2] *= -1.0
    return out


def run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stdout}")


def save_png(path: Path, image: o3d.geometry.Image) -> None:
    ok = o3d.io.write_image(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def make_ground_quad(xmin: float, xmax: float, ymin: float, ymax: float, z: float = 0.0) -> o3d.geometry.TriangleMesh:
    verts = np.asarray(
        [
            [xmin, ymin, z],
            [xmax, ymin, z],
            [xmax, ymax, z],
            [xmin, ymax, z],
        ],
        dtype=np.float64,
    )
    tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(tris)
    m.compute_vertex_normals()
    return m


def main() -> int:
    args = parse_args()

    if args.max_points < 1:
        raise ValueError("--max-points must be >= 1")
    if args.spring_stride < 1:
        raise ValueError("--spring-stride must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.slowmo_factor <= 0:
        raise ValueError("--slowmo-factor must be > 0")

    scene_npz = args.scene_npz.resolve()
    out_mp4 = args.out_mp4.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with np.load(scene_npz, allow_pickle=False) as data:
        q_obj = np.asarray(data["particle_q_object"], dtype=np.float32)  # [F,N,3]
        body_q = np.asarray(data["body_q"], dtype=np.float32)  # [F,7]
        hx = float(np.asarray(data["rigid_hx"]).ravel()[0])
        hy = float(np.asarray(data["rigid_hy"]).ravel()[0])
        hz = float(np.asarray(data["rigid_hz"]).ravel()[0])
        rigid_shape_kind = str(np.asarray(data.get("rigid_shape_kind", np.asarray("box"))).reshape(-1)[0])

        rigid_mesh_vertices_local = None
        rigid_mesh_indices = None
        rigid_mesh_scale = 1.0
        if rigid_shape_kind == "bunny_mesh":
            if "rigid_mesh_vertices_local" in data.files and "rigid_mesh_indices" in data.files:
                rigid_mesh_vertices_local = np.asarray(data["rigid_mesh_vertices_local"], dtype=np.float32).reshape(-1, 3)
                rigid_mesh_indices = np.asarray(data["rigid_mesh_indices"], dtype=np.int32).reshape(-1, 3)
                if "rigid_mesh_scale" in data.files:
                    rigid_mesh_scale = float(np.asarray(data["rigid_mesh_scale"]).ravel()[0])
            else:
                rigid_shape_kind = "box"

    if q_obj.ndim != 3 or q_obj.shape[-1] != 3:
        raise ValueError(f"Unexpected particle_q_object shape: {q_obj.shape}")
    if body_q.ndim != 2 or body_q.shape[1] < 7 or body_q.shape[0] != q_obj.shape[0]:
        raise ValueError(f"Unexpected body_q shape: {body_q.shape}")

    frames = int(q_obj.shape[0])
    if args.max_frames is not None:
        frames = max(1, min(frames, int(args.max_frames)))
        q_obj = q_obj[:frames]
        body_q = body_q[:frames]

    n_obj = int(q_obj.shape[1])
    if n_obj > args.max_points:
        keep_idx = np.linspace(0, n_obj - 1, num=args.max_points, dtype=np.int64)
        q_vis = q_obj[:, keep_idx, :]
    else:
        keep_idx = None
        q_vis = q_obj

    spring_edges = None
    if args.ir_npz is not None:
        with np.load(args.ir_npz.resolve(), allow_pickle=False) as ir:
            if "spring_edges" in ir and "num_object_points" in ir:
                edges = np.asarray(ir["spring_edges"], dtype=np.int32).reshape(-1, 2)
                n_ir_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
                edges = edges[(edges[:, 0] < n_ir_obj) & (edges[:, 1] < n_ir_obj)]
                if keep_idx is not None:
                    remap = -np.ones((n_ir_obj,), dtype=np.int32)
                    remap[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int32)
                    a = remap[edges[:, 0]]
                    b = remap[edges[:, 1]]
                    ok = (a >= 0) & (b >= 0)
                    edges = np.stack([a[ok], b[ok]], axis=1)
                if edges.shape[0] > 0:
                    spring_edges = edges[:: args.spring_stride].copy()

    # Camera bounds in display coordinates.
    q_vis_disp = maybe_flip_z(q_vis, args.display_reverse_z)
    body_pos_raw = body_q[:, 0:3].astype(np.float64)
    body_pos_disp = maybe_flip_z(body_pos_raw, args.display_reverse_z)

    xyz_min = np.min(q_vis_disp.reshape(-1, 3), axis=0)
    xyz_max = np.max(q_vis_disp.reshape(-1, 3), axis=0)

    if rigid_shape_kind == "bunny_mesh" and rigid_mesh_vertices_local is not None:
        r = float(np.max(np.linalg.norm(rigid_mesh_vertices_local * rigid_mesh_scale, axis=1)))
    else:
        r = float(np.linalg.norm(np.asarray([hx, hy, hz], dtype=np.float64)))
    xyz_min = np.minimum(xyz_min, np.min(body_pos_disp, axis=0) - r)
    xyz_max = np.maximum(xyz_max, np.max(body_pos_disp, axis=0) + r)

    center = 0.5 * (xyz_min + xyz_max)
    xy_span = max(float(np.max(xyz_max[:2] - xyz_min[:2])), float(args.min_span))
    z_span = max(float(xyz_max[2] - xyz_min[2]), float(args.min_span))
    span = max(xy_span, z_span)
    lookat = np.asarray(
        [
            center[0],
            center[1],
            xyz_min[2] + 0.35 * z_span,
        ],
        dtype=np.float64,
    )
    eye = np.asarray(
        [
            center[0] + args.camera_scale * 2.6 * xy_span,
            center[1] - args.camera_scale * 2.2 * xy_span,
            xyz_min[2] + args.camera_height_scale * 0.95 * z_span,
        ],
        dtype=np.float64,
    )
    if args.camera_lookat is not None:
        lookat = np.asarray(args.camera_lookat, dtype=np.float64)
    if args.camera_eye is not None:
        eye = np.asarray(args.camera_eye, dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    # Open3D renderer + materials.
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(args.width), int(args.height))
    scene = renderer.scene
    scene.set_background(np.asarray([0.94, 0.95, 0.97, 1.0], dtype=np.float32))
    scene.scene.set_sun_light(
        np.asarray([0.35, 0.5, -1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        75000.0,
    )
    scene.scene.enable_sun_light(True)
    scene.scene.enable_indirect_light(False)
    scene.camera.look_at(lookat, eye, up)

    mat_points = o3d.visualization.rendering.MaterialRecord()
    mat_points.shader = "defaultUnlit"
    mat_points.base_color = (0.25, 0.48, 0.74, 1.0)
    mat_points.point_size = float(args.point_size)

    mat_lines = o3d.visualization.rendering.MaterialRecord()
    mat_lines.shader = "unlitLine"
    mat_lines.base_color = (0.55, 0.67, 0.83, 1.0)
    mat_lines.line_width = 2.5

    mat_rigid = o3d.visualization.rendering.MaterialRecord()
    mat_rigid.shader = "defaultLit"
    mat_rigid.base_color = (0.86, 0.28, 0.23, 1.0)
    mat_rigid.roughness_img = None

    mat_ground = o3d.visualization.rendering.MaterialRecord()
    mat_ground.shader = "defaultLit"
    mat_ground.base_color = (0.80, 0.83, 0.87, 0.95)

    if args.draw_ground:
        ghalf = max(0.75 * span, 0.35)
        ground = make_ground_quad(
            center[0] - ghalf, center[0] + ghalf, center[1] - ghalf, center[1] + ghalf, z=0.0
        )
        scene.add_geometry("ground", ground, mat_ground)

    # Render frames to temporary PNGs.
    tmp_root = Path(tempfile.mkdtemp(prefix="open3d_probe_frames_"))
    try:
        dyn_names = ("rope_points", "rope_lines", "rigid_mesh")
        for f in range(frames):
            for n in dyn_names:
                if scene.has_geometry(n):
                    scene.remove_geometry(n)

            pts = q_vis[f].astype(np.float64)
            pts = maybe_flip_z(pts, args.display_reverse_z)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            scene.add_geometry("rope_points", pcd, mat_points)

            if spring_edges is not None and spring_edges.shape[0] > 0:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(pts)
                ls.lines = o3d.utility.Vector2iVector(spring_edges.astype(np.int32))
                scene.add_geometry("rope_lines", ls, mat_lines)

            bpos = body_q[f, 0:3].astype(np.float64)
            bquat = body_q[f, 3:7].astype(np.float64)

            if rigid_shape_kind == "bunny_mesh" and rigid_mesh_vertices_local is not None and rigid_mesh_indices is not None:
                v = mesh_vertices_world(bpos, bquat, rigid_mesh_vertices_local, rigid_mesh_scale)
                v = maybe_flip_z(v, args.display_reverse_z)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(v)
                mesh.triangles = o3d.utility.Vector3iVector(rigid_mesh_indices.astype(np.int32))
                mesh.compute_vertex_normals()
            else:
                v = box_vertices_world(bpos, bquat, hx, hy, hz)
                v = maybe_flip_z(v, args.display_reverse_z)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(v)
                mesh.triangles = o3d.utility.Vector3iVector(BOX_TRIANGLES)
                mesh.compute_vertex_normals()
            scene.add_geometry("rigid_mesh", mesh, mat_rigid)

            img = renderer.render_to_image()
            save_png(tmp_root / f"frame_{f:05d}.png", img)

        # Compose base MP4.
        base_mp4 = out_mp4.with_name(out_mp4.stem + "_base.mp4")
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(float(args.fps)),
                "-i",
                str(tmp_root / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(base_mp4),
            ]
        )

        # Optional slow-motion relabeling.
        if abs(args.slowmo_factor - 1.0) > 1e-8:
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            vf = (
                f"setpts={args.slowmo_factor}*PTS,"
                f"drawtext=fontfile={font}:text='{args.slowmo_label}':"
                "x=40:y=40:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
            )
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(base_mp4),
                    "-vf",
                    vf,
                    "-an",
                    str(out_mp4),
                ]
            )
            base_mp4.unlink(missing_ok=True)
        else:
            if out_mp4 != base_mp4:
                base_mp4.replace(out_mp4)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(str(out_mp4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
