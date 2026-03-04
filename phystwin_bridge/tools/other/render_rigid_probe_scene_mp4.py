#!/usr/bin/env python3
"""Render a simple scene MP4 for rigid-probe interaction.

This visualizes:
- spring-mass object particles (scatter)
- one native Newton rigid box (wireframe)

Input is produced by `newton_import_ir.py --rigid-probe` which writes
`<output_prefix>_scene.npz`.

Notes
-----
- This is intentionally a lightweight visualization (no PhysTwin camera overlays).
- We render an orthographic projection (XZ/XY/YZ) for clarity and robustness.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _quat_to_rotmat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.shape[0] < 4:
        raise ValueError(f"Expected quaternion length 4, got {q.shape}")
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n <= 0.0:
        return np.eye(3, dtype=np.float64)
    x /= n
    y /= n
    z /= n
    w /= n

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _box_corners_world(pos: np.ndarray, quat_xyzw: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    corners_body = np.asarray(
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
    R = _quat_to_rotmat_xyzw(quat_xyzw)
    return corners_body @ R.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)


_BOX_EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (3, 1),
    (3, 2),
    (3, 7),
    (5, 1),
    (5, 4),
    (5, 7),
    (6, 2),
    (6, 4),
    (6, 7),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _encode_mp4(frames_dir: Path, fps: float, out_mp4: Path) -> None:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid fps={fps}")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps}",
        "-start_number",
        "0",
        "-i",
        str(frames_dir / "%d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a simple particles+rigid-box scene MP4 from rigid-probe scene npz."
    )
    p.add_argument("--scene-npz", type=Path, required=True, help="*_scene.npz output from rigid probe")
    p.add_argument("--out-mp4", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--view", choices=["xz", "xy", "yz"], default="xz")
    p.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-points", type=int, default=1200)
    p.add_argument("--point-radius", type=int, default=2)
    p.add_argument("--pad-frac", type=float, default=0.08)
    p.add_argument(
        "--camera-mode",
        choices=["global", "track", "global_track"],
        default="global_track",
        help="global: fixed global bounds; track: dynamic zoom; global_track: left global + right tracking panel",
    )
    p.add_argument("--near-radius", type=float, default=0.08, help="3D radius for highlighting near-contact particles")
    p.add_argument("--track-smooth", type=float, default=0.90, help="EMA smoothing for tracking camera [0,1)")
    p.add_argument("--track-min-half-span", type=float, default=0.10, help="Minimum tracked half-span in world units")
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args()


def _to_px_in_rect(
    u: np.ndarray,
    v: np.ndarray,
    *,
    rect: tuple[int, int, int, int],
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, w, h = rect
    su = (w - 1) / max(u1 - u0, 1e-12)
    sv = (h - 1) / max(v1 - v0, 1e-12)
    x = x0 + (u - u0) * su
    y = y0 + (v1 - v) * sv
    return x, y


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    rect: tuple[int, int, int, int],
    points_uv: np.ndarray,
    near_mask: np.ndarray,
    box_uv: np.ndarray,
    body_uv: tuple[float, float],
    com_uv: tuple[float, float],
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    point_radius: int,
    title: str,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    x, y, w, h = rect

    # Panel border and title.
    draw.rectangle((x, y, x + w, y + h), outline=colors["border"], width=2)
    draw.rectangle((x, y, x + w, y + 34), fill=colors["panel_title_bg"])
    draw.text((x + 10, y + 6), title, fill=colors["panel_title_fg"], font=title_font)

    # Draw object points.
    pu = points_uv[:, 0]
    pv = points_uv[:, 1]
    px, py = _to_px_in_rect(pu, pv, rect=rect, u0=u0, u1=u1, v0=v0, v1=v1)

    r = max(1, int(point_radius))
    for i in range(px.shape[0]):
        xi = int(round(px[i]))
        yi = int(round(py[i]))
        if near_mask[i]:
            col = colors["points_near"]
            rr = r + 1
            draw.ellipse((xi - rr, yi - rr, xi + rr, yi + rr), fill=col)
        else:
            draw.ellipse((xi - r, yi - r, xi + r, yi + r), fill=colors["points"])

    # Draw rigid box wireframe.
    bx, by = _to_px_in_rect(
        box_uv[:, 0], box_uv[:, 1], rect=rect, u0=u0, u1=u1, v0=v0, v1=v1
    )
    for i, j in _BOX_EDGES:
        draw.line((float(bx[i]), float(by[i]), float(bx[j]), float(by[j])), fill=colors["box"], width=3)

    # Draw centers and connector.
    body_x, body_y = _to_px_in_rect(
        np.asarray([body_uv[0]], dtype=np.float64),
        np.asarray([body_uv[1]], dtype=np.float64),
        rect=rect,
        u0=u0,
        u1=u1,
        v0=v0,
        v1=v1,
    )
    com_x, com_y = _to_px_in_rect(
        np.asarray([com_uv[0]], dtype=np.float64),
        np.asarray([com_uv[1]], dtype=np.float64),
        rect=rect,
        u0=u0,
        u1=u1,
        v0=v0,
        v1=v1,
    )
    bxp = int(round(body_x[0]))
    byp = int(round(body_y[0]))
    cxp = int(round(com_x[0]))
    cyp = int(round(com_y[0]))
    draw.line((bxp, byp, cxp, cyp), fill=colors["connector"], width=2)
    draw.ellipse((bxp - 5, byp - 5, bxp + 5, byp + 5), fill=colors["box"])
    draw.ellipse((cxp - 5, cyp - 5, cxp + 5, cyp + 5), fill=colors["com"])

    # Mini legend.
    legend_x = x + 10
    legend_y = y + 42
    legend_step = 20
    draw.text((legend_x, legend_y), "blue: object particles", fill=colors["legend"], font=font)
    draw.text((legend_x, legend_y + legend_step), "cyan: near-contact points", fill=colors["legend"], font=font)
    draw.text((legend_x, legend_y + 2 * legend_step), "red: rigid box center", fill=colors["legend"], font=font)
    draw.text((legend_x, legend_y + 3 * legend_step), "yellow: object COM", fill=colors["legend"], font=font)


def main() -> int:
    args = parse_args()
    scene_npz = args.scene_npz.resolve()
    out_mp4 = args.out_mp4.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with np.load(scene_npz, allow_pickle=False) as data:
        q = np.asarray(data["particle_q_object"], dtype=np.float32)  # [F,N,3]
        body_q = np.asarray(data["body_q"], dtype=np.float32)  # [F,7] (assumed: xyz + quat_xyzw)
        hx = float(np.asarray(data["rigid_hx"]).ravel()[0])
        hy = float(np.asarray(data["rigid_hy"]).ravel()[0])
        hz = float(np.asarray(data["rigid_hz"]).ravel()[0])
        sim_dt = float(np.asarray(data["sim_dt"]).ravel()[0]) if "sim_dt" in data.files else 1.0 / max(float(args.fps), 1e-6)

    if q.ndim != 3 or q.shape[-1] != 3:
        raise ValueError(f"Unexpected particle_q_object shape: {q.shape}")
    if body_q.ndim != 2 or body_q.shape[0] != q.shape[0] or body_q.shape[1] < 7:
        raise ValueError(f"Unexpected body_q shape: {body_q.shape} (frames must match particles)")

    frames = int(q.shape[0])
    if args.max_frames is not None:
        frames = max(1, min(frames, int(args.max_frames)))
        q = q[:frames]
        body_q = body_q[:frames]

    n_pts = int(q.shape[1])
    if args.max_points is not None and n_pts > int(args.max_points):
        # Deterministic stride sampling.
        k = int(args.max_points)
        idx = np.linspace(0, n_pts - 1, num=k, dtype=np.int64)
        q_vis = q[:, idx, :]
    else:
        q_vis = q

    view = str(args.view)
    if view == "xz":
        iu, iv = (0, 2)
    elif view == "xy":
        iu, iv = (0, 1)
    elif view == "yz":
        iu, iv = (1, 2)
    else:
        raise ValueError(f"Unknown view: {view}")

    body_pos = body_q[:, 0:3].astype(np.float64)
    obj_com = q_vis.mean(axis=1).astype(np.float64)

    # Compute global bounds (particles + rigid box corners), used by global panel.
    u_min = float(np.min(q_vis[:, :, iu]))
    u_max = float(np.max(q_vis[:, :, iu]))
    v_min = float(np.min(q_vis[:, :, iv]))
    v_max = float(np.max(q_vis[:, :, iv]))
    for f in range(frames):
        pos = body_q[f, 0:3]
        quat = body_q[f, 3:7]
        corners = _box_corners_world(pos, quat, hx, hy, hz)
        u_min = min(u_min, float(np.min(corners[:, iu])))
        u_max = max(u_max, float(np.max(corners[:, iu])))
        v_min = min(v_min, float(np.min(corners[:, iv])))
        v_max = max(v_max, float(np.max(corners[:, iv])))

    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    pad_u = du * float(args.pad_frac)
    pad_v = dv * float(args.pad_frac)
    u0 = u_min - pad_u
    u1 = u_max + pad_u
    v0 = v_min - pad_v
    v1 = v_max + pad_v

    W, H = (int(args.size[0]), int(args.size[1]))

    tmp_dir = out_mp4.parent / f"._frames_{out_mp4.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    font = _load_font(16)
    title_font = _load_font(18)
    bg = (247, 249, 252)
    fg = (20, 24, 31)
    colors = {
        "border": (90, 98, 112),
        "panel_title_bg": (233, 238, 245),
        "panel_title_fg": (20, 24, 31),
        "points": (73, 111, 168),
        "points_near": (0, 185, 230),
        "box": (213, 65, 55),
        "com": (222, 170, 32),
        "connector": (130, 130, 130),
        "legend": (45, 52, 64),
    }

    camera_mode = str(args.camera_mode)
    if camera_mode == "global_track":
        left_rect = (10, 10, W // 2 - 15, H - 20)
        right_rect = (W // 2 + 5, 10, W // 2 - 15, H - 20)
    else:
        left_rect = (10, 10, W - 20, H - 20)
        right_rect = left_rect

    alpha = float(args.track_smooth)
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"--track-smooth must be in [0, 1), got {alpha}")
    track_center_u = None
    track_center_v = None
    track_half = None

    for f in range(frames):
        img = Image.new("RGB", (W, H), color=bg)
        draw = ImageDraw.Draw(img)

        points = q_vis[f].astype(np.float64)
        points_uv = points[:, [iu, iv]]
        body = body_pos[f]
        com = obj_com[f]
        body_uv = (float(body[iu]), float(body[iv]))
        com_uv = (float(com[iu]), float(com[iv]))

        pos = body_q[f, 0:3]
        quat = body_q[f, 3:7]
        corners = _box_corners_world(pos, quat, hx, hy, hz)
        box_uv = corners[:, [iu, iv]]

        near_radius = float(args.near_radius)
        near_mask = np.linalg.norm(points - body.reshape(1, 3), axis=1) <= near_radius

        # Track camera target around body + COM + local object span.
        uv_dist = float(math.hypot(body_uv[0] - com_uv[0], body_uv[1] - com_uv[1]))
        q_u = points_uv[:, 0]
        q_v = points_uv[:, 1]
        obj_span_u = float(np.quantile(q_u, 0.95) - np.quantile(q_u, 0.05))
        obj_span_v = float(np.quantile(q_v, 0.95) - np.quantile(q_v, 0.05))
        target_half = max(
            float(args.track_min_half_span),
            0.60 * max(obj_span_u, obj_span_v) + 0.90 * uv_dist + 5.0 * max(hx, hy, hz),
        )
        target_cu = 0.5 * (body_uv[0] + com_uv[0])
        target_cv = 0.5 * (body_uv[1] + com_uv[1])

        if track_center_u is None:
            track_center_u = target_cu
            track_center_v = target_cv
            track_half = target_half
        else:
            track_center_u = alpha * track_center_u + (1.0 - alpha) * target_cu
            track_center_v = alpha * track_center_v + (1.0 - alpha) * target_cv
            track_half = alpha * float(track_half) + (1.0 - alpha) * target_half

        # Ensure tracked window contains both centers with margin.
        margin = 0.04
        need_half = max(
            abs(body_uv[0] - track_center_u),
            abs(body_uv[1] - track_center_v),
            abs(com_uv[0] - track_center_u),
            abs(com_uv[1] - track_center_v),
        ) + margin
        track_half = max(float(track_half), need_half, float(args.track_min_half_span))

        track_u0 = track_center_u - float(track_half)
        track_u1 = track_center_u + float(track_half)
        track_v0 = track_center_v - float(track_half)
        track_v1 = track_center_v + float(track_half)

        if camera_mode in ("global", "global_track"):
            title = f"Global {view.upper()} view"
            _draw_panel(
                draw,
                rect=left_rect,
                points_uv=points_uv,
                near_mask=near_mask,
                box_uv=box_uv,
                body_uv=body_uv,
                com_uv=com_uv,
                u0=u0,
                u1=u1,
                v0=v0,
                v1=v1,
                point_radius=args.point_radius,
                title=title,
                font=font,
                title_font=title_font,
                colors=colors,
            )

        if camera_mode in ("track", "global_track"):
            rect = right_rect if camera_mode == "global_track" else left_rect
            title = f"Tracked {view.upper()} zoom"
            _draw_panel(
                draw,
                rect=rect,
                points_uv=points_uv,
                near_mask=near_mask,
                box_uv=box_uv,
                body_uv=body_uv,
                com_uv=com_uv,
                u0=track_u0,
                u1=track_u1,
                v0=track_v0,
                v1=track_v1,
                point_radius=args.point_radius,
                title=title,
                font=font,
                title_font=title_font,
                colors=colors,
            )

        body_speed = 0.0
        if f > 0 and sim_dt > 0.0:
            body_speed = float(np.linalg.norm(body_pos[f] - body_pos[f - 1]) / sim_dt)
        com_dist = float(np.linalg.norm(body - com))
        global_header = (
            f"Newton rigid probe scene | frame {f+1}/{frames} | "
            f"dist(body, COM)={com_dist:.4f} m | body_speed={body_speed:.3f} m/s"
        )
        draw.text((16, H - 28), global_header, fill=fg, font=font)

        img.save(tmp_dir / f"{f}.png")

    _encode_mp4(tmp_dir, fps=float(args.fps), out_mp4=out_mp4)

    if not args.keep_frames:
        shutil.rmtree(tmp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
