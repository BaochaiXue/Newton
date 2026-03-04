#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script directly from `tools/other/` while reusing shared
# defaults/utilities that live under `tools/core/`.
_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from path_defaults import repo_root


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping at {path}")
    return loaded


def _load_metadata(case_dir: Path) -> dict:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    metadata = _load_json(metadata_path)
    if "intrinsics" not in metadata:
        raise KeyError(f"metadata.json missing 'intrinsics': {metadata_path}")
    if "WH" not in metadata:
        raise KeyError(f"metadata.json missing 'WH': {metadata_path}")
    return metadata


def _load_w2c(case_dir: Path) -> np.ndarray:
    calib_path = case_dir / "calibrate.pkl"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibrate.pkl: {calib_path}")
    with calib_path.open("rb") as handle:
        c2ws = pickle.load(handle)
    w2cs = np.asarray(
        [np.linalg.inv(np.asarray(c2w, dtype=np.float64)) for c2w in c2ws],
        dtype=np.float64,
    )
    if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected w2c shape from {calib_path}: {w2cs.shape}")
    return w2cs


def _load_rollout_positions(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout npz: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "particle_q_object" in data:
            positions = data["particle_q_object"]
        elif "particle_q_all" in data:
            positions = data["particle_q_all"]
        else:
            raise KeyError(f"{path} missing particle positions.")
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Unexpected rollout shape in {path}: {positions.shape}")
    return positions


def _load_inference(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing inference.pkl: {path}")
    with path.open("rb") as handle:
        arr = np.asarray(pickle.load(handle), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected inference shape in {path}: {arr.shape}")
    return arr


def _parse_camera_indices(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != 3:
        raise ValueError(f"--camera-indices expects exactly 3 indices, got {value!r}")
    indices = [int(item) for item in items]
    if len(set(indices)) != 3:
        raise ValueError(f"--camera-indices must be distinct, got {indices}")
    return indices


def _parse_rgb(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB format 'R,G,B', got {value!r}")
    try:
        rgb = np.asarray([int(p) for p in parts], dtype=np.int32)
    except ValueError as exc:
        raise ValueError(f"Invalid RGB integer triplet: {value!r}") from exc
    if np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError(f"RGB values must be in [0,255], got {value!r}")
    return rgb.astype(np.uint8)


def _rgb_to_color_name(rgb_u8: np.ndarray) -> str:
    """Map an RGB triplet to a short, human-readable color name.

    The goal is to keep video labels short (avoid printing numeric RGB values).
    We use a small palette and pick the nearest color by L2 distance.
    """

    rgb = np.asarray(rgb_u8, dtype=np.int32).reshape(3)
    palette = {
        "Cyan": np.asarray([0, 255, 255], dtype=np.int32),
        "Orange": np.asarray([255, 180, 0], dtype=np.int32),
        "Red": np.asarray([255, 0, 0], dtype=np.int32),
        "Green": np.asarray([0, 255, 0], dtype=np.int32),
        "Blue": np.asarray([0, 0, 255], dtype=np.int32),
        "Magenta": np.asarray([255, 0, 255], dtype=np.int32),
        "Yellow": np.asarray([255, 255, 0], dtype=np.int32),
        "White": np.asarray([255, 255, 255], dtype=np.int32),
        "Black": np.asarray([0, 0, 0], dtype=np.int32),
        "Gray": np.asarray([128, 128, 128], dtype=np.int32),
    }
    best_name = "Custom"
    best_dist = None
    for name, ref in palette.items():
        dist = int(np.sum((rgb - ref) ** 2))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _project_points(
    points_world: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    y_sign: float,
    z_sign: float,
    z_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_world.astype(np.float64), ones], axis=1)
    cam_h = (w2c @ points_h.T).T
    x = cam_h[:, 0]
    y = cam_h[:, 1] * float(y_sign)
    z = cam_h[:, 2] * float(z_sign)
    valid = z > float(z_eps)
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty
    x = x[valid]
    y = y[valid]
    z = z[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return u, v, z


def _choose_projection_convention(
    sample_points_world: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    best = None
    best_score = -1
    for y_sign in (1.0, -1.0):
        for z_sign in (1.0, -1.0):
            u, v, _z = _project_points(
                sample_points_world, w2c, K, y_sign=y_sign, z_sign=z_sign, z_eps=1e-6
            )
            if u.size == 0:
                score = 0
            else:
                ui = np.rint(u).astype(np.int64)
                vi = np.rint(v).astype(np.int64)
                in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
                score = int(np.sum(in_bounds))
            if score > best_score:
                best_score = score
                best = (y_sign, z_sign)
    assert best is not None
    return best


def _circle_offsets(radius: int) -> list[tuple[int, int]]:
    radius = int(max(0, radius))
    if radius == 0:
        return [(0, 0)]
    r_sq = radius * radius
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r_sq:
                offsets.append((dy, dx))
    return offsets


def _draw_points(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    colors_u8: np.ndarray,
    radius: int,
) -> None:
    if u.size == 0:
        return
    height, width = image.shape[0], image.shape[1]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(in_bounds):
        return
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]
    colors_u8 = colors_u8[in_bounds]
    order = np.argsort(z)[::-1]
    ui = ui[order]
    vi = vi[order]
    colors_u8 = colors_u8[order]
    offsets = _circle_offsets(radius)
    for point_idx in range(ui.shape[0]):
        x0 = int(ui[point_idx])
        y0 = int(vi[point_idx])
        color = colors_u8[point_idx]
        for dy, dx in offsets:
            x = x0 + dx
            y = y0 + dy
            if 0 <= x < width and 0 <= y < height:
                image[y, x, :] = color


def _encode_mp4(frames_dir: Path, fps: float, out_mp4: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{float(fps)}",
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
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _compose_1x3_with_labels(
    panel_mp4s: list[Path],
    camera_indices: list[int],
    label_text: str,
    out_mp4: Path,
) -> None:
    if len(panel_mp4s) != 3:
        raise ValueError(f"Expected 3 panel videos, got {len(panel_mp4s)}")
    labels = [
        f"{label_text} | View {camera_indices[0]}",
        f"{label_text} | View {camera_indices[1]}",
        f"{label_text} | View {camera_indices[2]}",
    ]
    filters: list[str] = []
    for idx, text in enumerate(labels):
        safe_text = text.replace("'", "\\'")
        filters.append(
            f"[{idx}:v]drawtext=text='{safe_text}':"
            "x=20:y=20:fontsize=32:fontcolor=white:"
            "box=1:boxcolor=black@0.6:boxborderw=8[v"
            f"{idx}]"
        )
    filters.append("[v0][v1][v2]xstack=inputs=3:layout=0_0|w0_0|w0+w1_0[vout]")
    filter_complex = ";".join(filters)
    cmd = ["ffmpeg", "-y"]
    for panel in panel_mp4s:
        cmd.extend(["-i", str(panel)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_mp4),
        ]
    )
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a labeled 1x3 no-background comparison video: "
            "each panel overlays Newton rollout points and PhysTwin rollout points."
        )
    )
    parser.add_argument("--report", type=Path, required=True, help="Rollout report JSON path.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Optional case dir override.")
    parser.add_argument(
        "--inference-pkl",
        type=Path,
        default=None,
        help="Optional inference.pkl override. Defaults to report value or case dir.",
    )
    parser.add_argument("--camera-indices", default="0,1,2")
    parser.add_argument(
        "--out-mp4",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to <rollout_stem>_overlay_1x3_labeled.mp4",
    )
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument(
        "--newton-color",
        default="0,255,255",
        help="RGB color for Newton points as R,G,B (default cyan).",
    )
    parser.add_argument(
        "--phystwin-color",
        default="255,180,0",
        help="RGB color for PhysTwin points as R,G,B (default orange).",
    )
    parser.add_argument("--newton-label", default="Newton Rollout")
    parser.add_argument("--phystwin-label", default="PhysTwin Rollout")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report = _load_json(report_path)

    rollout_npz = Path(str(report["rollout_npz"])).resolve()
    case_name = str(report["case_name"])

    root = repo_root()
    case_dir = (
        args.case_dir.resolve()
        if args.case_dir is not None
        else (root / "phystwin_bridge" / "inputs" / "cases" / case_name).resolve()
    )
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case-dir: {case_dir}")

    if args.inference_pkl is not None:
        inference_pkl = args.inference_pkl.resolve()
    elif isinstance(report.get("inference_path"), str) and report["inference_path"]:
        inference_pkl = Path(str(report["inference_path"])).resolve()
    else:
        inference_pkl = (case_dir / "inference.pkl").resolve()

    camera_indices = _parse_camera_indices(args.camera_indices)
    newton_color = _parse_rgb(args.newton_color)
    phystwin_color = _parse_rgb(args.phystwin_color)

    newton_rollout = _load_rollout_positions(rollout_npz)
    phystwin_rollout = _load_inference(inference_pkl)

    frames_to_render = min(newton_rollout.shape[0], phystwin_rollout.shape[0])
    if args.max_frames is not None:
        frames_to_render = min(frames_to_render, int(args.max_frames))
    if frames_to_render <= 0:
        raise ValueError(
            f"No frames to render: newton={newton_rollout.shape[0]}, phystwin={phystwin_rollout.shape[0]}"
        )

    # Compare only points shared by both rollouts.
    points_to_render = min(newton_rollout.shape[1], phystwin_rollout.shape[1])
    if points_to_render <= 0:
        raise ValueError("No points available to render.")
    newton_rollout = newton_rollout[:frames_to_render, :points_to_render].astype(np.float32)
    phystwin_rollout = phystwin_rollout[:frames_to_render, :points_to_render].astype(np.float32)

    metadata = _load_metadata(case_dir)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected intrinsics shape in metadata.json: {intrinsics.shape}")
    wh = metadata["WH"]
    if not (isinstance(wh, list) and len(wh) == 2):
        raise ValueError(f"Unexpected WH in metadata.json: {wh!r}")
    width = int(wh[0])
    height = int(wh[1])
    fps = float(metadata.get("fps", 30.0))

    w2cs = _load_w2c(case_dir)

    out_mp4 = (
        args.out_mp4.resolve()
        if args.out_mp4 is not None
        else rollout_npz.with_name(f"{rollout_npz.stem}_overlay_1x3_labeled.mp4")
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_mp4.parent / f"._tmp_overlay1x3_{rollout_npz.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    panel_mp4s: list[Path] = []
    newton_color_name = _rgb_to_color_name(newton_color)
    phystwin_color_name = _rgb_to_color_name(phystwin_color)
    # Keep the label short so it doesn't overflow the panel width.
    # (Users asked to avoid printing numeric RGB values.)
    label_text = f"{newton_color_name}=Newton, {phystwin_color_name}=PhysTwin"

    try:
        for cam in camera_indices:
            if cam < 0 or cam >= intrinsics.shape[0]:
                raise ValueError(
                    f"--camera-indices contains {cam}, out of intrinsics range [0,{intrinsics.shape[0]-1}]"
                )
            if cam < 0 or cam >= w2cs.shape[0]:
                raise ValueError(
                    f"--camera-indices contains {cam}, out of calibrate camera range [0,{w2cs.shape[0]-1}]"
                )
            K = intrinsics[cam]
            w2c = w2cs[cam]

            sample_points = np.concatenate([newton_rollout[0], phystwin_rollout[0]], axis=0)
            y_sign, z_sign = _choose_projection_convention(
                sample_points_world=sample_points,
                w2c=w2c,
                K=K,
                width=width,
                height=height,
            )

            panel_frames = tmp_dir / f"frames_cam{cam}"
            panel_frames.mkdir(parents=True, exist_ok=False)
            panel_mp4 = tmp_dir / f"overlay_cam{cam}.mp4"
            panel_mp4s.append(panel_mp4)

            for frame_idx in range(frames_to_render):
                image = np.zeros((height, width, 3), dtype=np.uint8)

                u_n, v_n, z_n = _project_points(
                    points_world=newton_rollout[frame_idx],
                    w2c=w2c,
                    K=K,
                    y_sign=y_sign,
                    z_sign=z_sign,
                    z_eps=1e-6,
                )
                u_p, v_p, z_p = _project_points(
                    points_world=phystwin_rollout[frame_idx],
                    w2c=w2c,
                    K=K,
                    y_sign=y_sign,
                    z_sign=z_sign,
                    z_eps=1e-6,
                )

                if u_n.size > 0 and u_p.size > 0:
                    u = np.concatenate([u_n, u_p], axis=0)
                    v = np.concatenate([v_n, v_p], axis=0)
                    z = np.concatenate([z_n, z_p], axis=0)
                    colors_u8 = np.concatenate(
                        [
                            np.tile(newton_color[None, :], (u_n.shape[0], 1)),
                            np.tile(phystwin_color[None, :], (u_p.shape[0], 1)),
                        ],
                        axis=0,
                    )
                elif u_n.size > 0:
                    u, v, z = u_n, v_n, z_n
                    colors_u8 = np.tile(newton_color[None, :], (u_n.shape[0], 1))
                elif u_p.size > 0:
                    u, v, z = u_p, v_p, z_p
                    colors_u8 = np.tile(phystwin_color[None, :], (u_p.shape[0], 1))
                else:
                    u = np.zeros((0,), dtype=np.float64)
                    v = np.zeros((0,), dtype=np.float64)
                    z = np.zeros((0,), dtype=np.float64)
                    colors_u8 = np.zeros((0, 3), dtype=np.uint8)

                _draw_points(
                    image=image,
                    u=u,
                    v=v,
                    z=z,
                    colors_u8=colors_u8.astype(np.uint8, copy=False),
                    radius=int(args.point_radius),
                )

                Image.fromarray(image).save(panel_frames / f"{frame_idx}.png")

            _encode_mp4(frames_dir=panel_frames, fps=fps, out_mp4=panel_mp4)

        _compose_1x3_with_labels(
            panel_mp4s=panel_mp4s,
            camera_indices=camera_indices,
            label_text=label_text,
            out_mp4=out_mp4,
        )
    finally:
        if not args.keep_temp and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        json.dumps(
            {
                "case": case_name,
                "report": str(report_path),
                "rollout_npz": str(rollout_npz),
                "inference_pkl": str(inference_pkl),
                "frames_to_render": int(frames_to_render),
                "points_to_render": int(points_to_render),
                "camera_indices": camera_indices,
                "newton_color": args.newton_color,
                "phystwin_color": args.phystwin_color,
                "out_mp4": str(out_mp4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
