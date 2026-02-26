#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from path_defaults import resolve_overlay_base


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
    w2cs = np.asarray([np.linalg.inv(np.asarray(c2w, dtype=np.float64)) for c2w in c2ws], dtype=np.float64)
    if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected w2c shape from {calib_path}: {w2cs.shape}")
    return w2cs


def _load_final_data(case_dir: Path) -> dict:
    final_data_path = case_dir / "final_data.pkl"
    if not final_data_path.exists():
        raise FileNotFoundError(f"Missing final_data.pkl: {final_data_path}")
    with final_data_path.open("rb") as handle:
        final_data = pickle.load(handle)
    if not isinstance(final_data, dict):
        raise ValueError(f"Expected dict in final_data.pkl: {final_data_path}")
    if "object_colors" not in final_data:
        raise KeyError(f"final_data.pkl missing 'object_colors': {final_data_path}")
    if "controller_points" not in final_data:
        raise KeyError(f"final_data.pkl missing 'controller_points': {final_data_path}")
    return final_data


def _load_rollout_positions(rollout_npz: Path) -> np.ndarray:
    if not rollout_npz.exists():
        raise FileNotFoundError(f"Missing rollout npz: {rollout_npz}")
    with np.load(rollout_npz, allow_pickle=False) as data:
        if "particle_q_object" in data:
            positions = data["particle_q_object"]
        elif "particle_q_all" in data:
            positions = data["particle_q_all"]
        else:
            raise KeyError(
                f"{rollout_npz} missing 'particle_q_object' (and 'particle_q_all'). "
                f"Available keys: {sorted(data.files)}"
            )
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Unexpected rollout position shape in {rollout_npz}: {positions.shape}")
    return positions


def _resolve_overlay_path(
    overlay_base: Path, case_name: str, camera_idx: int, frame_idx: int
) -> Path:
    candidates = [
        overlay_base / case_name / "color" / str(camera_idx) / f"{frame_idx}.png",
        overlay_base / case_name / "color" / str(camera_idx) / f"{frame_idx:05d}.png",
        overlay_base / case_name / "color" / str(camera_idx) / f"{frame_idx:04d}.png",
        overlay_base / case_name / "color" / str(camera_idx) / f"{frame_idx:03d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing overlay frame. Tried:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def _project_points(
    points_world: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    y_sign: float,
    z_sign: float,
    z_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_world.astype(np.float64), ones], axis=1)  # [N,4]
    cam_h = (w2c @ points_h.T).T  # [N,4]
    x = cam_h[:, 0]
    y = cam_h[:, 1] * float(y_sign)
    z = cam_h[:, 2] * float(z_sign)
    valid = z > float(z_eps)
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty, np.zeros((0,), dtype=np.int64)
    valid_idx = np.flatnonzero(valid).astype(np.int64)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return u, v, z, valid_idx


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
            u, v, _z, _idx = _project_points(
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
    offsets: list[tuple[int, int]] = []
    if radius == 0:
        return [(0, 0)]
    r_sq = radius * radius
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

    order = np.argsort(z)[::-1]  # far -> near (larger z first)
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


def _resolve_default_tmp_dir(out_mp4: Path, rollout_stem: str, camera_idx: int) -> Path:
    return out_mp4.parent / f"._frames_{rollout_stem}_cam{camera_idx}"


def _ensure_clean_tmp_dir(tmp_dir: Path, user_provided: bool) -> None:
    if tmp_dir.exists():
        if user_provided:
            raise FileExistsError(
                f"Temporary directory already exists: {tmp_dir} (refusing to overwrite)"
            )
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def _encode_mp4(frames_dir: Path, fps: float, out_mp4: Path) -> None:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid fps={fps} for mp4 encoding.")
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
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed.\nCommand: {' '.join(cmd)}\nStderr:\n{stderr}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Newton rollout .npz into an MP4 video with PhysTwin-style "
            "camera parameters + forced RGB overlay."
        )
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--rollout-npz", type=Path, required=True)
    parser.add_argument(
        "--overlay-base",
        type=Path,
        default=None,
        help=(
            "Base directory containing <case_name>/color/<cam>/<frame>.png overlay frames. "
            "If unset, auto-resolve via PHYSTWIN_OVERLAY_BASE and common local paths."
        ),
    )
    parser.add_argument("--camera-idx", type=int, default=0)
    parser.add_argument(
        "--out-mp4",
        type=Path,
        default=None,
        help="Output mp4 path. Default: alongside rollout npz as <stem>_vis_cam<idx>.mp4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on rendered frames (default: full rollout length).",
    )
    parser.add_argument("--point-radius", type=int, default=1, help="Object point radius in pixels.")
    parser.add_argument(
        "--controller-radius", type=int, default=4, help="Controller point radius in pixels."
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate PNG frames instead of deleting them.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Optional directory to store intermediate PNG frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    rollout_npz = args.rollout_npz.resolve()
    overlay_base = resolve_overlay_base(args.overlay_base)
    camera_idx = int(args.camera_idx)

    metadata = _load_metadata(case_dir)
    case_name = case_dir.name

    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected intrinsics shape in metadata.json: {intrinsics.shape}")
    if camera_idx < 0 or camera_idx >= intrinsics.shape[0]:
        raise ValueError(
            f"--camera-idx {camera_idx} out of range for intrinsics count {intrinsics.shape[0]}"
        )
    K = intrinsics[camera_idx]

    wh = metadata["WH"]
    if not (isinstance(wh, list) and len(wh) == 2):
        raise ValueError(f"Unexpected WH in metadata.json: {wh!r}")
    width = int(wh[0])
    height = int(wh[1])

    fps = float(metadata.get("fps", 30.0))

    w2cs = _load_w2c(case_dir)
    if camera_idx < 0 or camera_idx >= w2cs.shape[0]:
        raise ValueError(
            f"--camera-idx {camera_idx} out of range for calibrate.pkl camera count {w2cs.shape[0]}"
        )
    w2c = w2cs[camera_idx]

    final_data = _load_final_data(case_dir)
    object_colors = np.asarray(final_data["object_colors"], dtype=np.float32)
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float32)
    if object_colors.ndim != 3 or object_colors.shape[-1] != 3:
        raise ValueError(f"Unexpected object_colors shape in final_data.pkl: {object_colors.shape}")
    if controller_points.ndim != 3 or controller_points.shape[-1] != 3:
        raise ValueError(
            f"Unexpected controller_points shape in final_data.pkl: {controller_points.shape}"
        )

    rollout = _load_rollout_positions(rollout_npz)
    rollout_frames = int(rollout.shape[0])

    max_frames = rollout_frames if args.max_frames is None else int(args.max_frames)
    if max_frames <= 0:
        raise ValueError(f"--max-frames must be > 0 (got {max_frames}).")
    frames_to_render = min(rollout_frames, max_frames)

    if object_colors.shape[0] < frames_to_render:
        raise ValueError(
            "final_data object_colors has fewer frames than rollout: "
            f"colors_frames={object_colors.shape[0]}, rollout_frames={frames_to_render}"
        )
    if controller_points.shape[0] < frames_to_render:
        raise ValueError(
            "final_data controller_points has fewer frames than rollout: "
            f"ctrl_frames={controller_points.shape[0]}, rollout_frames={frames_to_render}"
        )

    rollout_stem = rollout_npz.stem
    out_mp4 = (
        args.out_mp4.resolve()
        if args.out_mp4 is not None
        else rollout_npz.with_name(f"{rollout_stem}_vis_cam{camera_idx}.mp4")
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    user_tmp_dir = args.tmp_dir is not None
    tmp_dir = (
        args.tmp_dir.resolve()
        if args.tmp_dir is not None
        else _resolve_default_tmp_dir(out_mp4=out_mp4, rollout_stem=rollout_stem, camera_idx=camera_idx)
    )
    _ensure_clean_tmp_dir(tmp_dir, user_provided=user_tmp_dir)

    num_points = int(rollout.shape[1])
    base_color_count = int(object_colors.shape[1])
    padded_colors = np.full((num_points, 3), 0.3, dtype=np.float32)
    padded_colors[: min(num_points, base_color_count)] = object_colors[0, : min(num_points, base_color_count)]

    y_sign, z_sign = _choose_projection_convention(
        sample_points_world=rollout[0],
        w2c=w2c,
        K=K,
        width=width,
        height=height,
    )

    for frame_idx in range(frames_to_render):
        overlay_path = _resolve_overlay_path(
            overlay_base=overlay_base,
            case_name=case_name,
            camera_idx=camera_idx,
            frame_idx=frame_idx,
        )
        overlay = Image.open(overlay_path).convert("RGB")
        if overlay.size != (width, height):
            raise ValueError(
                f"Overlay frame size mismatch at {overlay_path}: got {overlay.size}, expected {(width, height)}."
            )
        image = np.asarray(overlay, dtype=np.uint8).copy()

        colors = padded_colors.copy()
        if base_color_count > 0:
            colors[: min(num_points, base_color_count)] = object_colors[frame_idx, : min(num_points, base_color_count)]
        colors_u8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)

        u, v, z, idx_visible = _project_points(
            points_world=rollout[frame_idx],
            w2c=w2c,
            K=K,
            y_sign=y_sign,
            z_sign=z_sign,
            z_eps=1e-6,
        )
        colors_u8_visible = colors_u8[idx_visible] if idx_visible.size > 0 else colors_u8[:0]
        _draw_points(
            image=image,
            u=u,
            v=v,
            z=z,
            colors_u8=colors_u8_visible,
            radius=int(args.point_radius),
        )

        ctrl = controller_points[frame_idx]
        if ctrl.size > 0:
            ctrl_u, ctrl_v, ctrl_z, _ctrl_idx = _project_points(
                points_world=ctrl,
                w2c=w2c,
                K=K,
                y_sign=y_sign,
                z_sign=z_sign,
                z_eps=1e-6,
            )
            ctrl_colors_u8 = np.tile(np.asarray([255, 0, 0], dtype=np.uint8), (ctrl_u.shape[0], 1))
            _draw_points(
                image=image,
                u=ctrl_u,
                v=ctrl_v,
                z=ctrl_z,
                colors_u8=ctrl_colors_u8,
                radius=int(args.controller_radius),
            )

        frame_out = tmp_dir / f"{frame_idx}.png"
        Image.fromarray(image).save(frame_out)

    encoded_ok = False
    try:
        _encode_mp4(frames_dir=tmp_dir, fps=fps, out_mp4=out_mp4)
        encoded_ok = True
    finally:
        # Keep frames when encoding fails for debugging, even if --keep-frames is not set.
        if encoded_ok and not args.keep_frames:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(json.dumps({"case": case_name, "rollout_npz": str(rollout_npz), "out_mp4": str(out_mp4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
