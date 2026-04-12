#!/usr/bin/env python3
"""Force-visualization helpers for cloth+bunny demos.

This module keeps camera, overlay, and frame-selection helpers separate from
the main simulation/orchestration script so the demo entry point can focus on
scene setup and rollout logic.
"""
from __future__ import annotations

import argparse

import numpy as np

from bridge_shared import compute_visual_particle_radii


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    peak = float(np.max(values))
    if peak <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / peak).astype(np.float32, copy=False)


def _select_force_topk(
    external_norm: np.ndarray,
    *,
    internal_force_normal: np.ndarray,
    penetration_depth_signed: np.ndarray,
    geom_contact_mask: np.ndarray,
    topk: int,
    eps: float,
    mode: str,
) -> np.ndarray:
    geom_candidates = np.flatnonzero(np.asarray(geom_contact_mask, dtype=bool))
    force_candidates = np.flatnonzero(np.asarray(external_norm, dtype=np.float32) > eps)
    if geom_candidates.size:
        candidates = geom_candidates
    elif force_candidates.size:
        candidates = force_candidates
    else:
        candidates = np.arange(external_norm.shape[0], dtype=np.int32)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int32)

    external_score = _normalized(np.asarray(external_norm, dtype=np.float32)[candidates])
    internal_score = _normalized(
        np.abs(np.asarray(internal_force_normal, dtype=np.float32)[candidates])
    )
    geom_score = _normalized(
        np.asarray(penetration_depth_signed, dtype=np.float32)[candidates]
    )

    if mode == "external":
        score = external_score
    elif mode == "geom":
        score = geom_score
    else:
        score = (
            0.45 * external_score + 0.35 * internal_score + 0.20 * geom_score
        ).astype(np.float32, copy=False)

    order = np.argsort(score)[::-1]
    limit = min(max(1, int(topk)), candidates.size)
    return candidates[order[:limit]].astype(np.int32, copy=False)


def _thin_force_probe_indices(
    candidate_indices: np.ndarray,
    closest_points_world: np.ndarray,
    *,
    max_count: int,
    min_dist: float,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int32)
    limit = min(max(1, int(max_count)), candidate_indices.size)
    if limit >= candidate_indices.size or float(min_dist) <= 0.0:
        return candidate_indices[:limit].astype(np.int32, copy=False)

    closest_points_world = np.asarray(closest_points_world, dtype=np.float32)
    selected: list[int] = []
    min_sq = float(min_dist) * float(min_dist)
    for idx in candidate_indices.tolist():
        point = closest_points_world[int(idx)]
        keep = True
        for prev in selected:
            delta = point - closest_points_world[int(prev)]
            if float(np.dot(delta, delta)) < min_sq:
                keep = False
                break
        if keep:
            selected.append(int(idx))
            if len(selected) >= limit:
                break
    if not selected:
        return candidate_indices[:limit].astype(np.int32, copy=False)
    if len(selected) < limit:
        used = set(selected)
        for idx in candidate_indices.tolist():
            if int(idx) in used:
                continue
            selected.append(int(idx))
            if len(selected) >= limit:
                break
    return np.asarray(selected[:limit], dtype=np.int32)


def _camera_basis_from_pitch_yaw(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))
    forward = np.asarray(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ],
        dtype=np.float32,
    )
    forward /= max(np.linalg.norm(forward), 1.0e-8)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up).astype(np.float32, copy=False)
    if np.linalg.norm(right) <= 1.0e-8:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right /= max(np.linalg.norm(right), 1.0e-8)
    up = np.cross(right, forward).astype(np.float32, copy=False)
    up /= max(np.linalg.norm(up), 1.0e-8)
    return forward, right, up


def _fit_camera_to_points(
    points_world: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    aspect: float,
    min_distance: float = 0.22,
    pad: float = 1.14,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        origin = np.zeros((3,), dtype=np.float32)
        return origin, origin
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    center = (0.5 * (bbox_min + bbox_max)).astype(np.float32, copy=False)
    forward, right, up = _camera_basis_from_pitch_yaw(pitch_deg, yaw_deg)
    rel = points - center[None, :]
    x = np.abs(rel @ right)
    y = np.abs(rel @ up)
    z_off = rel @ forward
    tan_half_y = max(np.tan(np.deg2rad(float(fov_deg)) * 0.5), 1.0e-6)
    tan_half_x = tan_half_y * max(float(aspect), 1.0e-6)
    req_x = np.max(x / tan_half_x - z_off) if x.size else 0.0
    req_y = np.max(y / tan_half_y - z_off) if y.size else 0.0
    distance = max(float(min_distance), float(pad) * max(float(req_x), float(req_y), 0.0))
    cam_pos = (center - forward * distance).astype(np.float32, copy=False)
    return cam_pos, center.astype(np.float32, copy=False)


def _resolve_force_camera(
    args: argparse.Namespace,
    *,
    focus_world: np.ndarray | None,
    particle_world: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float]:
    pitch = float(args.force_camera_pitch if args.force_camera_pitch is not None else args.camera_pitch)
    yaw = float(args.force_camera_yaw if args.force_camera_yaw is not None else args.camera_yaw)
    fov = float(args.force_camera_fov if args.force_camera_fov is not None else args.camera_fov)
    if args.force_camera_pos is not None:
        cam_pos = np.asarray(args.force_camera_pos, dtype=np.float32).reshape(3)
        return cam_pos, pitch, yaw, fov
    points: list[np.ndarray] = []
    if particle_world is not None:
        pts = np.asarray(particle_world, dtype=np.float32).reshape(-1, 3)
        if pts.size:
            points.append(pts)
    if focus_world is not None:
        points.append(np.asarray(focus_world, dtype=np.float32).reshape(1, 3))
    if points:
        scene_points = np.concatenate(points, axis=0)
        cam_pos, _ = _fit_camera_to_points(
            scene_points,
            yaw_deg=yaw,
            pitch_deg=pitch,
            fov_deg=fov,
            aspect=float(args.screen_width) / max(float(args.screen_height), 1.0),
        )
        return cam_pos, pitch, yaw, fov
    return np.asarray(args.camera_pos, dtype=np.float32).reshape(3), pitch, yaw, fov


def _resolve_main_camera(args: argparse.Namespace) -> tuple[np.ndarray, float, float, float]:
    cam_pos = np.asarray(args.camera_pos, dtype=np.float32).reshape(3)
    return cam_pos, float(args.camera_pitch), float(args.camera_yaw), float(args.camera_fov)


def _continuous_output_frame_indices(
    *,
    n_sim_frames: int,
    render_end_frame: int,
    sim_frame_dt: float,
    fps_out: float,
    slowdown: float,
) -> np.ndarray:
    usable = max(1, min(int(render_end_frame) + 1, int(n_sim_frames)))
    if usable <= 1:
        return np.zeros((1,), dtype=np.int32)
    if fps_out <= 0.0 or sim_frame_dt <= 0.0:
        return np.arange(usable, dtype=np.int32)

    sim_end_time = float(usable - 1) * float(sim_frame_dt)
    out_duration = max(sim_end_time * max(float(slowdown), 1.0), sim_end_time)
    out_count = max(usable, int(round(out_duration * float(fps_out))) + 1)
    out_times = np.arange(out_count, dtype=np.float32) / max(float(fps_out), 1.0e-8)
    sim_times = out_times / max(float(slowdown), 1.0e-8)
    indices = np.rint(sim_times / max(float(sim_frame_dt), 1.0e-8)).astype(np.int32, copy=False)
    indices = np.clip(indices, 0, usable - 1)
    return indices


def _compute_force_visual_particle_radii(
    physical_radii: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return compute_visual_particle_radii(
        physical_radii,
        radius_scale=min(float(args.particle_radius_vis_scale), 1.35),
        radius_cap=min(float(args.particle_radius_vis_min), 0.0035),
    )


def _project_world_to_screen(
    point_world: np.ndarray,
    *,
    cam_pos: np.ndarray,
    pitch_deg: float,
    yaw_deg: float,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    forward, right, up = _camera_basis_from_pitch_yaw(pitch_deg, yaw_deg)
    v = np.asarray(point_world, dtype=np.float32) - np.asarray(cam_pos, dtype=np.float32)
    z = float(np.dot(v, forward))
    if z <= 1.0e-6:
        return None
    x = float(np.dot(v, right))
    y = float(np.dot(v, up))
    aspect = float(width) / max(float(height), 1.0)
    tan_half = np.tan(np.deg2rad(float(fov_deg)) * 0.5)
    if tan_half <= 1.0e-8:
        return None
    ndc_x = x / (z * tan_half * aspect)
    ndc_y = y / (z * tan_half)
    px = (ndc_x * 0.5 + 0.5) * float(width)
    py = (0.5 - ndc_y * 0.5) * float(height)
    return float(px), float(py)


def _overlay_zoom_panel_rgb(
    frame: np.ndarray,
    *,
    center_px: tuple[float, float] | None,
    label: str = "contact zoom",
    panel_frac: float = 0.32,
    zoom: float = 2.6,
    margin: int = 18,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame
    if center_px is None:
        return frame
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame.astype(np.uint8, copy=False)
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    if np.issubdtype(frame.dtype, np.floating):
        if float(np.nanmax(frame)) <= 1.0 + 1.0e-6:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
    frame = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
    height, width = frame.shape[:2]
    panel_w = max(160, int(width * panel_frac))
    panel_h = max(120, int(height * panel_frac))
    crop_w = max(40, int(panel_w / max(zoom, 1.0)))
    crop_h = max(40, int(panel_h / max(zoom, 1.0)))
    cx = int(round(center_px[0]))
    cy = int(round(center_px[1]))
    x0 = int(np.clip(cx - crop_w // 2, 0, max(0, width - crop_w)))
    y0 = int(np.clip(cy - crop_h // 2, 0, max(0, height - crop_h)))
    x1 = min(width, x0 + crop_w)
    y1 = min(height, y0 + crop_h)
    if x1 <= x0 or y1 <= y0:
        return frame

    img = Image.fromarray(frame, mode="RGB")
    crop = img.crop((x0, y0, x1, y1)).resize((panel_w, panel_h), resample=Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img, mode="RGBA")

    draw.rectangle((x0, y0, x1, y1), outline=(255, 215, 0, 255), width=3)

    panel_x1 = width - margin
    panel_y0 = max(margin + 96, (height - panel_h) // 2)
    panel_x0 = panel_x1 - panel_w
    panel_y1 = panel_y0 + panel_h
    draw.rectangle((panel_x0 - 8, panel_y0 - 30, panel_x1 + 8, panel_y1 + 8), fill=(0, 0, 0, 108))
    img.paste(crop, (panel_x0, panel_y0))
    draw.rectangle((panel_x0, panel_y0, panel_x1, panel_y1), outline=(255, 215, 0, 255), width=4)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((panel_x0 + 8, panel_y0 - 24), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def _draw_arrow_2d(draw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 4) -> None:
    import math

    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    head_len = max(10.0, width * 2.2)
    head_w = max(5.0, width * 1.2)
    p1 = (end[0] - ux * head_len + px * head_w, end[1] - uy * head_len + py * head_w)
    p2 = (end[0] - ux * head_len - px * head_w, end[1] - uy * head_len - py * head_w)
    draw.polygon([end, p1, p2], fill=color)


def _overlay_force_glyphs_rgb(
    frame: np.ndarray,
    *,
    cam_pos: np.ndarray,
    pitch_deg: float,
    yaw_deg: float,
    fov_deg: float,
    topk_q: np.ndarray,
    topk_closest: np.ndarray,
    topk_anchor: np.ndarray,
    topk_normals: np.ndarray,
    closest_offset_world: np.ndarray,
    external_force_normal_vec: np.ndarray,
    internal_force_normal_vec: np.ndarray,
    acceleration_normal_vec: np.ndarray,
    force_arrow_scale: float,
    normal_display_len: float,
    gap_display_cap: float,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    width = int(frame.shape[1])
    height = int(frame.shape[0])

    def proj(world: np.ndarray) -> tuple[float, float] | None:
        return _project_world_to_screen(
            np.asarray(world, dtype=np.float32),
            cam_pos=np.asarray(cam_pos, dtype=np.float32),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            fov_deg=float(fov_deg),
            width=width,
            height=height,
        )

    for particle_pt, closest_pt in zip(topk_q, topk_closest):
        p_px = proj(particle_pt)
        c_px = proj(closest_pt)
        if p_px is not None:
            draw.ellipse((p_px[0] - 7, p_px[1] - 7, p_px[0] + 7, p_px[1] + 7), fill=(255, 219, 46, 235))
        if c_px is not None:
            draw.ellipse((c_px[0] - 5, c_px[1] - 5, c_px[0] + 5, c_px[1] + 5), fill=(140, 240, 250, 235))

    gap_vectors = np.asarray(closest_offset_world, dtype=np.float32)
    if gap_vectors.size:
        gap_norm = np.linalg.norm(gap_vectors, axis=1)
        gap_scale = np.minimum(1.0, float(gap_display_cap) / np.maximum(gap_norm, 1.0e-8))
        gap_vectors = (gap_vectors * gap_scale[:, None]).astype(np.float32, copy=False)

    specs = [
        (topk_closest, topk_closest + topk_normals * float(normal_display_len), (255, 255, 255), 4),
        (topk_closest, topk_closest + gap_vectors, (179, 179, 179), 4),
        (topk_anchor, None, (242, 64, 64), 4),
        (topk_anchor, None, (158, 82, 235), 4),
        (topk_anchor, None, (38, 217, 64), 4),
    ]
    force_vecs = [
        external_force_normal_vec,
        internal_force_normal_vec,
        acceleration_normal_vec,
    ]
    scaled_force_ends: list[np.ndarray] = []
    for vectors in force_vecs:
        norms = np.linalg.norm(vectors, axis=1)
        peak = float(np.max(norms)) if norms.size else 0.0
        if peak <= 1.0e-12:
            scaled_force_ends.append(topk_anchor.copy())
        else:
            scaled = (vectors / peak * float(force_arrow_scale)).astype(np.float32, copy=False)
            scaled_force_ends.append((topk_anchor + scaled).astype(np.float32, copy=False))
    specs[2] = (topk_anchor, scaled_force_ends[0], (242, 64, 64), 5)
    specs[3] = (topk_anchor, scaled_force_ends[1], (158, 82, 235), 5)
    specs[4] = (topk_anchor, scaled_force_ends[2], (38, 217, 64), 5)

    for starts, ends, color, line_w in specs:
        for start_world, end_world in zip(starts, ends):
            s_px = proj(start_world)
            e_px = proj(end_world)
            if s_px is None or e_px is None:
                continue
            _draw_arrow_2d(draw, s_px, e_px, color, width=line_w)

    return np.asarray(img, dtype=np.uint8)


def _stage_label_for_frame(
    frame_idx: int,
    *,
    first_contact_frame: int | None,
    max_penetration_frame: int | None,
    rebound_frame: int | None,
) -> str:
    if first_contact_frame is None:
        return "Pre-contact"
    if frame_idx < first_contact_frame:
        return "Pre-contact"
    if frame_idx == first_contact_frame:
        return "First contact"
    if max_penetration_frame is not None and frame_idx < max_penetration_frame:
        return "Penetration growth"
    if max_penetration_frame is not None and frame_idx == max_penetration_frame:
        return "Max penetration"
    if rebound_frame is not None and frame_idx <= rebound_frame:
        return "Rebound / settle"
    return "Post-contact settle"


def _resolve_force_active_interval(
    *,
    sim_data: dict[str, object],
    n_frames: int,
    render_end_frame: int,
    first_contact_frame: int | None,
    rebound_frame: int | None,
    pre_frames: int = 3,
) -> tuple[int, int]:
    last_frame = max(0, min(int(render_end_frame), int(n_frames) - 1))
    if first_contact_frame is None:
        return 0, last_frame
    start = max(0, int(first_contact_frame) - max(1, int(pre_frames)))
    if rebound_frame is None:
        end = last_frame
    else:
        end = min(last_frame, max(int(first_contact_frame), int(rebound_frame)))
    if end < start:
        end = start
    return int(start), int(end)


def _build_process_story_keyframes(
    *,
    n_frames: int,
    first_contact_frame: int | None,
    max_penetration_frame: int | None,
    rebound_frame: int | None,
) -> list[tuple[int, str]]:
    last_idx = max(0, int(n_frames) - 1)
    if first_contact_frame is None:
        return [(0, "Pre-contact"), (last_idx, "No contact reached")]

    pre_idx = max(0, int(first_contact_frame) - 1)
    first_idx = int(first_contact_frame)
    if max_penetration_frame is not None and int(max_penetration_frame) > first_idx + 1:
        growth_idx = int(round(0.5 * (first_idx + int(max_penetration_frame))))
    else:
        growth_idx = min(last_idx, first_idx + 1)
    max_idx = int(max_penetration_frame) if max_penetration_frame is not None else growth_idx
    rebound_idx = int(rebound_frame) if rebound_frame is not None else last_idx
    settle_idx = last_idx

    ordered = [
        (pre_idx, "Pre-contact"),
        (first_idx, "First contact"),
        (growth_idx, "Penetration growth"),
        (max_idx, "Max penetration"),
        (rebound_idx, "Rebound / settle"),
        (settle_idx, "Final settle"),
    ]

    keyframes: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, label in ordered:
        idx = int(np.clip(idx, 0, last_idx))
        if idx in seen:
            continue
        seen.add(idx)
        keyframes.append((idx, label))
    return keyframes


def _force_active_interval_from_render_indices(
    *,
    sim_data: dict[str, object],
    render_indices: np.ndarray,
) -> dict[str, object]:
    render_indices = np.asarray(render_indices, dtype=np.int32).reshape(-1)
    if render_indices.size == 0:
        return {
            "first_contact_frame": None,
            "max_penetration_frame": None,
            "rebound_frame": None,
            "active_display_start": None,
            "active_display_end": None,
            "active_render_frames": np.zeros((0,), dtype=np.int32),
        }

    first_contact_frame = int(sim_data.get("first_rigid_contact_frame", -1))
    max_penetration_frame = int(sim_data.get("max_rigid_penetration_frame", -1))
    rebound_frame = int(sim_data.get("rigid_rebound_frame", -1))
    if first_contact_frame < 0:
        first_contact_frame = None
    if max_penetration_frame < 0:
        max_penetration_frame = None
    if rebound_frame < 0:
        rebound_frame = None
    if first_contact_frame is None:
        return {
            "first_contact_frame": None,
            "max_penetration_frame": max_penetration_frame,
            "rebound_frame": rebound_frame,
            "active_display_start": None,
            "active_display_end": None,
            "active_render_frames": np.zeros((0,), dtype=np.int32),
        }

    start_candidates = np.flatnonzero(render_indices >= int(first_contact_frame))
    if start_candidates.size:
        active_display_start = max(0, int(start_candidates[0]) - 3)
    else:
        active_display_start = 0
    active_end_ref = rebound_frame
    if active_end_ref is None:
        active_end_ref = max_penetration_frame
    if active_end_ref is None:
        active_end_ref = int(render_indices[-1])
    end_candidates = np.flatnonzero(render_indices >= int(active_end_ref))
    active_display_end = int(end_candidates[0]) if end_candidates.size else int(render_indices.shape[0] - 1)
    active_display_end = max(active_display_start, active_display_end)
    active_render_frames = np.unique(render_indices[active_display_start : active_display_end + 1]).astype(np.int32, copy=False)
    return {
        "first_contact_frame": first_contact_frame,
        "max_penetration_frame": max_penetration_frame,
        "rebound_frame": rebound_frame,
        "active_display_start": int(active_display_start),
        "active_display_end": int(active_display_end),
        "active_render_frames": active_render_frames,
    }


def _quat_angle_deg(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> float:
    q0 = np.asarray(q0_xyzw, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q1_xyzw, dtype=np.float64).reshape(-1)
    if q0.size != 4 or q1.size != 4:
        return 0.0
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 <= 1.0e-12 or n1 <= 1.0e-12:
        return 0.0
    dot = float(np.clip(np.abs(np.dot(q0 / n0, q1 / n1)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))
