#!/usr/bin/env python3
"""Output and summary helpers for the cloth+bunny drop demo.

This module keeps result serialization and summary construction separate from
the main demo script so the entry point can stay focused on model setup,
simulation, and orchestration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from bridge_bootstrap import newton, newton_import_ir
from bridge_deformable_common import quat_to_rotmat
from .scene import _box_signed_distance, _effective_spring_scales, _mass_tag

__all__ = [
    "build_summary",
    "save_collision_force_rollout",
    "save_scene_npz",
    "save_summary_json",
]


def save_scene_npz(args: argparse.Namespace, sim_data: dict[str, Any], meta: dict[str, Any], n_obj: int) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_npz = args.out_dir / f"{args.prefix}_{_mass_tag(args.rigid_mass)}_scene.npz"
    np.savez_compressed(
        scene_npz,
        particle_q_all=sim_data["particle_q_all"],
        particle_q_object=sim_data["particle_q_object"],
        body_q=sim_data["body_q"],
        body_vel=sim_data["body_vel"],
        sim_dt=np.float32(sim_data["sim_dt"]),
        substeps=np.int32(sim_data["substeps"]),
        rigid_mesh_vertices_local=meta["mesh_verts_local"],
        rigid_mesh_indices=meta["mesh_tri_indices"],
        rigid_mesh_scale=np.float32(meta["mesh_scale"]),
        rigid_mesh_render_edges=meta["mesh_render_edges"],
        rigid_shape_kind=np.asarray(str(meta.get("rigid_shape", "bunny"))),
        rigid_box_half_extents=np.asarray(meta.get("box_half_extents", np.zeros(3, dtype=np.float32)), dtype=np.float32),
        rigid_mass=np.float32(args.rigid_mass),
        drop_height=np.float32(args.drop_height),
        num_object_points=np.int32(n_obj),
    )
    return scene_npz


def save_collision_force_rollout(
    args: argparse.Namespace,
    sim_data: dict[str, Any],
    scene_npz_path: Path | None = None,
) -> tuple[Path, Path] | tuple[None, None]:
    rollout = sim_data.get("collision_force_rollout")
    if not isinstance(rollout, dict):
        return None, None
    detector_dir = args.out_dir / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    npz_path = detector_dir / "collision_force_rollout_bundle.npz"
    summary_path = detector_dir / "summary.json"
    np.savez_compressed(
        npz_path,
        particle_q=np.asarray(rollout["particle_q"], dtype=np.float32),
        body_q=np.asarray(rollout["body_q"], dtype=np.float32),
        penalty_force=np.asarray(rollout["penalty_force"], dtype=np.float32),
        total_force=np.asarray(rollout["total_force"], dtype=np.float32),
        force_contact_mask=np.asarray(rollout["force_contact_mask"], dtype=np.bool_),
        sim_time_s=np.asarray(rollout["sim_time_s"], dtype=np.float32),
        frame_indices=np.asarray(rollout["frame_indices"], dtype=np.int32),
    )
    summary = {
        "npz_path": str(npz_path),
        "scene_npz_path": None if scene_npz_path is None else str(scene_npz_path),
        "rigid_shape": str(sim_data.get("rigid_shape", "")),
        "node_selection_mode": "all_force_contact_nodes",
        "first_force_contact_frame_index": rollout.get("first_force_contact_frame"),
        "force_definition_penalty": str(rollout.get("force_definition_penalty", "f_external_total")),
        "force_definition_total": str(
            rollout.get("force_definition_total", "f_internal_total + f_external_total + mass * gravity_vec")
        ),
        "sim_frame_count": int(np.asarray(rollout["frame_indices"], dtype=np.int32).shape[0]),
        "sim_frame_dt_s": float(sim_data["sim_dt"]) * float(sim_data["substeps"]),
        "force_contact_node_count_per_frame": [
            int(v)
            for v in np.sum(np.asarray(rollout["force_contact_mask"], dtype=np.bool_), axis=1, dtype=np.int32).tolist()
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, summary_path


def _first_negative(values: np.ndarray) -> int | None:
    idx = np.flatnonzero(values < 0.0)
    return int(idx[0]) if idx.size else None


def build_summary(
    model: newton.Model,
    args: argparse.Namespace,
    ir_obj: dict[str, Any],
    sim_data: dict[str, Any],
    meta: dict[str, Any],
    n_obj: int,
    out_mp4: Path | None,
    particle_contacts_enabled: bool,
    disable_particle_contact_kernel: bool,
) -> dict[str, Any]:
    cloth_q = np.asarray(sim_data["particle_q_object"], dtype=np.float32)
    body_q = np.asarray(sim_data["body_q"], dtype=np.float32)
    if body_q.ndim == 3 and body_q.shape[1] > 0 and bool(meta.get("has_bunny", False)):
        body_speed = np.linalg.norm(sim_data["body_vel"][:, 0, :], axis=1)
        body_z = body_q[:, 0, 2]
        bunny_top_offset = float(meta["bunny_top_z"]) - float(body_z[0])
        dynamic_bunny_top = body_z + bunny_top_offset
    else:
        body_speed = np.zeros((cloth_q.shape[0],), dtype=np.float32)
        dynamic_bunny_top = np.zeros((cloth_q.shape[0],), dtype=np.float32)
    cloth_z_min = cloth_q[:, :, 2].min(axis=1)
    cloth_z_mean = cloth_q[:, :, 2].mean(axis=1)
    cloth_z_max = cloth_q[:, :, 2].max(axis=1)
    clearance_min = cloth_z_min - dynamic_bunny_top
    clearance_mean = cloth_z_mean - dynamic_bunny_top
    clearance_max = cloth_z_max - dynamic_bunny_top

    sim_frame_dt = float(sim_data["sim_dt"]) * float(sim_data["substeps"])
    sim_duration = max((sim_data["particle_q_all"].shape[0] - 1) * sim_frame_dt, 0.0)
    video_duration = sim_duration * float(args.slowdown)
    rendered_frame_count = max(1, int(round(video_duration * float(args.render_fps))))
    if "rendered_frame_count" in sim_data:
        rendered_frame_count = int(sim_data["rendered_frame_count"])
    if "video_duration_target_sec" in sim_data:
        video_duration = float(sim_data["video_duration_target_sec"])
    object_mass = float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].mean())
    spring_ke_scale, spring_kd_scale = _effective_spring_scales(ir_obj, args)
    max_penetration_depth = None
    final_penetration_p99 = None
    if str(meta.get("rigid_shape")) == "box" and body_q.ndim == 3 and body_q.shape[1] > 0:
        hx, hy, hz = [float(v) for v in np.asarray(meta["box_half_extents"]).ravel()]
        radii = model.particle_radius.numpy().astype(np.float32)[:n_obj]
        penetration = np.zeros((cloth_q.shape[0], cloth_q.shape[1]), dtype=np.float32)
        for frame_idx in range(cloth_q.shape[0]):
            pos = body_q[frame_idx, 0, :3].astype(np.float32)
            quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
            rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
            local = (cloth_q[frame_idx] - pos[None, :]) @ rot
            sdf = _box_signed_distance(local.astype(np.float32, copy=False), hx, hy, hz)
            penetration[frame_idx] = np.maximum(radii - sdf, 0.0)
        max_penetration_depth = float(np.max(penetration))
        final_penetration_p99 = float(np.quantile(penetration[-1], 0.99))
    bunny_mesh_max_penetration = None
    bunny_mesh_final_p99_penetration = None
    if (
        not bool(args.skip_render)
        and str(meta.get("rigid_shape")) == "bunny"
        and body_q.ndim == 3
        and body_q.shape[1] > 0
    ):
        faces = np.asarray(meta.get("mesh_tri_indices", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        verts_local = np.asarray(meta.get("mesh_verts_local", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if faces.size and verts_local.size:
            radius = model.particle_radius.numpy().astype(np.float32)[:n_obj]
            scale = float(meta["mesh_scale"])
            sampled_max = 0.0
            frame_ids = list(range(0, cloth_q.shape[0], 15))
            if frame_ids[-1] != cloth_q.shape[0] - 1:
                frame_ids.append(cloth_q.shape[0] - 1)
            final_penetration = None
            for frame_idx in frame_ids:
                pos = body_q[frame_idx, 0, :3].astype(np.float32)
                quat_xyzw = body_q[frame_idx, 0, 3:7].astype(np.float32)
                rot = quat_to_rotmat([float(v) for v in quat_xyzw]).astype(np.float32)
                verts = (verts_local * scale) @ rot.T + pos[None, :]
                tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                _, dist, _ = trimesh.proximity.closest_point(tri, cloth_q[frame_idx].astype(np.float64))
                penetration = np.maximum(radius - dist.astype(np.float32, copy=False), 0.0)
                sampled_max = max(sampled_max, float(np.max(penetration)))
                if frame_idx == cloth_q.shape[0] - 1:
                    final_penetration = penetration
            bunny_mesh_max_penetration = float(sampled_max)
            if final_penetration is not None:
                bunny_mesh_final_p99_penetration = float(np.quantile(final_penetration, 0.99))
    summary = {
        "experiment": "cloth_bunny_drop_object_only",
        "self_collision_case": "off",
        "ir_path": str(args.ir.resolve()),
        "object_only": True,
        "drop_height_m": float(args.drop_height),
        "initial_velocity_z_mps": float(args.initial_velocity_z),
        "cloth_shift_x_m": float(args.cloth_shift_x),
        "cloth_shift_y_m": float(args.cloth_shift_y),
        "object_mass_per_particle": object_mass,
        "auto_set_weight_target_kg": (
            None if args.auto_set_weight is None else float(args.auto_set_weight)
        ),
        "weight_scale": float(ir_obj.get("weight_scale", 1.0)),
        "mass_spring_scale": (
            None if args.mass_spring_scale is None else float(args.mass_spring_scale)
        ),
        "n_object_particles": int(n_obj),
        "total_object_mass": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum()),
        "rigid_mass": float(args.rigid_mass),
        "mass_ratio": float(np.asarray(ir_obj["mass"], dtype=np.float32)[:n_obj].sum() / max(args.rigid_mass, 1.0e-8)),
        "rigid_geometry": str(meta.get("rigid_shape", "bunny")),
        "has_bunny": bool(meta.get("has_bunny", False)),
        "has_ground_plane": bool(meta.get("has_ground_plane", False)),
        "bunny_scale": float(args.bunny_scale),
        "bunny_quat_xyzw": [float(v) for v in meta["bunny_quat_xyzw"]],
        "bunny_top_z": float(meta["bunny_top_z"]),
        "reverse_z": bool(newton_import_ir.ir_bool(ir_obj, "reverse_z", default=False)),
        "sim_coord_system": "newton_z_up_gravity_negative_z",
        "contact_collision_dist_used": float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]),
        "contact_dist_scale": float(args.contact_dist_scale),
        "frames": int(args.frames),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_frame_dt),
        "sim_duration_sec": float(sim_duration),
        "wall_time_sec": float(sim_data["wall_time"]),
        "slowdown_factor": float(args.slowdown),
        "render_fps": float(args.render_fps),
        "rendered_frame_count": int(rendered_frame_count),
        "video_duration_target_sec": float(video_duration),
        "first_rigid_contact_frame": (
            None
            if int(sim_data.get("first_rigid_contact_frame", -1)) < 0
            else int(sim_data["first_rigid_contact_frame"])
        ),
        "render_end_frame": int(sim_data.get("render_end_frame", cloth_q.shape[0] - 1)),
        "post_contact_video_seconds": float(args.post_contact_video_seconds),
        "body_speed_initial": float(body_speed[0]),
        "body_speed_final": float(body_speed[-1]),
        "body_speed_max": float(np.max(body_speed)),
        "dynamic_bunny_top_final_z": float(dynamic_bunny_top[-1]),
        "first_min_below_top_frame": _first_negative(clearance_min),
        "first_mean_below_top_frame": _first_negative(clearance_mean),
        "first_all_below_top_frame": _first_negative(clearance_max),
        "min_clearance_min_to_bunny_top": float(np.min(clearance_min)),
        "min_clearance_mean_to_bunny_top": float(np.min(clearance_mean)),
        "min_clearance_max_to_bunny_top": float(np.min(clearance_max)),
        "final_clearance_min_to_bunny_top": float(clearance_min[-1]),
        "final_clearance_mean_to_bunny_top": float(clearance_mean[-1]),
        "final_clearance_max_to_bunny_top": float(clearance_max[-1]),
        "apply_drag": bool(args.apply_drag),
        "drag_damping_scale": float(args.drag_damping_scale),
        "drag_ignore_gravity_axis": bool(args.drag_ignore_gravity_axis),
        "spring_ke_scale": float(spring_ke_scale),
        "spring_kd_scale": float(spring_kd_scale),
        "shape_contact_scale": (
            None if args.shape_contact_scale is None else float(args.shape_contact_scale)
        ),
        "shape_contact_damping_multiplier": float(args.shape_contact_damping_multiplier),
        "use_decoupled_shape_materials": bool(
            sim_data.get("use_decoupled_shape_materials", False)
        ),
        "soft_contact_ke_final": float(model.soft_contact_ke),
        "soft_contact_kd_final": float(model.soft_contact_kd),
        "soft_contact_kf_final": float(model.soft_contact_kf),
        "max_penetration_depth_rigid_m": max_penetration_depth,
        "final_penetration_p99_rigid_m": final_penetration_p99,
        "max_penetration_depth_bunny_mesh_m": bunny_mesh_max_penetration,
        "final_penetration_p99_bunny_mesh_m": bunny_mesh_final_p99_penetration,
        "particle_contact_ke_final": float(model.particle_ke),
        "particle_contact_kd_final": float(model.particle_kd),
        "particle_contact_kf_final": float(model.particle_kf),
        "particle_contacts_enabled": bool(particle_contacts_enabled),
        "disable_particle_contact_kernel": bool(disable_particle_contact_kernel),
        "camera_pos": [float(v) for v in args.camera_pos],
        "camera_pitch": float(args.camera_pitch),
        "camera_yaw": float(args.camera_yaw),
        "camera_fov": float(args.camera_fov),
        "camera_mode": "manual",
        "render_video": str(out_mp4) if out_mp4 is not None else None,
    }
    force_diagnostic = sim_data.get("force_diagnostic")
    if isinstance(force_diagnostic, dict):
        summary.update(
            {
                "force_diagnostic_enabled": bool(force_diagnostic.get("enabled", False)),
                "force_snapshot_frame_mode": str(force_diagnostic.get("snapshot_frame_mode", "trigger")),
                "force_snapshot_substeps_after_trigger": int(
                    force_diagnostic.get("snapshot_substeps_after_trigger", 0)
                ),
                "force_diagnostic_trigger_substep_global": int(
                    force_diagnostic.get("trigger_substep_global", -1)
                ),
                "force_diagnostic_trigger_frame_index": int(
                    force_diagnostic.get("trigger_frame_index", -1)
                ),
                "force_diagnostic_trigger_substep_index_in_frame": int(
                    force_diagnostic.get("trigger_substep_index_in_frame", -1)
                ),
                "force_diagnostic_contact_node_count": int(force_diagnostic.get("contact_node_count", 0)),
                "force_diagnostic_topk_particle_count": int(force_diagnostic.get("topk_particle_count", 0)),
                "force_diagnostic_topk_mode": str(force_diagnostic.get("topk_mode", "")),
                "force_diagnostic_normal_source": str(force_diagnostic.get("normal_source", "")),
                "force_diagnostic_used_manual_force_path": bool(
                    force_diagnostic.get("used_manual_force_path", False)
                ),
                "force_diagnostic_dominant_issue_guess": str(
                    force_diagnostic.get("dominant_issue_guess", "undetermined")
                ),
                "force_diagnostic_force_eps": float(force_diagnostic.get("force_eps", 0.0)),
                "force_diagnostic_npz": str(force_diagnostic.get("npz_path", "")),
                "force_diagnostic_summary_json": str(force_diagnostic.get("summary_json_path", "")),
                "force_diagnostic_snapshot_png": str(force_diagnostic.get("snapshot_png_path", "")),
                "force_diagnostic_snapshot_video": str(force_diagnostic.get("snapshot_video_path", "")),
                "force_diagnostic_window_video": str(force_diagnostic.get("window_video_path", "")),
                "force_diagnostic_window_keyframe_count": int(len(force_diagnostic.get("window_sequence", []))),
                "rollout_truncated_for_force_diagnostic": bool(
                    force_diagnostic.get("rollout_truncated", False)
                ),
            }
        )
        parity_check = force_diagnostic.get("parity_check")
        if isinstance(parity_check, dict):
            summary.update(
                {
                    "force_diagnostic_parity_check_enabled": True,
                    "force_diagnostic_parity_consistent": bool(
                        parity_check.get("parity_consistent", False)
                    ),
                    "force_diagnostic_parity_baseline_trigger_found": bool(
                        parity_check.get("baseline_trigger_found", False)
                    ),
                    "force_diagnostic_parity_delta_trigger_substep": int(
                        parity_check.get("delta_trigger_substep", 0)
                    ),
                    "force_diagnostic_parity_delta_cloth_com_mm": float(
                        parity_check.get("delta_cloth_com_mm", 0.0)
                    ),
                    "force_diagnostic_parity_delta_rigid_pos_mm": float(
                        parity_check.get("delta_rigid_pos_mm", 0.0)
                    ),
                    "force_diagnostic_parity_delta_rigid_rot_deg": float(
                        parity_check.get("delta_rigid_rot_deg", 0.0)
                    ),
                    "force_diagnostic_parity_delta_max_penetration_mm": float(
                        parity_check.get("delta_max_penetration_mm", 0.0)
                    ),
                    "force_diagnostic_parity_contact_topk_jaccard": float(
                        parity_check.get("contact_topk_jaccard", 0.0)
                    ),
                }
            )
    collision_force_rollout = sim_data.get("collision_force_rollout")
    if isinstance(collision_force_rollout, dict):
        summary.update(
            {
                "collision_force_rollout_enabled": True,
                "collision_force_rollout_first_force_contact_frame": collision_force_rollout.get("first_force_contact_frame"),
                "collision_force_rollout_force_definition_penalty": str(
                    collision_force_rollout.get("force_definition_penalty", "f_external_total")
                ),
                "collision_force_rollout_force_definition_total": str(
                    collision_force_rollout.get(
                        "force_definition_total",
                        "f_internal_total + f_external_total + mass * gravity_vec",
                    )
                ),
            }
        )
    return summary


def save_summary_json(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.prefix}_{_mass_tag(args.rigid_mass)}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
