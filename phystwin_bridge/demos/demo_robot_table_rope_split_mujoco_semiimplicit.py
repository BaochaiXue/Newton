#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from PIL import Image, ImageDraw
from pxr import Usd

from bridge_bootstrap import newton, newton_import_ir
from bridge_deformable_common import (
    _copy_object_only_ir,
    _effective_spring_scales,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    load_ir,
)
from bridge_shared import apply_viewer_shape_colors
from newton._src.viewer.viewer_gl import ViewerGL
from newton.geometry import HydroelasticSDF

import newton.ik as ik
import newton.usd
import newton.utils


DEFAULT_IR = Path(__file__).resolve().parents[1] / "ir" / "rope_double_hand" / "phystwin_ir_v2_bf_strict.npz"


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _ffmpeg_encode_mp4(width: int, height: int, fps: int, output_path: Path) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _make_gif(mp4_path: Path, gif_path: Path, fps: int = 15, width: int = 720) -> None:
    palette = gif_path.with_suffix(".palette.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(gif_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)


def _save_contact_sheet(frames: list[np.ndarray], output_path: Path, cols: int = 3) -> None:
    if not frames:
        return
    h, w, _ = frames[0].shape
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        image = Image.fromarray(frame)
        x = c * w
        y = r * h
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 180, y + 28), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), f"frame {idx}", fill=(255, 255, 255))
    canvas.save(output_path)


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(q_xyzw, dtype=np.float32)]
    n = max((x * x + y * y + z * z + w * w) ** 0.5, 1.0e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_xyzw_to_rotmat(q_xyzw) @ np.asarray(v, dtype=np.float32)


def _transform_point_np(tf_xyzw: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    pos = np.asarray(tf_xyzw[:3], dtype=np.float32)
    quat = np.asarray(tf_xyzw[3:7], dtype=np.float32)
    return pos + _quat_rotate_np(quat, p_local)


def _transform_wrench_body_to_world(tf_xyzw: np.ndarray, wrench_body: np.ndarray) -> np.ndarray:
    quat = np.asarray(tf_xyzw[3:7], dtype=np.float32)
    f_world = _quat_rotate_np(quat, wrench_body[:3])
    tau_world = _quat_rotate_np(quat, wrench_body[3:6])
    return np.concatenate([f_world, tau_world], axis=0).astype(np.float32, copy=False)


def _parse_contact_count(count_array: wp.array) -> int:
    return int(np.asarray(count_array.numpy()).reshape(-1)[0])


@dataclass
class FingerShapeSpec:
    body_name: str
    mesh: Any
    local_xform: Any
    scale: tuple[float, float, float]
    label: str
    is_pad: bool


@dataclass
class RigidStepMetrics:
    frame: int
    time_s: float
    target_xyz: np.ndarray
    actual_target_xyz: np.ndarray
    target_error_m: float
    actual_ee_xyz: np.ndarray
    ee_error_m: float
    robot_table_contacts: int
    finger_table_contacts: int
    nonfinger_table_contacts: int


@dataclass
class WindowMetrics:
    rope_table_contact_frames: int
    rope_ground_contact_frames: int
    robot_table_contact_frames: int
    max_rope_com_z: float
    max_abs_delta_rope_com_z: float
    max_ground_penetration_m: float
    max_table_penetration_m: float
    max_support_penetration_m: float
    final_ground_penetration_p99_m: float
    final_table_penetration_p99_m: float
    final_support_penetration_p99_m: float


@dataclass
class CalibrationCandidate:
    ground_shape_contact_scale: float
    ground_shape_contact_damping_multiplier: float
    table_shape_contact_scale: float
    table_shape_contact_damping_multiplier: float
    finger_shape_contact_scale: float
    finger_shape_contact_damping_multiplier: float
    table_edge_inset_y: float
    overhang_drop_factor: float


def _scale_contact_triplet(ke: float, kd: float, kf: float, scale: float, damping_mult: float) -> tuple[float, float, float]:
    return (
        float(ke * scale),
        float(kd * scale * damping_mult),
        float(kf * scale * damping_mult),
    )


def _box_signed_distance_np(points: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    q = np.abs(points - center[None, :]) - half_extents[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _penetration_stats(depths: np.ndarray) -> tuple[float, float]:
    if depths.size == 0:
        return 0.0, 0.0
    return float(np.max(depths)), float(np.quantile(depths, 0.99))


def _support_soft_contact_scale(args: argparse.Namespace) -> tuple[float, float]:
    support_scale = 0.5 * (float(args.ground_shape_contact_scale) + float(args.table_shape_contact_scale))
    support_damping = 0.5 * (
        float(args.ground_shape_contact_damping_multiplier) + float(args.table_shape_contact_damping_multiplier)
    )
    return support_scale, support_damping


def _apply_split_shape_contact_scaling(model: Any, args: argparse.Namespace) -> None:
    support_scale, support_damping = _support_soft_contact_scale(args)
    soft_ke, soft_kd, soft_kf = _scale_contact_triplet(
        float(model.soft_contact_ke),
        float(model.soft_contact_kd),
        float(model.soft_contact_kf),
        support_scale,
        support_damping,
    )
    model.soft_contact_ke = soft_ke
    model.soft_contact_kd = soft_kd
    model.soft_contact_kf = soft_kf

    labels = [str(label).lower() for label in list(model.shape_label)]
    for arr_name in ("shape_material_ke", "shape_material_kd", "shape_material_kf"):
        arr = getattr(model, arr_name)
        vals = arr.numpy().astype(np.float32, copy=False).copy()
        for idx, label in enumerate(labels):
            if "rope_ground" in label or "ground" in label or "plane" in label:
                scale = float(args.ground_shape_contact_scale)
                damping_mult = float(args.ground_shape_contact_damping_multiplier)
            elif "rope_table" in label or "table" in label:
                scale = float(args.table_shape_contact_scale)
                damping_mult = float(args.table_shape_contact_damping_multiplier)
            elif "mirror_" in label:
                scale = float(args.finger_shape_contact_scale)
                damping_mult = float(args.finger_shape_contact_damping_multiplier)
            else:
                scale = 1.0
                damping_mult = 1.0
            vals[idx] *= np.float32(scale)
            if arr_name != "shape_material_ke":
                vals[idx] *= np.float32(damping_mult)
        arr.assign(vals)


class NativePandaRigidSide:
    def __init__(self, viewer: ViewerGL, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.video_fps = int(args.video_fps)
        self.frame_dt = 1.0 / float(self.video_fps)
        self.sim_substeps = int(args.sim_substeps_rigid)
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.frame_idx = 0

        self.table_half_extents = np.array([0.10, 0.10, 0.05], dtype=np.float32)
        self.table_pos = np.array([0.08, -0.50, 0.05], dtype=np.float32)
        self.table_top_z = float(self.table_pos[2] + self.table_half_extents[2])

        self.push_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.presentation_contact_point: np.ndarray | None = None
        self.presentation_scene_center: np.ndarray | None = None
        self.position_target_link_index = -1
        self.position_target_local_offset = np.zeros(3, dtype=np.float32)
        self.drive_target_is_pad_center = False
        self._build_scene()
        self._setup_ik()
        self._setup_motion_defaults()
        self._set_camera()

    def _build_scene(self) -> None:
        sdf_max_resolution = 64
        sdf_narrow_band_range = (-0.01, 0.01)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        # Follow upstream panda_hydro style: keep a plain base config, then derive
        # explicit hydroelastic configs for meshes and primitives.
        shape_cfg_meshes = replace(shape_cfg, is_hydroelastic=True)
        shape_cfg_primitives = replace(
            shape_cfg,
            is_hydroelastic=True,
            sdf_max_resolution=sdf_max_resolution,
            sdf_narrow_band_range=sdf_narrow_band_range,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        def find_body(name: str) -> int:
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        self.left_finger_body_idx = find_body("fr3_leftfinger")
        self.right_finger_body_idx = find_body("fr3_rightfinger")
        self.hand_body_idx = find_body("fr3_hand")
        self.ee_index = 10

        finger_body_indices = {self.left_finger_body_idx, self.right_finger_body_idx, self.hand_body_idx}
        self.finger_shape_indices: list[int] = []
        self.nonfinger_shape_indices: list[int] = []
        self.contact_shape_specs: list[FingerShapeSpec] = []
        self.pad_center_locals: dict[str, np.ndarray] = {}

        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=sdf_max_resolution,
                        narrow_band_range=sdf_narrow_band_range,
                        margin=shape_cfg.gap,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
                self.finger_shape_indices.append(shape_idx)
            elif body_idx >= 0:
                self.nonfinger_shape_indices.append(shape_idx)

        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=self.nonfinger_shape_indices,
            keep_visual_shapes=True,
        )

        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.04, 0.04]
        builder.joint_target_pos[:9] = [*init_q, 0.04, 0.04]
        builder.joint_target_ke[:9] = [650.0] * 9
        builder.joint_target_kd[:9] = [100.0] * 9
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [20.0] * 2
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        pad_asset_path = newton.utils.download_asset("manipulation_objects/pad")
        pad_stage = Usd.Stage.Open(str(pad_asset_path / "model.usda"))
        pad_mesh = newton.usd.get_mesh(
            pad_stage.GetPrimAtPath("/root/Model/Model"),
            load_normals=True,
            face_varying_normal_conversion="vertex_splitting",
        )
        pad_scale = np.asarray(newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
        if not np.allclose(pad_scale, 1.0):
            pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
        pad_mesh.build_sdf(
            max_resolution=sdf_max_resolution,
            narrow_band_range=sdf_narrow_band_range,
            margin=shape_cfg.gap,
        )
        self.pad_xform = wp.transform(
            wp.vec3(0.0, 0.005, 0.045),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )
        left_pad_shape = builder.add_shape_mesh(
            body=self.left_finger_body_idx,
            mesh=pad_mesh,
            xform=self.pad_xform,
            cfg=shape_cfg_meshes,
            label="left_pad",
        )
        right_pad_shape = builder.add_shape_mesh(
            body=self.right_finger_body_idx,
            mesh=pad_mesh,
            xform=self.pad_xform,
            cfg=shape_cfg_meshes,
            label="right_pad",
        )
        self.finger_shape_indices.extend([left_pad_shape, right_pad_shape])

        table_mesh = newton.Mesh.create_box(
            float(self.table_half_extents[0]),
            float(self.table_half_extents[1]),
            float(self.table_half_extents[2]),
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        table_mesh.build_sdf(
            max_resolution=sdf_max_resolution,
            narrow_band_range=sdf_narrow_band_range,
            margin=shape_cfg.gap,
        )
        self.table_shape_idx = builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(*self.table_pos.tolist()), wp.quat_identity()),
            cfg=shape_cfg_meshes,
            label="split_probe_table",
        )

        builder.add_ground_plane(cfg=shape_cfg_primitives)

        for body_name, body_idx in (("left", self.left_finger_body_idx), ("right", self.right_finger_body_idx)):
            for shape_idx in builder.body_shapes[body_idx]:
                if builder.shape_type[shape_idx] != newton.GeoType.MESH:
                    continue
                mesh = builder.shape_source[shape_idx]
                if mesh is None:
                    continue
                label = str(builder.shape_label[shape_idx])
                is_pad = "pad" in label
                local_xform = builder.shape_transform[shape_idx]
                scale = tuple(float(v) for v in builder.shape_scale[shape_idx])
                self.contact_shape_specs.append(
                    FingerShapeSpec(
                        body_name=body_name,
                        mesh=mesh,
                        local_xform=local_xform,
                        scale=scale,
                        label=label,
                        is_pad=is_pad,
                    )
                )
                if is_pad:
                    self.pad_center_locals[body_name] = np.asarray(local_xform[:3], dtype=np.float32)

        self.model_single = copy.deepcopy(builder).finalize()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=hasattr(self.viewer, "renderer"),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )
        self.control = self.model.control()
        self.viewer.set_model(self.model)

    def _setup_motion_defaults(self) -> None:
        if str(self.args.video_mode) == "presentation_lifted":
            self.settle_frames = int(round(float(self.args.presentation_opening_seconds) / self.frame_dt))
            self.approach_frames = int(round(float(self.args.presentation_approach_seconds) / self.frame_dt))
            self.retract_frames = int(round(float(self.args.presentation_retract_seconds) / self.frame_dt))
        else:
            self.settle_frames = int(round(2.0 / self.frame_dt))
            self.approach_frames = int(round(0.5 / self.frame_dt))
            self.retract_frames = int(round(1.0 / self.frame_dt))
        self.push_path_length_m = 0.08
        self.push_speed_mps = 0.03
        self.push_frames = max(1, int(round(self.push_path_length_m / (self.push_speed_mps * self.frame_dt))))
        self.total_frames = self.settle_frames + self.approach_frames + self.push_frames + self.retract_frames
        self.target_y = float(self.table_pos[1] + self.table_half_extents[1] - 0.05)
        self.prepush_x = -0.08
        self.push_end_x = self.prepush_x + self.push_path_length_m
        self.prepush_z = float(self.table_top_z + float(self.args.push_clearance))
        self.retract_z = self.prepush_z + 0.06
        self.gripper_opening = float(self.args.gripper_opening)
        self.target_rotation = wp.mul(
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(self.args.gripper_yaw)),
        )
        self.leading_side_name = "right"
        self.leading_pad_offset_world = np.zeros(3, dtype=np.float32)
        self.pad_far_xyz = np.array([self.prepush_x - 0.12, self.target_y - 0.03, self.prepush_z + 0.08], dtype=np.float32)
        self.pad_push_start_xyz = np.array([self.prepush_x, self.target_y, self.prepush_z], dtype=np.float32)
        self.pad_push_end_xyz = np.array([self.push_end_x, self.target_y, self.prepush_z], dtype=np.float32)

    def configure_for_rope(self, rope_radius_m: float, rope_line_center_x: float, rope_line_y: float) -> None:
        self.target_y = float(rope_line_y)
        self.prepush_z = float(self.table_top_z + float(rope_radius_m) + 0.003)
        self.retract_z = self.prepush_z + 0.06
        self.pad_push_start_xyz = np.array([float(rope_line_center_x - 0.06), self.target_y, self.prepush_z], dtype=np.float32)
        self.pad_push_end_xyz = np.array([float(self.pad_push_start_xyz[0] + self.push_path_length_m), self.target_y, self.prepush_z], dtype=np.float32)
        self.pad_far_xyz = np.array(
            [float(self.pad_push_start_xyz[0] - 0.12), float(self.target_y - 0.03), self.prepush_z + 0.08],
            dtype=np.float32,
        )
        self._calibrate_pad_offset()

    def configure_presentation_for_rope(self, rope_points: np.ndarray, rope_radius_m: float) -> None:
        points = np.asarray(rope_points, dtype=np.float32)
        if points.size == 0:
            return
        overall_center = np.asarray(points.mean(axis=0), dtype=np.float32)
        self.presentation_scene_center = overall_center.copy()
        table_hover_z = float(self.table_top_z + max(float(rope_radius_m) * 2.5, 0.012))
        contact_point = np.array(
            [
                float(overall_center[0]),
                float(overall_center[1]),
                float(table_hover_z + 0.03),
            ],
            dtype=np.float32,
        )
        self.presentation_contact_point = contact_point.copy()

        contact_offset_x = max(float(rope_radius_m) * 3.0, 0.018)
        standoff_x = contact_offset_x + 0.03
        standoff_z = max(float(rope_radius_m) * 5.0, 0.035)
        standoff_y = 0.015
        contact_lift = max(float(rope_radius_m) * 1.0, 0.006)
        push_distance = float(self.args.presentation_push_distance)
        start_y = float(contact_point[1] - 0.005)
        end_y = float(contact_point[1] + 0.08)
        push_start_z = float(max(float(contact_point[2] + contact_lift), table_hover_z + 0.02))
        push_end_z = float(table_hover_z)
        self.target_y = start_y
        self.push_path_length_m = float(push_distance + contact_offset_x)
        self.push_speed_mps = float(self.args.presentation_push_speed_mps)
        self.push_frames = max(1, int(round(self.push_path_length_m / (self.push_speed_mps * self.frame_dt))))
        self.total_frames = self.settle_frames + self.approach_frames + self.push_frames + self.retract_frames
        self.pad_far_xyz = np.array(
            [
                float(contact_point[0] - standoff_x),
                float(start_y - standoff_y),
                float(push_start_z + standoff_z),
            ],
            dtype=np.float32,
        )
        self.pad_push_start_xyz = np.array(
            [
                float(contact_point[0] - contact_offset_x),
                float(start_y),
                float(push_start_z),
            ],
            dtype=np.float32,
        )
        self.pad_push_end_xyz = np.array(
            [
                float(contact_point[0] + push_distance),
                float(end_y),
                float(push_end_z),
            ],
            dtype=np.float32,
        )
        self.prepush_z = float(self.pad_push_start_xyz[2])
        self.retract_z = float(self.pad_push_end_xyz[2] + 0.05)
        self._calibrate_pad_offset()
        self._snap_to_target_position(self.pad_far_xyz)
        self._set_camera()

    def _solve_single_fk(
        self, ee_target_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.pos_obj.set_target_positions(wp.array([ee_target_xyz], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        q_np = self.joint_q_ik.numpy().astype(np.float32, copy=False).reshape(-1)
        state_tmp = self.model_single.state()
        q_wp = wp.array(q_np, dtype=wp.float32, device=self.model_single.device)
        qd_wp = wp.zeros(self.model_single.joint_dof_count, dtype=wp.float32, device=self.model_single.device)
        newton.eval_fk(self.model_single, q_wp, qd_wp, state_tmp)
        body_q_np = state_tmp.body_q.numpy().astype(np.float32, copy=False)
        ee_xyz = np.asarray(body_q_np[self.ee_index][:3], dtype=np.float32)
        side_centers = {}
        side_body_q = {}
        for side, body_idx in (("left", self.left_finger_body_idx), ("right", self.right_finger_body_idx)):
            center = _transform_point_np(body_q_np[body_idx], self.pad_center_locals[side])
            side_centers[side] = center
            side_body_q[side] = body_q_np[body_idx].copy()
        return q_np, ee_xyz, side_centers, side_body_q

    def _calibrate_pad_offset(self) -> None:
        nominal_ee = self.pad_push_start_xyz.copy()
        _, ee_xyz, side_centers, _ = self._solve_single_fk(nominal_ee)
        projections = {side: float(np.dot(center, self.push_direction)) for side, center in side_centers.items()}
        self.leading_side_name = max(projections, key=projections.get)
        self.leading_pad_offset_world = ee_xyz - side_centers[self.leading_side_name]
        if str(self.args.video_mode) == "presentation_lifted":
            leading_body_idx = self.right_finger_body_idx if self.leading_side_name == "right" else self.left_finger_body_idx
            self._configure_position_target_ik(leading_body_idx, self.pad_center_locals[self.leading_side_name])
        else:
            self._configure_position_target_ik(self.ee_index, np.zeros(3, dtype=np.float32))
        self.parking_mirror_state = {
            "left": {
                "body_q": np.asarray([-0.55, -0.95, 0.45, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "body_qd": np.zeros(6, dtype=np.float32),
            },
            "right": {
                "body_q": np.asarray([-0.52, -0.95, 0.45, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "body_qd": np.zeros(6, dtype=np.float32),
            },
        }

    def _snap_to_target_position(self, ee_target_xyz: np.ndarray) -> None:
        q_np, _, _, _ = self._solve_single_fk(np.asarray(ee_target_xyz, dtype=np.float32))
        q_np = np.asarray(q_np, dtype=np.float32).reshape(-1)
        q_single = q_np.reshape(1, -1)
        q_full = self.model.joint_q.numpy().astype(np.float32, copy=False)
        qd_full = self.model.joint_qd.numpy().astype(np.float32, copy=False)
        q_full[: q_np.shape[0]] = q_np
        qd_full[:] = 0.0
        self.model.joint_q.assign(q_full)
        self.model.joint_qd.assign(qd_full)

        q_wp = wp.array(q_full, dtype=wp.float32, device=self.model.device)
        qd_wp = wp.array(qd_full, dtype=wp.float32, device=self.model.device)
        self.state_0.joint_q.assign(q_full)
        self.state_0.joint_qd.assign(qd_full)
        self.state_1.joint_q.assign(q_full)
        self.state_1.joint_qd.assign(qd_full)
        newton.eval_fk(self.model, q_wp, qd_wp, self.state_0)
        newton.eval_fk(self.model, q_wp, qd_wp, self.state_1)
        self.model_single.joint_q.assign(q_np)
        self.model_single.joint_qd.zero_()
        newton.eval_fk(self.model_single, self.model_single.joint_q, self.model_single.joint_qd, self.state_single)
        self.joint_q_ik = wp.array(q_single, dtype=wp.float32, device=self.model_single.device)
        self.pos_obj.set_target_positions(wp.array([ee_target_xyz], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.target_rotation)], dtype=wp.vec4))
        self.control.joint_target_pos.zero_()
        wp.copy(self.control.joint_target_pos, wp.array(q_np, dtype=wp.float32, device=self.model.device))
        self.control.joint_target_pos[7:9].fill_(self.gripper_opening)
        self.sim_time = 0.0
        self.frame_idx = 0

    def _setup_ik(self) -> None:
        self.state_single = self.model_single.state()
        newton.eval_fk(self.model_single, self.model_single.joint_q, self.model_single.joint_qd, self.state_single)
        body_q_np = self.state_single.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        )
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_single.joint_limit_lower,
            joint_limit_upper=self.model_single.joint_limit_upper,
        )
        self.joint_q_ik = wp.array(self.model_single.joint_q, shape=(1, self.model_single.joint_coord_count))
        self.ik_iters = 24
        self._configure_position_target_ik(self.ee_index, np.zeros(3, dtype=np.float32))

    def _configure_position_target_ik(self, link_index: int, local_offset: np.ndarray) -> None:
        local_offset = np.asarray(local_offset, dtype=np.float32).reshape(3)
        body_q_np = self.state_single.body_q.numpy()
        target_world = _transform_point_np(body_q_np[link_index], local_offset)
        self.position_target_link_index = int(link_index)
        self.position_target_local_offset = local_offset.copy()
        self.drive_target_is_pad_center = bool(link_index != self.ee_index or np.linalg.norm(local_offset) > 0.0)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=link_index,
            link_offset=wp.vec3(float(local_offset[0]), float(local_offset[1]), float(local_offset[2])),
            target_positions=wp.array([target_world], dtype=wp.vec3),
        )
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _set_camera(self) -> None:
        if str(self.args.video_mode) == "presentation_lifted" and self.presentation_contact_point is not None:
            focus = (
                np.asarray(self.presentation_scene_center, dtype=np.float32)
                if self.presentation_scene_center is not None
                else np.asarray(self.presentation_contact_point, dtype=np.float32)
            )
            self.viewer.set_camera(
                wp.vec3(float(focus[0] + 0.44), float(focus[1] + 0.18), float(focus[2] + 0.25)),
                -18.0,
                -144.0,
            )
        else:
            self.viewer.set_camera(wp.vec3(0.36, -0.03, 0.33), -14.0, -120.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 82.0 if str(self.args.video_mode) == "presentation_lifted" else 68.0

    def current_target_position(self) -> np.ndarray:
        if self.frame_idx < self.settle_frames:
            pad_xyz = self.pad_far_xyz
        elif self.frame_idx < self.settle_frames + self.approach_frames:
            alpha = (self.frame_idx - self.settle_frames) / max(self.approach_frames - 1, 1)
            pad_xyz = (1.0 - alpha) * self.pad_far_xyz + alpha * self.pad_push_start_xyz
        elif self.frame_idx < self.settle_frames + self.approach_frames + self.push_frames:
            alpha = (self.frame_idx - self.settle_frames - self.approach_frames) / max(self.push_frames - 1, 1)
            pad_xyz = (1.0 - alpha) * self.pad_push_start_xyz + alpha * self.pad_push_end_xyz
        else:
            alpha = (self.frame_idx - self.settle_frames - self.approach_frames - self.push_frames) / max(self.retract_frames - 1, 1)
            pad_xyz = (1.0 - alpha) * self.pad_push_end_xyz + alpha * (self.pad_far_xyz + np.array([0.02, 0.0, 0.0], dtype=np.float32))
        if self.drive_target_is_pad_center:
            return np.asarray(pad_xyz, dtype=np.float32).copy()
        return pad_xyz + self.leading_pad_offset_world

    def _actual_position_target_world(self, body_q: np.ndarray) -> np.ndarray:
        if self.position_target_link_index == self.ee_index and np.linalg.norm(self.position_target_local_offset) == 0.0:
            return np.asarray(body_q[self.ee_index][:3], dtype=np.float32)
        return _transform_point_np(body_q[self.position_target_link_index], self.position_target_local_offset)

    def _robot_table_counts(self) -> tuple[int, int, int]:
        count = _parse_contact_count(self.contacts.rigid_contact_count)
        if count <= 0:
            return 0, 0, 0
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count].astype(np.int32, copy=False)
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count].astype(np.int32, copy=False)
        robot_total = 0
        finger = 0
        nonfinger = 0
        finger_shapes = set(self.finger_shape_indices)
        nonfinger_shapes = set(self.nonfinger_shape_indices)
        for s0, s1 in zip(shape0, shape1, strict=False):
            pair = {int(s0), int(s1)}
            if self.table_shape_idx not in pair:
                continue
            other = int(s1) if int(s0) == self.table_shape_idx else int(s0)
            robot_total += 1
            if other in finger_shapes:
                finger += 1
            elif other in nonfinger_shapes:
                nonfinger += 1
        return robot_total, finger, nonfinger

    def step(self, external_world_wrenches: dict[str, np.ndarray] | None = None) -> RigidStepMetrics:
        target_position = self.current_target_position()
        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        self.control.joint_target_pos.zero_()
        wp.copy(self.control.joint_target_pos, self.joint_q_ik.flatten())
        self.control.joint_target_pos[7:9].fill_(self.gripper_opening)

        external = external_world_wrenches or {}
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            if external:
                body_f = self.state_0.body_f.numpy()
                for body_name, wrench in external.items():
                    body_idx = self.left_finger_body_idx if body_name == "left" else self.right_finger_body_idx
                    body_f[body_idx, :6] += np.asarray(wrench, dtype=np.float32)
                self.state_0.body_f.assign(body_f)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        body_q = self.state_0.body_q.numpy()
        actual_ee = np.asarray(body_q[self.ee_index][:3], dtype=np.float32)
        actual_target = self._actual_position_target_world(body_q)
        robot_total, finger_count, nonfinger_count = self._robot_table_counts()
        target_error = float(np.linalg.norm(target_position - actual_target))
        ee_target = target_position if not self.drive_target_is_pad_center else (target_position + self.leading_pad_offset_world)
        ee_error = float(np.linalg.norm(ee_target - actual_ee))
        metrics = RigidStepMetrics(
            frame=int(self.frame_idx),
            time_s=float(self.sim_time),
            target_xyz=target_position,
            actual_target_xyz=actual_target,
            target_error_m=target_error,
            actual_ee_xyz=actual_ee,
            ee_error_m=ee_error,
            robot_table_contacts=robot_total,
            finger_table_contacts=finger_count,
            nonfinger_table_contacts=nonfinger_count,
        )
        self.frame_idx += 1
        self.sim_time += self.frame_dt
        return metrics

    def pad_centers_world(self) -> dict[str, np.ndarray]:
        body_q = self.state_0.body_q.numpy().astype(np.float32, copy=False)
        return {
            side: _transform_point_np(body_q[body_idx], self.pad_center_locals[side])
            for side, body_idx in (("left", self.left_finger_body_idx), ("right", self.right_finger_body_idx))
        }

    def leading_pad_world_xyz(self) -> np.ndarray:
        return self.pad_centers_world()[self.leading_side_name].copy()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

    def end_render(self) -> None:
        self.viewer.end_frame()

    def mirror_body_state(self) -> dict[str, dict[str, np.ndarray]]:
        body_q = self.state_0.body_q.numpy().astype(np.float32, copy=False)
        body_qd = self.state_0.body_qd.numpy().astype(np.float32, copy=False)
        return {
            "left": {"body_q": body_q[self.left_finger_body_idx].copy(), "body_qd": body_qd[self.left_finger_body_idx].copy()},
            "right": {"body_q": body_q[self.right_finger_body_idx].copy(), "body_qd": body_qd[self.right_finger_body_idx].copy()},
        }

    def leading_side(self) -> str:
        return self.leading_side_name

    def parked_mirror_state(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            side: {
                "body_q": self.parking_mirror_state[side]["body_q"].copy(),
                "body_qd": self.parking_mirror_state[side]["body_qd"].copy(),
            }
            for side in ("left", "right")
        }


class SemiImplicitRopeSide:
    def __init__(self, args: argparse.Namespace, rigid: NativePandaRigidSide):
        self.args = args
        self.rigid = rigid
        self.device = args.device
        self._build_rope_ir()
        self._build_model()
        self._setup_render_metadata()
        self.frame_idx = 0
        self.last_world_wrench = {"left": np.zeros(6, dtype=np.float32), "right": np.zeros(6, dtype=np.float32)}

    def _build_rope_ir(self) -> None:
        raw_ir = load_ir(Path(self.args.ir))
        _validate_scaling_args(self.args)
        _maybe_autoset_mass_spring_scale(self.args, raw_ir)
        rope_ir = _copy_object_only_ir(raw_ir, self.args)
        x0 = np.asarray(rope_ir["x0"], dtype=np.float32).copy()
        bbox_min = x0.min(axis=0)
        bbox_max = x0.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        rope_radius_m = float(np.mean(np.asarray(rope_ir["collision_radius"], dtype=np.float32)))
        z_clearance = float(self.rigid.table_top_z + rope_radius_m + float(self.args.min_clearance_extra))
        center_z_for_table_clearance = float(z_clearance - float(bbox_min[2] - center[2]))

        target_center = np.array(
            [
                (
                    max(
                        float(self.rigid.prepush_x + 0.05),
                        float(
                            self.rigid.table_pos[0]
                            - self.rigid.table_half_extents[0]
                            + 0.5 * float(bbox_max[0] - bbox_min[0])
                            + float(self.args.min_clearance_extra)
                        ),
                    )
                    if str(self.args.video_mode) == "presentation_lifted"
                    else float(self.rigid.prepush_x + 0.05)
                ),
                self.rigid.table_pos[1]
                - self.rigid.table_half_extents[1]
                + (
                    float(self.args.presentation_table_edge_inset_y)
                    if str(self.args.video_mode) == "presentation_lifted"
                    else float(self.args.table_edge_inset_y)
                ),
                center_z_for_table_clearance,
            ],
            dtype=np.float32,
        )
        if str(self.args.video_mode) == "presentation_lifted":
            target_center[2] += float(self.args.presentation_lift_height)
        shift = target_center - center
        shifted = (x0 + shift).astype(np.float32, copy=False)
        table_edge_y = float(self.rigid.table_pos[1] - self.rigid.table_half_extents[1])
        overhang_mask = shifted[:, 1] < table_edge_y
        if np.any(overhang_mask):
            hang = (table_edge_y - shifted[overhang_mask, 1]).astype(np.float32, copy=False)
            drop_factor = (
                float(self.args.presentation_overhang_drop_factor)
                if str(self.args.video_mode) == "presentation_lifted"
                else float(self.args.overhang_drop_factor)
            )
            shifted[overhang_mask, 2] -= drop_factor * hang
            min_clearance = rope_radius_m + float(self.args.min_clearance_extra)
            shifted[:, 2] = np.maximum(shifted[:, 2], min_clearance)
        rope_ir["x0"] = shifted
        rope_ir["v0"] = np.zeros_like(rope_ir["x0"], dtype=np.float32)
        self.rope_ir = rope_ir
        self.rope_shift = shift.astype(np.float32, copy=False)
        self.n_obj = int(np.asarray(rope_ir["num_object_points"]).ravel()[0])
        self.rope_center_target_x = float(target_center[0])
        self.rope_line_target_y = float(target_center[1])

    def _contact_cfg(self, params: dict[str, float]) -> Any:
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = float(params["ke"])
        cfg.kd = float(params["kd"])
        cfg.kf = float(params["kf"])
        cfg.mu = float(params["mu"])
        return cfg

    def _build_model(self) -> None:
        spring_ke_scale, spring_kd_scale = _effective_spring_scales(self.rope_ir, self.args)
        cfg = newton_import_ir.SimConfig(
            ir_path=Path(self.args.ir).resolve(),
            out_dir=Path(self.args.out_dir).resolve(),
            output_prefix="robot_table_rope_split",
            spring_ke_scale=float(spring_ke_scale),
            spring_kd_scale=float(spring_kd_scale),
            angular_damping=float(self.args.angular_damping),
            friction_smoothing=float(self.args.friction_smoothing),
            enable_tri_contact=True,
            disable_particle_contact_kernel=True,
            shape_contacts=True,
            add_ground_plane=True,
            strict_physics_checks=False,
            apply_drag=bool(self.args.apply_drag),
            drag_damping_scale=float(self.args.drag_damping_scale),
            gravity=-float(self.args.gravity_mag),
            gravity_from_reverse_z=False,
            up_axis="Z",
            particle_contacts=False,
            device=self.device,
        )
        rope_checks = newton_import_ir.validate_ir_physics(self.rope_ir, cfg)
        rope_contact = self._map_object_contact_params(self.rope_ir, cfg)
        rope_ground = self._map_ground_contact_params(self.rope_ir, cfg)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        radius, _, _ = newton_import_ir._add_particles(builder, self.rope_ir, cfg, particle_contacts=False)
        newton_import_ir._add_springs(builder, self.rope_ir, cfg, rope_checks)
        ground_cfg = self._contact_cfg(rope_ground)
        self.ground_shape_idx = builder.add_ground_plane(cfg=ground_cfg, label="rope_ground")

        shape_cfg = self._contact_cfg(rope_contact)
        table_mesh = newton.Mesh.create_box(
            float(self.rigid.table_half_extents[0]),
            float(self.rigid.table_half_extents[1]),
            float(self.rigid.table_half_extents[2]),
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        table_mesh.build_sdf(
            max_resolution=64,
            narrow_band_range=(-0.01, 0.01),
            margin=0.05,
        )
        self.table_shape_idx = builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(*self.rigid.table_pos.tolist()), wp.quat_identity()),
            cfg=shape_cfg,
            label="rope_table",
        )

        self.mirror_body_ids: dict[str, int] = {}
        self.mirror_joint_q_start: dict[str, int] = {}
        self.mirror_joint_qd_start: dict[str, int] = {}
        self.side_shape_indices = {"left": [], "right": []}
        self.pad_shape_indices = {"left": [], "right": []}

        for side in ("left", "right"):
            state = self.rigid.mirror_body_state()[side]
            body_id = builder.add_link(
                xform=wp.transform(*state["body_q"]),
                label=f"mirror_{side}_finger",
                is_kinematic=True,
            )
            joint_id = builder.add_joint_free(child=body_id, label=f"mirror_{side}_joint")
            builder.add_articulation([joint_id], label=f"mirror_{side}_articulation")
            self.mirror_body_ids[side] = body_id
            self.mirror_joint_q_start[side] = builder.joint_q_start[joint_id]
            self.mirror_joint_qd_start[side] = builder.joint_qd_start[joint_id]

        for spec in self.rigid.contact_shape_specs:
            body_id = self.mirror_body_ids[spec.body_name]
            mesh_copy = spec.mesh.copy()
            shape_idx = builder.add_shape_mesh(
                body=body_id,
                mesh=mesh_copy,
                xform=spec.local_xform,
                scale=spec.scale,
                cfg=shape_cfg,
                label=f"mirror_{spec.body_name}_{spec.label}",
            )
            self.side_shape_indices[spec.body_name].append(shape_idx)
            if spec.is_pad:
                self.pad_shape_indices[spec.body_name].append(shape_idx)

        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, 0.0, -float(self.args.gravity_mag)))
        self.model.soft_contact_ke = float(rope_contact["ke"])
        self.model.soft_contact_kd = float(rope_contact["kd"])
        self.model.soft_contact_kf = float(rope_contact["kf"])
        self.model.soft_contact_mu = float(rope_contact["mu"])
        _apply_split_shape_contact_scaling(self.model, self.args)

        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            angular_damping=float(self.args.angular_damping),
            friction_smoothing=float(self.args.friction_smoothing),
            enable_tri_contact=True,
        )
        self.physical_radius = self.model.particle_radius.numpy().astype(np.float32, copy=False)[: self.n_obj].copy()
        self.render_radius = self.physical_radius.copy()
        self.rope_spring_edges = np.asarray(self.rope_ir["spring_edges"], dtype=np.int32).copy()
        self.drag = float(newton_import_ir.ir_scalar(self.rope_ir, "drag_damping")) * float(self.args.drag_damping_scale) if self.args.apply_drag and "drag_damping" in self.rope_ir else 0.0

        gravity_vec = np.array([0.0, 0.0, -float(self.args.gravity_mag)], dtype=np.float32)
        gravity_norm = float(np.linalg.norm(gravity_vec))
        self.gravity_axis = None
        if gravity_norm > 1.0e-12:
            self.gravity_axis = (-gravity_vec / gravity_norm).astype(np.float32)

    def _map_object_contact_params(self, ir: dict[str, Any], cfg: newton_import_ir.SimConfig) -> dict[str, float]:
        elas_raw = float(newton_import_ir.ir_scalar(ir, "contact_collide_object_elas"))
        fric_raw = float(newton_import_ir.ir_scalar(ir, "contact_collide_object_fric"))
        mu = float(np.clip(fric_raw, 0.0, newton_import_ir.MU_MAX))
        zeta = float(newton_import_ir._restitution_to_damping_ratio(elas_raw))
        ke = 1.0e3
        m_ref = float(newton_import_ir._resolve_object_mass_reference(ir))
        m_eff_ref = max(0.5 * m_ref, float(newton_import_ir.EPSILON))
        kd = 2.0 * zeta * np.sqrt(ke * m_eff_ref)
        kf = kd
        return {"ke": float(ke), "kd": float(kd), "kf": float(kf), "mu": float(mu)}

    def _map_ground_contact_params(self, ir: dict[str, Any], cfg: newton_import_ir.SimConfig) -> dict[str, float]:
        tmp_builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)
        gcfg = tmp_builder.default_shape_cfg.copy()
        checks: dict[str, Any] = {}
        newton_import_ir._configure_ground_contact_material(gcfg, ir, cfg, checks, context="split_ground")
        return {"ke": float(gcfg.ke), "kd": float(gcfg.kd), "kf": float(gcfg.kf), "mu": float(gcfg.mu)}

    def _setup_render_metadata(self) -> None:
        self.rope_color = wp.array(np.tile(np.asarray([[0.96, 0.78, 0.47]], dtype=np.float32), (self.n_obj, 1)), dtype=wp.vec3)
        stride = max(1, int(self.args.rope_spring_stride))
        self.render_edges = self.rope_spring_edges[::stride].astype(np.int32, copy=False)

    def _sync_mirror_body(self, side: str, pose: dict[str, np.ndarray]) -> None:
        body_q = self.state_in.body_q.numpy()
        body_qd = self.state_in.body_qd.numpy()
        q = self.state_in.joint_q.numpy()
        qd = self.state_in.joint_qd.numpy()
        body_idx = self.mirror_body_ids[side]
        q_start = self.mirror_joint_q_start[side]
        qd_start = self.mirror_joint_qd_start[side]

        body_q[body_idx] = pose["body_q"]
        body_qd[body_idx] = pose["body_qd"]
        q[q_start : q_start + 7] = pose["body_q"]
        qd[qd_start : qd_start + 6] = pose["body_qd"]

        self.state_in.body_q.assign(body_q)
        self.state_in.body_qd.assign(body_qd)
        self.state_in.joint_q.assign(q)
        self.state_in.joint_qd.assign(qd)

        self.state_out.body_q.assign(body_q)
        self.state_out.body_qd.assign(body_qd)
        self.state_out.joint_q.assign(q)
        self.state_out.joint_qd.assign(qd)

    def sync_mirrors(self, mirror_state: dict[str, dict[str, np.ndarray]]) -> None:
        for side in ("left", "right"):
            self._sync_mirror_body(side, mirror_state[side])

    def step_frame(self, mirror_state: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
        self.sync_mirrors(mirror_state)
        contact_any = {"left": 0, "right": 0, "ground": 0, "table": 0}
        reaction_world = {"left": np.zeros(6, dtype=np.float32), "right": np.zeros(6, dtype=np.float32)}

        sim_dt = float(
            self.args.rope_sim_dt
            if self.args.rope_sim_dt is not None
            else (self.rigid.frame_dt / float(self.args.sim_substeps_rope))
        )
        for _ in range(int(self.args.sim_substeps_rope)):
            self.state_in.clear_forces()
            self.model.collide(self.state_in, self.contacts)
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, sim_dt)

            soft_count = _parse_contact_count(self.contacts.soft_contact_count)
            if soft_count > 0:
                soft_shapes = self.contacts.soft_contact_shape.numpy()[:soft_count].astype(np.int32, copy=False)
                if np.any(soft_shapes == self.ground_shape_idx):
                    contact_any["ground"] += 1
                if np.any(soft_shapes == self.table_shape_idx):
                    contact_any["table"] += 1
                for side in ("left", "right"):
                    side_shapes = set(self.side_shape_indices[side])
                    count_side = sum(int(s in side_shapes) for s in soft_shapes.tolist())
                    contact_any[side] += count_side

            if self.args.coupling_mode == "two_way":
                body_f_np = self.state_out.body_f.numpy().astype(np.float32, copy=False)
                body_q_np = self.state_out.body_q.numpy().astype(np.float32, copy=False)
                for side in ("left", "right"):
                    body_idx = self.mirror_body_ids[side]
                    local_wrench = body_f_np[body_idx, :6].copy()
                    reaction_world[side] += _transform_wrench_body_to_world(body_q_np[body_idx], local_wrench)

            self.state_in, self.state_out = self.state_out, self.state_in

            if self.drag > 0.0 and self.gravity_axis is not None:
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=self.n_obj,
                    inputs=[self.state_in.particle_q, self.state_in.particle_qd, self.n_obj, sim_dt, self.drag],
                    device=self.device,
                )

        self.frame_idx += 1
        self.last_world_wrench = reaction_world
        return {
            "contact_any": contact_any,
            "reaction_world": reaction_world,
            "particle_q": self.state_in.particle_q.numpy().astype(np.float32, copy=False)[: self.n_obj].copy(),
        }

    def rope_points_wp(self) -> wp.array:
        return wp.array(self.state_in.particle_q.numpy().astype(np.float32, copy=False)[: self.n_obj].copy(), dtype=wp.vec3)

    def rope_radii_wp(self) -> wp.array:
        return wp.array(self.render_radius.astype(np.float32, copy=False), dtype=wp.float32)

    def rope_line_buffers(self) -> tuple[wp.array, wp.array] | tuple[None, None]:
        points = self.state_in.particle_q.numpy().astype(np.float32, copy=False)[: self.n_obj]
        if self.render_edges.size == 0:
            return None, None
        starts = wp.array(points[self.render_edges[:, 0]].astype(np.float32, copy=False), dtype=wp.vec3)
        ends = wp.array(points[self.render_edges[:, 1]].astype(np.float32, copy=False), dtype=wp.vec3)
        return starts, ends

    def support_penetration_proxy(self, particle_q: np.ndarray) -> dict[str, np.ndarray | float]:
        points = np.asarray(particle_q, dtype=np.float32)
        radii = self.physical_radius[: points.shape[0]]
        ground_depth = np.maximum(radii - points[:, 2], 0.0).astype(np.float32, copy=False)

        table_center = np.asarray(self.rigid.table_pos, dtype=np.float32)
        table_half_extents = np.asarray(self.rigid.table_half_extents, dtype=np.float32)
        table_sdf = _box_signed_distance_np(points, table_center, table_half_extents).astype(np.float32, copy=False)
        table_depth = np.maximum(radii - table_sdf, 0.0).astype(np.float32, copy=False)
        support_depth = np.maximum(ground_depth, table_depth).astype(np.float32, copy=False)

        max_ground, p99_ground = _penetration_stats(ground_depth)
        max_table, p99_table = _penetration_stats(table_depth)
        max_support, p99_support = _penetration_stats(support_depth)
        return {
            "ground_depth": ground_depth,
            "table_depth": table_depth,
            "support_depth": support_depth,
            "max_ground_penetration_m": max_ground,
            "max_table_penetration_m": max_table,
            "max_support_penetration_m": max_support,
            "p99_ground_penetration_m": p99_ground,
            "p99_table_penetration_m": p99_table,
            "p99_support_penetration_m": p99_support,
        }


class SplitDemo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.device:
            wp.set_device(args.device)

        self.viewer = ViewerGL(width=args.width, height=args.height, headless=True)
        self.rigid = NativePandaRigidSide(self.viewer, args)
        self.rope = SemiImplicitRopeSide(args, self.rigid)
        self.rigid.configure_for_rope(
            rope_radius_m=float(np.mean(self.rope.physical_radius)),
            rope_line_center_x=float(self.rope.rope_center_target_x),
            rope_line_y=float(self.rope.rope_line_target_y),
        )
        if str(self.args.video_mode) == "presentation_lifted":
            self.rigid.configure_presentation_for_rope(
                rope_points=np.asarray(self.rope.rope_ir["x0"], dtype=np.float32),
                rope_radius_m=float(np.mean(self.rope.physical_radius)),
            )
        try:
            apply_viewer_shape_colors(self.viewer, self.rigid.model)
        except Exception:
            pass

        self.leading_side = self.rigid.leading_side()
        self.trailing_side = "right" if self.leading_side == "left" else "left"
        self.timeseries: list[dict[str, Any]] = []
        self.first_finger_contact_frame: int | None = None
        self.first_rope_motion_frame: int | None = None
        self.first_contact_side: str | None = None
        self.first_contact_reaction_nonzero: int | None = None
        self.first_leading_pad_proximity_frame: int | None = None
        self.baseline_com_x = None
        self.preview_frames: list[np.ndarray] = []
        self.first_30_frames: list[np.ndarray] = []
        self.render_frame_count = self._resolve_render_frame_count()
        self.preview_indices = sorted(set(np.linspace(0, max(self.render_frame_count - 1, 0), 6, dtype=int).tolist()))

    def _record_start_mode(self) -> str:
        return "visible_opening" if str(self.args.video_mode) == "presentation_lifted" else "post_settle"

    def _effective_preroll_seconds(self) -> float:
        return 0.0 if str(self.args.video_mode) == "presentation_lifted" else float(self.args.rope_preroll_seconds)

    def _resolve_render_frame_count(self) -> int:
        requested = max(0, int(self.args.num_frames))
        if str(self.args.video_mode) == "presentation_lifted":
            full_cycle = int(self.rigid.total_frames + int(self.args.presentation_tail_frames))
            return max(requested, full_cycle)
        return requested

    def _current_mirror_state(self, control_frame: int) -> dict[str, dict[str, np.ndarray]]:
        if str(self.args.video_mode) == "presentation_lifted":
            return self.rigid.mirror_body_state()
        return self.rigid.parked_mirror_state() if control_frame < self.rigid.settle_frames else self.rigid.mirror_body_state()

    def _advance_one_frame(
        self, pending_world_wrench: dict[str, np.ndarray]
    ) -> tuple[RigidStepMetrics, dict[str, Any], np.ndarray, float, dict[str, np.ndarray], int, dict[str, Any]]:
        control_frame = int(self.rigid.frame_idx)
        rigid_metrics = self.rigid.step(
            external_world_wrenches=(pending_world_wrench if self.args.coupling_mode == "two_way" else None)
        )
        mirror_state = self._current_mirror_state(control_frame)
        rope_metrics = self.rope.step_frame(mirror_state)
        rope_q = rope_metrics["particle_q"]
        rope_com = rope_q.mean(axis=0)
        pad_centers = self.rigid.pad_centers_world()
        left_pad_center = np.asarray(pad_centers["left"], dtype=np.float32)
        right_pad_center = np.asarray(pad_centers["right"], dtype=np.float32)
        left_pad_min_distance = float(np.min(np.linalg.norm(rope_q - left_pad_center[None, :], axis=1)))
        right_pad_min_distance = float(np.min(np.linalg.norm(rope_q - right_pad_center[None, :], axis=1)))
        leading_pad_center = left_pad_center if self.leading_side == "left" else right_pad_center
        leading_pad_min_distance = left_pad_min_distance if self.leading_side == "left" else right_pad_min_distance
        pad_metrics = {
            "left_pad_center_xyz": left_pad_center,
            "right_pad_center_xyz": right_pad_center,
            "leading_pad_center_xyz": leading_pad_center,
            "left_pad_to_rope_min_distance_m": left_pad_min_distance,
            "right_pad_to_rope_min_distance_m": right_pad_min_distance,
            "leading_pad_to_rope_min_distance_m": float(leading_pad_min_distance),
        }
        if control_frame == self.rigid.settle_frames - 1 and self.baseline_com_x is None:
            self.baseline_com_x = float(rope_com[0])
        rope_com_shift_x = 0.0 if self.baseline_com_x is None else float(rope_com[0] - self.baseline_com_x)
        next_pending_world_wrench = rope_metrics["reaction_world"]
        return rigid_metrics, rope_metrics, rope_com, rope_com_shift_x, next_pending_world_wrench, control_frame, pad_metrics

    def _record_metrics(
        self,
        frame_idx: int,
        rigid_metrics: RigidStepMetrics,
        rope_metrics: dict[str, Any],
        pad_metrics: dict[str, Any],
        rope_com: np.ndarray,
        rope_com_shift_x: float,
        penetration: dict[str, np.ndarray | float],
    ) -> None:
        contact_any = rope_metrics["contact_any"]
        if self.first_finger_contact_frame is None and (contact_any["left"] > 0 or contact_any["right"] > 0):
            self.first_finger_contact_frame = frame_idx
            self.first_contact_side = "left" if contact_any["left"] >= contact_any["right"] else "right"

        if (
            self.first_rope_motion_frame is None
            and self.first_finger_contact_frame is not None
            and frame_idx >= self.first_finger_contact_frame
            and abs(rope_com_shift_x) > float(self.args.rope_motion_threshold)
        ):
            self.first_rope_motion_frame = frame_idx

        if (
            self.first_contact_reaction_nonzero is None
            and self.args.coupling_mode == "two_way"
            and any(np.linalg.norm(v[:3]) > 1.0e-6 or np.linalg.norm(v[3:]) > 1.0e-6 for v in rope_metrics["reaction_world"].values())
        ):
            self.first_contact_reaction_nonzero = frame_idx

        if (
            self.first_leading_pad_proximity_frame is None
            and float(pad_metrics["leading_pad_to_rope_min_distance_m"]) <= float(self.args.presentation_proximity_threshold)
        ):
            self.first_leading_pad_proximity_frame = frame_idx

        self.timeseries.append(
            {
                "frame": int(frame_idx),
                "time_s": float(rigid_metrics.time_s),
                "target_x_m": float(rigid_metrics.target_xyz[0]),
                "target_y_m": float(rigid_metrics.target_xyz[1]),
                "target_z_m": float(rigid_metrics.target_xyz[2]),
                "actual_target_x_m": float(rigid_metrics.actual_target_xyz[0]),
                "actual_target_y_m": float(rigid_metrics.actual_target_xyz[1]),
                "actual_target_z_m": float(rigid_metrics.actual_target_xyz[2]),
                "target_error_m": float(rigid_metrics.target_error_m),
                "actual_ee_x_m": float(rigid_metrics.actual_ee_xyz[0]),
                "actual_ee_y_m": float(rigid_metrics.actual_ee_xyz[1]),
                "actual_ee_z_m": float(rigid_metrics.actual_ee_xyz[2]),
                "ee_error_m": float(rigid_metrics.ee_error_m),
                "robot_table_contact_frames": int(rigid_metrics.robot_table_contacts > 0),
                "rope_table_contact_frames": int(contact_any["table"] > 0),
                "rope_ground_contact_frames": int(contact_any["ground"] > 0),
                "left_finger_rope_contacts": int(contact_any["left"]),
                "right_finger_rope_contacts": int(contact_any["right"]),
                "left_pad_center_x_m": float(pad_metrics["left_pad_center_xyz"][0]),
                "left_pad_center_y_m": float(pad_metrics["left_pad_center_xyz"][1]),
                "left_pad_center_z_m": float(pad_metrics["left_pad_center_xyz"][2]),
                "right_pad_center_x_m": float(pad_metrics["right_pad_center_xyz"][0]),
                "right_pad_center_y_m": float(pad_metrics["right_pad_center_xyz"][1]),
                "right_pad_center_z_m": float(pad_metrics["right_pad_center_xyz"][2]),
                "leading_pad_center_x_m": float(pad_metrics["leading_pad_center_xyz"][0]),
                "leading_pad_center_y_m": float(pad_metrics["leading_pad_center_xyz"][1]),
                "leading_pad_center_z_m": float(pad_metrics["leading_pad_center_xyz"][2]),
                "left_pad_to_rope_min_distance_m": float(pad_metrics["left_pad_to_rope_min_distance_m"]),
                "right_pad_to_rope_min_distance_m": float(pad_metrics["right_pad_to_rope_min_distance_m"]),
                "leading_pad_to_rope_min_distance_m": float(pad_metrics["leading_pad_to_rope_min_distance_m"]),
                "rope_com_x_m": float(rope_com[0]),
                "rope_com_y_m": float(rope_com[1]),
                "rope_com_z_m": float(rope_com[2]),
                "rope_com_shift_x_m": float(rope_com_shift_x),
                "max_ground_penetration_m": float(penetration["max_ground_penetration_m"]),
                "max_table_penetration_m": float(penetration["max_table_penetration_m"]),
                "max_support_penetration_m": float(penetration["max_support_penetration_m"]),
                "p99_ground_penetration_m": float(penetration["p99_ground_penetration_m"]),
                "p99_table_penetration_m": float(penetration["p99_table_penetration_m"]),
                "p99_support_penetration_m": float(penetration["p99_support_penetration_m"]),
            }
        )

    def _render_current_frame(self, ffmpeg: subprocess.Popen[bytes], frame_idx: int) -> None:
        if ffmpeg.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")
        self.rigid.render()
        rope_points_wp = self.rope.rope_points_wp()
        rope_radii_wp = self.rope.rope_radii_wp()
        rope_starts_wp, rope_ends_wp = self.rope.rope_line_buffers()
        self.viewer.log_points("/demo/rope_points", rope_points_wp, rope_radii_wp, self.rope.rope_color, hidden=False)
        if rope_starts_wp is not None and rope_ends_wp is not None:
            self.viewer.log_lines(
                "/demo/rope_lines",
                rope_starts_wp,
                rope_ends_wp,
                (0.98, 0.78, 0.47),
                width=float(self.args.rope_line_width),
                hidden=False,
            )
        self.rigid.end_render()
        frame = self.viewer.get_frame(render_ui=False).numpy()
        ffmpeg.stdin.write(frame.tobytes())
        if frame_idx in self.preview_indices:
            self.preview_frames.append(frame.copy())
        if frame_idx < 30:
            self.first_30_frames.append(frame.copy())

    def _run_preroll(self) -> dict[str, np.ndarray]:
        pre_frames = max(0, int(round(self._effective_preroll_seconds() * float(self.args.video_fps))))
        pending_world_wrench = {"left": np.zeros(6, dtype=np.float32), "right": np.zeros(6, dtype=np.float32)}
        for _ in range(pre_frames):
            _, rope_metrics, rope_com, _, pending_world_wrench, _, _ = self._advance_one_frame(pending_world_wrench)
            if not np.all(np.isfinite(rope_com)) or float(rope_com[2]) > 0.5:
                break
        self.rigid.sim_time = 0.0
        return pending_world_wrench

    def evaluate_window(self, num_frames: int = 30, *, freeze_robot: bool = False) -> WindowMetrics:
        pending_world_wrench = self._run_preroll()
        rope_z_values: list[float] = []
        rope_table_frames = 0
        rope_ground_frames = 0
        robot_table_frames = 0
        max_ground_penetration = 0.0
        max_table_penetration = 0.0
        max_support_penetration = 0.0
        final_ground_p99 = 0.0
        final_table_p99 = 0.0
        final_support_p99 = 0.0
        for _ in range(int(num_frames)):
            if freeze_robot:
                mirror_state = self.rigid.mirror_body_state()
                rope_metrics = self.rope.step_frame(mirror_state)
                rope_com = rope_metrics["particle_q"].mean(axis=0)
                robot_table_contact = 0
            else:
                rigid_metrics, rope_metrics, rope_com, _, pending_world_wrench, _, _ = self._advance_one_frame(pending_world_wrench)
                robot_table_contact = int(rigid_metrics.robot_table_contacts > 0)
            rope_z = float(rope_com[2])
            rope_z_values.append(rope_z)
            contact_any = rope_metrics["contact_any"]
            rope_table_frames += int(contact_any["table"] > 0)
            rope_ground_frames += int(contact_any["ground"] > 0)
            robot_table_frames += int(robot_table_contact)
            penetration = self.rope.support_penetration_proxy(rope_metrics["particle_q"])
            max_ground_penetration = max(max_ground_penetration, float(penetration["max_ground_penetration_m"]))
            max_table_penetration = max(max_table_penetration, float(penetration["max_table_penetration_m"]))
            max_support_penetration = max(max_support_penetration, float(penetration["max_support_penetration_m"]))
            final_ground_p99 = float(penetration["p99_ground_penetration_m"])
            final_table_p99 = float(penetration["p99_table_penetration_m"])
            final_support_p99 = float(penetration["p99_support_penetration_m"])
            if not np.isfinite(rope_z) or rope_z > 0.35:
                break
        if not rope_z_values:
            rope_z_values = [np.inf]
        diffs = np.diff(np.asarray(rope_z_values, dtype=np.float32))
        max_abs_delta = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        return WindowMetrics(
            rope_table_contact_frames=int(rope_table_frames),
            rope_ground_contact_frames=int(rope_ground_frames),
            robot_table_contact_frames=int(robot_table_frames),
            max_rope_com_z=float(np.max(np.asarray(rope_z_values, dtype=np.float32))),
            max_abs_delta_rope_com_z=float(max_abs_delta),
            max_ground_penetration_m=float(max_ground_penetration),
            max_table_penetration_m=float(max_table_penetration),
            max_support_penetration_m=float(max_support_penetration),
            final_ground_penetration_p99_m=float(final_ground_p99),
            final_table_penetration_p99_m=float(final_table_p99),
            final_support_penetration_p99_m=float(final_support_p99),
        )

    def run(self) -> dict[str, Any]:
        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mp4_path = out_dir / "hero.mp4"
        gif_path = out_dir / "hero.gif"
        sheet_path = out_dir / "contact_sheet.jpg"
        ffmpeg = _ffmpeg_encode_mp4(self.args.width, self.args.height, self.args.video_fps, mp4_path)
        if ffmpeg.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")

        pending_world_wrench = self._run_preroll()
        try:
            for frame_idx in range(int(self.render_frame_count)):
                rigid_metrics, rope_metrics, rope_com, rope_com_shift_x, pending_world_wrench, _, pad_metrics = self._advance_one_frame(pending_world_wrench)
                penetration = self.rope.support_penetration_proxy(rope_metrics["particle_q"])
                self._record_metrics(frame_idx, rigid_metrics, rope_metrics, pad_metrics, rope_com, rope_com_shift_x, penetration)
                self._render_current_frame(ffmpeg, frame_idx)
        finally:
            ffmpeg.stdin.close()
            ffmpeg.wait()
            self.viewer.close()

        _make_gif(mp4_path, gif_path)
        _save_contact_sheet(self.preview_frames, sheet_path)
        first_30_sheet_path = out_dir / "first_30_frames_sheet.jpg"
        _save_contact_sheet(self.first_30_frames, first_30_sheet_path, cols=5)
        return self._finalize(out_dir, mp4_path, gif_path, sheet_path, first_30_sheet_path)

    def _finalize(self, out_dir: Path, mp4_path: Path, gif_path: Path, sheet_path: Path, first_30_sheet_path: Path) -> dict[str, Any]:
        arr = self.timeseries
        robot_table_contact_frames = int(sum(int(row["robot_table_contact_frames"]) for row in arr))
        rope_table_contact_frames = int(sum(int(row["rope_table_contact_frames"]) for row in arr))
        rope_ground_contact_frames = int(sum(int(row["rope_ground_contact_frames"]) for row in arr))
        first_30 = arr[: min(30, len(arr))]
        rope_table_contact_frames_first_30 = int(sum(int(row["rope_table_contact_frames"]) for row in first_30))
        rope_ground_contact_frames_first_30 = int(sum(int(row["rope_ground_contact_frames"]) for row in first_30))
        robot_table_contact_frames_first_30 = int(sum(int(row["robot_table_contact_frames"]) for row in first_30))
        rope_z_first_30 = np.asarray([row["rope_com_z_m"] for row in first_30], dtype=np.float32) if first_30 else np.zeros((0,), dtype=np.float32)
        max_rope_com_z_first_30 = float(np.max(rope_z_first_30)) if rope_z_first_30.size else 0.0
        max_abs_delta_rope_com_z_first_30 = (
            float(np.max(np.abs(np.diff(rope_z_first_30)))) if rope_z_first_30.size > 1 else 0.0
        )
        max_ground_penetration_m = float(max(float(row["max_ground_penetration_m"]) for row in arr)) if arr else 0.0
        max_table_penetration_m = float(max(float(row["max_table_penetration_m"]) for row in arr)) if arr else 0.0
        max_support_penetration_m = float(max(float(row["max_support_penetration_m"]) for row in arr)) if arr else 0.0
        final_ground_penetration_p99_m = float(arr[-1]["p99_ground_penetration_m"]) if arr else 0.0
        final_table_penetration_p99_m = float(arr[-1]["p99_table_penetration_m"]) if arr else 0.0
        final_support_penetration_p99_m = float(arr[-1]["p99_support_penetration_m"]) if arr else 0.0
        min_leading_pad_to_rope_distance_m = (
            float(min(float(row["leading_pad_to_rope_min_distance_m"]) for row in arr)) if arr else 0.0
        )
        rope_motion_after_contact = bool(
            self.first_finger_contact_frame is not None
            and self.first_rope_motion_frame is not None
            and self.first_rope_motion_frame >= self.first_finger_contact_frame
        )
        rope_render_matches_physics = bool(np.allclose(self.rope.render_radius, self.rope.physical_radius))
        leading_side_is_first = self.first_contact_side == self.leading_side if self.first_contact_side is not None else False
        net_rope_to_robot_wrench_nonzero = bool(
            self.first_contact_reaction_nonzero is not None
            or any(np.linalg.norm(v[:3]) > 1.0e-6 or np.linalg.norm(v[3:]) > 1.0e-6 for v in self.rope.last_world_wrench.values())
        )

        summary = {
            "demo": "robot_table_rope_split_mujoco_semiimplicit",
            "coupling_mode": str(self.args.coupling_mode),
            "rigid_frame_dt_s": float(self.rigid.frame_dt),
            "rigid_substep_dt_s": float(self.rigid.sim_dt),
            "rope_substep_dt_s": float(
                self.args.rope_sim_dt
                if self.args.rope_sim_dt is not None
                else (self.rigid.frame_dt / float(self.args.sim_substeps_rope))
            ),
            "scene_topology": str(self.args.scene_topology),
            "motion_pattern": str(self.args.motion_pattern),
            "rope_render_mode": str(self.args.rope_render_mode),
            "video_mode": str(self.args.video_mode),
            "record_start_mode": self._record_start_mode(),
            "rope_preroll_seconds": float(self._effective_preroll_seconds()),
            "finger_contact_set": str(self.args.finger_contact_set),
            "leading_side_expected": self.leading_side,
            "first_contact_side": self.first_contact_side,
            "leading_side_first_contact": bool(leading_side_is_first),
            "first_finger_rope_contact_frame": self.first_finger_contact_frame,
            "first_rope_motion_frame": self.first_rope_motion_frame,
            "first_leading_pad_proximity_frame": self.first_leading_pad_proximity_frame,
            "rope_motion_after_contact": rope_motion_after_contact,
            "robot_table_contact_frames": robot_table_contact_frames,
            "rope_table_contact_frames": rope_table_contact_frames,
            "rope_ground_contact_frames": rope_ground_contact_frames,
            "robot_table_contact_frames_first_30": robot_table_contact_frames_first_30,
            "rope_table_contact_frames_first_30": rope_table_contact_frames_first_30,
            "rope_ground_contact_frames_first_30": rope_ground_contact_frames_first_30,
            "rope_original_total_object_mass_kg": float(self.rope.rope_ir.get("original_total_object_mass", 0.0)),
            "rope_current_total_object_mass_kg": float(self.rope.rope_ir.get("current_total_object_mass", 0.0)),
            "rope_weight_scale": float(self.rope.rope_ir.get("weight_scale", 1.0)),
            "rope_mass_spring_scale": (
                None if self.args.mass_spring_scale is None else float(self.args.mass_spring_scale)
            ),
            "rope_object_mass_per_particle_kg": float(np.asarray(self.rope.rope_ir["mass"], dtype=np.float32)[: self.rope.n_obj].mean()),
            "particle_radius_scale": float(self.args.particle_radius_scale),
            "ground_shape_contact_scale": float(self.args.ground_shape_contact_scale),
            "ground_shape_contact_damping_multiplier": float(self.args.ground_shape_contact_damping_multiplier),
            "table_shape_contact_scale": float(self.args.table_shape_contact_scale),
            "table_shape_contact_damping_multiplier": float(self.args.table_shape_contact_damping_multiplier),
            "finger_shape_contact_scale": float(self.args.finger_shape_contact_scale),
            "finger_shape_contact_damping_multiplier": float(self.args.finger_shape_contact_damping_multiplier),
            "table_edge_inset_y": float(self.args.table_edge_inset_y),
            "overhang_drop_factor": float(self.args.overhang_drop_factor),
            "min_clearance_extra": float(self.args.min_clearance_extra),
            "presentation_table_edge_inset_y": float(self.args.presentation_table_edge_inset_y),
            "max_rope_com_z_first_30": float(max_rope_com_z_first_30),
            "max_abs_delta_rope_com_z_first_30": float(max_abs_delta_rope_com_z_first_30),
            "max_ground_penetration_m": float(max_ground_penetration_m),
            "max_table_penetration_m": float(max_table_penetration_m),
            "max_support_penetration_m": float(max_support_penetration_m),
            "final_ground_penetration_p99_m": float(final_ground_penetration_p99_m),
            "final_table_penetration_p99_m": float(final_table_penetration_p99_m),
            "final_support_penetration_p99_m": float(final_support_penetration_p99_m),
            "min_leading_pad_to_rope_distance_m": float(min_leading_pad_to_rope_distance_m),
            "presentation_contact_target_x_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[0])
            ),
            "presentation_contact_target_y_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[1])
            ),
            "presentation_contact_target_z_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[2])
            ),
            "rope_physical_radius_m": float(np.mean(self.rope.physical_radius)),
            "rope_render_radius_m": float(np.mean(self.rope.render_radius)),
            "rope_render_matches_physics": rope_render_matches_physics,
            "net_rope_to_robot_wrench_nonzero": bool(net_rope_to_robot_wrench_nonzero),
            "push_speed_mps": float(self.rigid.push_speed_mps),
            "push_path_length_m": float(self.rigid.push_path_length_m),
            "table_top_z_m": float(self.rigid.table_top_z),
            "render_frame_count": int(self.render_frame_count),
            "artifacts": {
                "scene_npz": str(out_dir / "scene.npz"),
                "timeseries_csv": str(out_dir / "timeseries.csv"),
                "hero_mp4": str(mp4_path),
                "hero_gif": str(gif_path),
                "contact_sheet": str(sheet_path),
                "first_30_frames_sheet": str(first_30_sheet_path),
            },
        }

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez(
            out_dir / "scene.npz",
            frame=np.asarray([row["frame"] for row in arr], dtype=np.int32),
            target_xyz=np.asarray([[row["target_x_m"], row["target_y_m"], row["target_z_m"]] for row in arr], dtype=np.float32),
            actual_target_xyz=np.asarray(
                [[row["actual_target_x_m"], row["actual_target_y_m"], row["actual_target_z_m"]] for row in arr], dtype=np.float32
            ),
            actual_ee_xyz=np.asarray([[row["actual_ee_x_m"], row["actual_ee_y_m"], row["actual_ee_z_m"]] for row in arr], dtype=np.float32),
            rope_com_xyz=np.asarray([[row["rope_com_x_m"], row["rope_com_y_m"], row["rope_com_z_m"]] for row in arr], dtype=np.float32),
            rope_physical_radius=self.rope.physical_radius.astype(np.float32, copy=False),
            rope_render_radius=self.rope.render_radius.astype(np.float32, copy=False),
        )
        with (out_dir / "timeseries.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(arr[0].keys()))
            writer.writeheader()
            writer.writerows(arr)
        (out_dir / "README.md").write_text(
            "\n".join(
                [
                    "# Robot Table Rope Split MuJoCo SemiImplicit",
                    "",
                    f"- Coupling mode: `{self.args.coupling_mode}`",
                    f"- Scene topology: `{self.args.scene_topology}`",
                    f"- Motion pattern: `{self.args.motion_pattern}`",
                    f"- Video mode: `{self.args.video_mode}`",
                    f"- Rope render mode: `{self.args.rope_render_mode}`",
                    f"- First finger contact frame: `{self.first_finger_contact_frame}`",
                    f"- First rope motion frame: `{self.first_rope_motion_frame}`",
                    f"- Rope motion after contact: `{rope_motion_after_contact}`",
                    f"- Min leading-pad distance to rope: `{min_leading_pad_to_rope_distance_m:.6f}`",
                    f"- Rope render matches physics: `{rope_render_matches_physics}`",
                    f"- Record start mode: `{self._record_start_mode()}`",
                    f"- Rope pre-roll seconds: `{self._effective_preroll_seconds():.3f}`",
                    f"- Presentation table-edge inset y (m): `{self.args.presentation_table_edge_inset_y:.3f}`",
                    "",
                    "Artifacts:",
                    "- `summary.json`",
                    "- `scene.npz`",
                    "- `timeseries.csv`",
                    "- `hero.mp4`",
                    "- `hero.gif`",
                    "- `contact_sheet.jpg`",
                    "- `first_30_frames_sheet.jpg`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return summary


def _clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def _ordered_unique_floats(*values: float) -> list[float]:
    ordered: list[float] = []
    seen: set[float] = set()
    for value in values:
        val = float(value)
        if val in seen:
            continue
        ordered.append(val)
        seen.add(val)
    return ordered


def _candidate_key(candidate: CalibrationCandidate) -> tuple[float, ...]:
    return (
        float(candidate.ground_shape_contact_scale),
        float(candidate.ground_shape_contact_damping_multiplier),
        float(candidate.table_shape_contact_scale),
        float(candidate.table_shape_contact_damping_multiplier),
        float(candidate.finger_shape_contact_scale),
        float(candidate.finger_shape_contact_damping_multiplier),
        float(candidate.table_edge_inset_y),
        float(candidate.overhang_drop_factor),
    )


def _dedupe_candidates(candidates: list[CalibrationCandidate]) -> list[CalibrationCandidate]:
    out: list[CalibrationCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        out.append(candidate)
        seen.add(key)
    return out


def _candidate_from_args(args: argparse.Namespace) -> CalibrationCandidate:
    return CalibrationCandidate(
        ground_shape_contact_scale=float(args.ground_shape_contact_scale),
        ground_shape_contact_damping_multiplier=float(args.ground_shape_contact_damping_multiplier),
        table_shape_contact_scale=float(args.table_shape_contact_scale),
        table_shape_contact_damping_multiplier=float(args.table_shape_contact_damping_multiplier),
        finger_shape_contact_scale=float(args.finger_shape_contact_scale),
        finger_shape_contact_damping_multiplier=float(args.finger_shape_contact_damping_multiplier),
        table_edge_inset_y=float(args.table_edge_inset_y),
        overhang_drop_factor=float(args.overhang_drop_factor),
    )


def _support_candidates(base_args: argparse.Namespace) -> list[CalibrationCandidate]:
    base = _candidate_from_args(base_args)
    candidates: list[CalibrationCandidate] = [base]

    ground_scales = _ordered_unique_floats(
        base.ground_shape_contact_scale,
        2.0e-5,
        5.0e-5,
        1.0e-4,
        2.0e-4,
        5.0e-4,
        1.0e-3,
    )
    ground_dampings = _ordered_unique_floats(
        base.ground_shape_contact_damping_multiplier,
        12.0,
        16.0,
        20.0,
    )

    for ground_scale in ground_scales:
        for ground_damping in ground_dampings:
            candidates.append(
                replace(
                    base,
                    ground_shape_contact_scale=ground_scale,
                    ground_shape_contact_damping_multiplier=ground_damping,
                )
            )

    for overhang_drop in _ordered_unique_floats(base.overhang_drop_factor, 0.10, 0.05):
        for ground_scale in ground_scales[1:5]:
            for ground_damping in ground_dampings[:3]:
                candidates.append(
                    replace(
                        base,
                        ground_shape_contact_scale=ground_scale,
                        ground_shape_contact_damping_multiplier=ground_damping,
                        overhang_drop_factor=overhang_drop,
                    )
                )

    for inset_y in _ordered_unique_floats(base.table_edge_inset_y, 0.18, 0.16):
        for overhang_drop in _ordered_unique_floats(base.overhang_drop_factor, 0.10):
            for ground_scale in ground_scales[1:4]:
                for ground_damping in ground_dampings[:2]:
                    candidates.append(
                        replace(
                            base,
                            ground_shape_contact_scale=ground_scale,
                            ground_shape_contact_damping_multiplier=ground_damping,
                            table_edge_inset_y=inset_y,
                            overhang_drop_factor=overhang_drop,
                        )
                    )

    return _dedupe_candidates(candidates)


def _finger_candidates(base: CalibrationCandidate) -> list[CalibrationCandidate]:
    candidates: list[CalibrationCandidate] = []
    for finger_scale in (0.10, 0.20, 0.35):
        for finger_damping in (1.5, 1.0):
            candidates.append(
                CalibrationCandidate(
                    ground_shape_contact_scale=base.ground_shape_contact_scale,
                    ground_shape_contact_damping_multiplier=base.ground_shape_contact_damping_multiplier,
                    table_shape_contact_scale=base.table_shape_contact_scale,
                    table_shape_contact_damping_multiplier=base.table_shape_contact_damping_multiplier,
                    finger_shape_contact_scale=finger_scale,
                    finger_shape_contact_damping_multiplier=finger_damping,
                    table_edge_inset_y=base.table_edge_inset_y,
                    overhang_drop_factor=base.overhang_drop_factor,
                )
            )
    return candidates


def _support_passes(metrics: WindowMetrics) -> bool:
    return (
        metrics.rope_table_contact_frames >= 20
        and metrics.rope_ground_contact_frames >= 20
        and metrics.max_rope_com_z <= 0.25
        and metrics.max_abs_delta_rope_com_z <= 0.10
        and metrics.max_support_penetration_m <= 0.003
        and metrics.final_support_penetration_p99_m <= 0.001
    )


def _support_precheck_passes(metrics: WindowMetrics) -> bool:
    return (
        metrics.rope_table_contact_frames >= 8
        and metrics.rope_ground_contact_frames >= 8
        and metrics.max_rope_com_z <= 0.25
        and metrics.max_abs_delta_rope_com_z <= 0.10
        and metrics.max_support_penetration_m <= 0.010
        and metrics.final_support_penetration_p99_m <= 0.005
    )


def _finger_passes(metrics: WindowMetrics) -> bool:
    return _support_passes(metrics) and metrics.robot_table_contact_frames == 0


def _candidate_score(metrics: WindowMetrics) -> tuple[float, float, float, float, float, float]:
    return (
        float(metrics.max_support_penetration_m),
        float(metrics.final_support_penetration_p99_m),
        -float(metrics.rope_table_contact_frames + metrics.rope_ground_contact_frames),
        float(metrics.robot_table_contact_frames),
        float(metrics.max_rope_com_z),
        float(metrics.max_abs_delta_rope_com_z),
    )


def _evaluate_candidate(
    base_args: argparse.Namespace,
    candidate: CalibrationCandidate,
    *,
    freeze_robot: bool,
    sim_substeps_rope: int | None = None,
    num_frames: int = 30,
) -> WindowMetrics:
    sim_substeps = int(base_args.sim_substeps_rope if sim_substeps_rope is None else sim_substeps_rope)
    candidate_args = _clone_args(
        base_args,
        width=64,
        height=64,
        rope_sim_dt=None,
        sim_substeps_rope=sim_substeps,
        ground_shape_contact_scale=candidate.ground_shape_contact_scale,
        ground_shape_contact_damping_multiplier=candidate.ground_shape_contact_damping_multiplier,
        table_shape_contact_scale=candidate.table_shape_contact_scale,
        table_shape_contact_damping_multiplier=candidate.table_shape_contact_damping_multiplier,
        finger_shape_contact_scale=candidate.finger_shape_contact_scale,
        finger_shape_contact_damping_multiplier=candidate.finger_shape_contact_damping_multiplier,
        table_edge_inset_y=candidate.table_edge_inset_y,
        overhang_drop_factor=candidate.overhang_drop_factor,
    )
    demo = SplitDemo(candidate_args)
    try:
        return demo.evaluate_window(num_frames=int(num_frames), freeze_robot=freeze_robot)
    finally:
        try:
            demo.viewer.close()
        except Exception:
            pass


def _select_support_args(base_args: argparse.Namespace) -> argparse.Namespace:
    best_support: tuple[CalibrationCandidate, WindowMetrics] | None = None
    selected_support: CalibrationCandidate | None = None
    for candidate in _support_candidates(base_args):
        precheck = _evaluate_candidate(
            base_args,
            candidate,
            freeze_robot=False,
            sim_substeps_rope=int(base_args.sim_substeps_rope),
            num_frames=10,
        )
        print(
            "[support-calibration-precheck]",
            json.dumps(
                {
                    "candidate": candidate.__dict__,
                    "metrics": precheck.__dict__,
                    "sim_substeps_rope": int(base_args.sim_substeps_rope),
                }
            ),
            flush=True,
        )
        if not _support_precheck_passes(precheck):
            if best_support is None or _candidate_score(precheck) < _candidate_score(best_support[1]):
                best_support = (candidate, precheck)
            continue

        metrics = _evaluate_candidate(
            base_args,
            candidate,
            freeze_robot=False,
            sim_substeps_rope=int(base_args.sim_substeps_rope),
            num_frames=30,
        )
        print(
            "[support-calibration]",
            json.dumps(
                {
                    "candidate": candidate.__dict__,
                    "metrics": metrics.__dict__,
                    "sim_substeps_rope": int(base_args.sim_substeps_rope),
                }
            ),
            flush=True,
        )
        if best_support is None or _candidate_score(metrics) < _candidate_score(best_support[1]):
            best_support = (candidate, metrics)
        if _support_passes(metrics):
            selected_support = candidate
            break
    if selected_support is None or best_support is None:
        raise RuntimeError(
            "Light-rope support calibration failed. Best candidate: "
            + json.dumps(
                {
                    "candidate": best_support[0].__dict__,
                    "metrics": best_support[1].__dict__,
                },
                indent=2,
            )
        )

    return _clone_args(
        base_args,
        ground_shape_contact_scale=selected_support.ground_shape_contact_scale,
        ground_shape_contact_damping_multiplier=selected_support.ground_shape_contact_damping_multiplier,
        table_shape_contact_scale=selected_support.table_shape_contact_scale,
        table_shape_contact_damping_multiplier=selected_support.table_shape_contact_damping_multiplier,
        table_edge_inset_y=selected_support.table_edge_inset_y,
        overhang_drop_factor=selected_support.overhang_drop_factor,
    )


def _select_finger_args(base_args: argparse.Namespace) -> argparse.Namespace:
    support_args = _select_support_args(base_args)
    selected_support = _candidate_from_args(support_args)

    best_finger: tuple[CalibrationCandidate, WindowMetrics] | None = None
    selected_candidate: CalibrationCandidate | None = None
    for candidate in _finger_candidates(selected_support):
        metrics = _evaluate_candidate(
            support_args,
            candidate,
            freeze_robot=False,
            sim_substeps_rope=int(support_args.calibration_sim_substeps_rope),
        )
        print(
            "[finger-calibration]",
            json.dumps({"candidate": candidate.__dict__, "metrics": metrics.__dict__}),
            flush=True,
        )
        if best_finger is None or _candidate_score(metrics) < _candidate_score(best_finger[1]):
            best_finger = (candidate, metrics)
        if _finger_passes(metrics):
            selected_candidate = candidate
            break
    if selected_candidate is None or best_finger is None:
        raise RuntimeError(
            "Light-rope finger calibration failed. Best candidate: "
            + json.dumps(
                {
                    "candidate": best_finger[0].__dict__,
                    "metrics": best_finger[1].__dict__,
                },
                indent=2,
            )
        )

    return _clone_args(
        support_args,
        ground_shape_contact_scale=selected_candidate.ground_shape_contact_scale,
        ground_shape_contact_damping_multiplier=selected_candidate.ground_shape_contact_damping_multiplier,
        table_shape_contact_scale=selected_candidate.table_shape_contact_scale,
        table_shape_contact_damping_multiplier=selected_candidate.table_shape_contact_damping_multiplier,
        finger_shape_contact_scale=selected_candidate.finger_shape_contact_scale,
        finger_shape_contact_damping_multiplier=selected_candidate.finger_shape_contact_damping_multiplier,
        table_edge_inset_y=selected_candidate.table_edge_inset_y,
        overhang_drop_factor=selected_candidate.overhang_drop_factor,
    )


def _select_calibrated_args(base_args: argparse.Namespace) -> argparse.Namespace:
    return _select_finger_args(base_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-finger split demo: MuJoCo rigid robot/table + SemiImplicit rope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--coupling-mode", choices=["one_way", "two_way"], default="one_way")
    parser.add_argument("--scene-topology", choices=["table_edge_drape"], default="table_edge_drape")
    parser.add_argument("--motion-pattern", choices=["side_finger_push"], default="side_finger_push")
    parser.add_argument("--rope-render-mode", choices=["physical_only"], default="physical_only")
    parser.add_argument("--video-mode", choices=["support_default", "presentation_lifted"], default="support_default")
    parser.add_argument("--finger-contact-set", choices=["fingers_plus_pads"], default="fingers_plus_pads")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=170)
    parser.add_argument("--sim-substeps-rigid", type=int, default=10)
    parser.add_argument("--sim-substeps-rope", type=int, default=667)
    parser.add_argument("--calibration-sim-substeps-rope", type=int, default=32)
    parser.add_argument("--rope-sim-dt", type=float, default=None)
    parser.add_argument("--rope-preroll-seconds", type=float, default=2.0)
    parser.add_argument("--presentation-lift-height", type=float, default=0.08)
    parser.add_argument("--presentation-table-edge-inset-y", type=float, default=0.22)
    parser.add_argument("--presentation-overhang-drop-factor", type=float, default=0.0)
    parser.add_argument("--presentation-opening-seconds", type=float, default=0.1)
    parser.add_argument("--presentation-approach-seconds", type=float, default=0.2)
    parser.add_argument("--presentation-retract-seconds", type=float, default=0.6)
    parser.add_argument("--presentation-push-distance", type=float, default=0.05)
    parser.add_argument("--presentation-push-speed-mps", type=float, default=0.02)
    parser.add_argument("--presentation-tail-frames", type=int, default=15)
    parser.add_argument("--presentation-proximity-threshold", type=float, default=0.015)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gravity-mag", type=float, default=9.8)
    parser.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drag-damping-scale", type=float, default=1.0)
    parser.add_argument("--auto-set-weight", type=float, default=0.1)
    parser.add_argument("--mass-spring-scale", type=float, default=None)
    parser.add_argument("--object-mass", type=float, default=1.0)
    parser.add_argument("--spring-ke-scale", type=float, default=1.0)
    parser.add_argument("--spring-kd-scale", type=float, default=1.0)
    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument("--friction-smoothing", type=float, default=1.0)
    parser.add_argument("--particle-radius-scale", type=float, default=0.2)
    parser.add_argument("--ground-shape-contact-scale", type=float, default=1.0e-3)
    parser.add_argument("--ground-shape-contact-damping-multiplier", type=float, default=64.0)
    parser.add_argument("--table-shape-contact-scale", type=float, default=1.0e-3)
    parser.add_argument("--table-shape-contact-damping-multiplier", type=float, default=64.0)
    parser.add_argument("--finger-shape-contact-scale", type=float, default=0.1)
    parser.add_argument("--finger-shape-contact-damping-multiplier", type=float, default=1.5)
    parser.add_argument("--table-edge-inset-y", type=float, default=0.17)
    parser.add_argument("--overhang-drop-factor", type=float, default=0.02)
    parser.add_argument("--min-clearance-extra", type=float, default=0.002)
    parser.add_argument("--auto-calibrate-support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-calibrate-light-rope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-opening", type=float, default=0.04)
    parser.add_argument("--push-clearance", type=float, default=0.029)
    parser.add_argument("--gripper-yaw", type=float, default=1.32079632679)
    parser.add_argument("--rope-line-width", type=float, default=0.008)
    parser.add_argument("--rope-spring-stride", type=int, default=4)
    parser.add_argument("--rope-motion-threshold", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.auto_calibrate_support:
        args = _select_support_args(args)
    if args.auto_calibrate_light_rope:
        args = _select_calibrated_args(args)
    demo = SplitDemo(args)
    summary = demo.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
