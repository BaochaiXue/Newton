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


def _inverse_transform_point_np(tf_xyzw: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    pos = np.asarray(tf_xyzw[:3], dtype=np.float32)
    quat = np.asarray(tf_xyzw[3:7], dtype=np.float32)
    rot = _quat_xyzw_to_rotmat(quat)
    return rot.T @ (np.asarray(p_world, dtype=np.float32) - pos)


def _transform_wrench_body_to_world(tf_xyzw: np.ndarray, wrench_body: np.ndarray) -> np.ndarray:
    quat = np.asarray(tf_xyzw[3:7], dtype=np.float32)
    f_world = _quat_rotate_np(quat, wrench_body[:3])
    tau_world = _quat_rotate_np(quat, wrench_body[3:6])
    return np.concatenate([f_world, tau_world], axis=0).astype(np.float32, copy=False)


def _parse_contact_count(count_array: wp.array) -> int:
    return int(np.asarray(count_array.numpy()).reshape(-1)[0])


def _is_presentation_mode(args: argparse.Namespace) -> bool:
    return str(args.video_mode).startswith("presentation_")


def _is_pick_place_mode(args: argparse.Namespace) -> bool:
    return str(args.motion_pattern) == "grasp_lift_place"


def _uses_rope_cradle_geometry(args: argparse.Namespace) -> bool:
    return _is_pick_place_mode(args) and str(args.presentation_gripper_geometry) == "rope_cradle"


def _uses_aux_panda_pad_geometry(args: argparse.Namespace) -> bool:
    return str(args.presentation_gripper_geometry) == "panda_pads"


def _effective_table_shape_contact_scale(args: argparse.Namespace) -> float:
    if _is_pick_place_mode(args):
        return float(args.presentation_table_shape_contact_scale)
    return float(args.table_shape_contact_scale)


def _effective_table_shape_contact_damping_multiplier(args: argparse.Namespace) -> float:
    if _is_pick_place_mode(args):
        return float(args.presentation_table_shape_contact_damping_multiplier)
    return float(args.table_shape_contact_damping_multiplier)


def _lerp_np(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * np.asarray(a, dtype=np.float32) + float(alpha) * np.asarray(b, dtype=np.float32)


@wp.kernel
def _apply_grasp_assist_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_indices: wp.array(dtype=wp.int32),
    particle_offsets: wp.array(dtype=wp.vec3),
    target_center: wp.vec3,
    ke: float,
    kd: float,
    max_force: float,
    strength: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = particle_indices[tid]
    desired = target_center + particle_offsets[tid]
    delta = desired - particle_q[particle_idx]
    force = strength * (ke * delta - kd * particle_qd[particle_idx])
    force_norm = wp.length(force)
    if force_norm > max_force:
        force = force * (max_force / (force_norm + 1.0e-8))
    wp.atomic_add(particle_f, particle_idx, force)


@dataclass
class FingerShapeSpec:
    body_name: str
    shape_type: Any
    mesh: Any | None
    local_xform: Any
    scale: tuple[float, float, float]
    label: str
    is_pad: bool


@dataclass
class RigidStepMetrics:
    frame: int
    time_s: float
    motion_phase: str
    motion_phase_alpha: float
    target_xyz: np.ndarray
    actual_target_xyz: np.ndarray
    target_error_m: float
    actual_ee_xyz: np.ndarray
    ee_error_m: float
    gripper_opening_m: float
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
    support_scale = 0.5 * (float(args.ground_shape_contact_scale) + _effective_table_shape_contact_scale(args))
    support_damping = 0.5 * (
        float(args.ground_shape_contact_damping_multiplier) + _effective_table_shape_contact_damping_multiplier(args)
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
                scale = _effective_table_shape_contact_scale(args)
                damping_mult = _effective_table_shape_contact_damping_multiplier(args)
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
    mu_vals = model.shape_material_mu.numpy().astype(np.float32, copy=False).copy()
    for idx, label in enumerate(labels):
        if "mirror_" in label:
            mu_vals[idx] *= np.float32(float(args.finger_shape_friction_multiplier))
    model.shape_material_mu.assign(mu_vals)


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
        self.grasp_center_local_offset = np.zeros(3, dtype=np.float32)
        self.pick_place_phase_frames: dict[str, int] = {}
        self.pick_place_targets: dict[str, np.ndarray] = {}
        self.gripper_opening_open = 0.04
        self.gripper_opening_closed = 0.006
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
        self.presentation_gripper_geometry = str(self.args.presentation_gripper_geometry)
        self.gripper_geometry_shape_indices: list[int] = []
        self.contact_shape_specs: list[FingerShapeSpec] = []
        self.pad_center_locals: dict[str, np.ndarray] = {}
        self.grasp_contact_locals: dict[str, np.ndarray] = {}

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
        builder.joint_target_ke[:7] = [float(self.args.arm_joint_target_ke)] * 7
        builder.joint_target_kd[:7] = [float(self.args.arm_joint_target_kd)] * 7
        builder.joint_target_ke[7:9] = [float(self.args.finger_joint_target_ke)] * 2
        builder.joint_target_kd[7:9] = [float(self.args.finger_joint_target_kd)] * 2
        builder.joint_effort_limit[:7] = [float(self.args.arm_joint_effort_limit)] * 7
        builder.joint_effort_limit[7:9] = [float(self.args.finger_joint_effort_limit)] * 2
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        self.pad_xform = wp.transform(
            wp.vec3(
                float(self.args.presentation_panda_finger_grasp_local_x),
                float(self.args.presentation_panda_finger_grasp_local_y),
                float(self.args.presentation_panda_finger_grasp_local_z),
            ),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )
        if _uses_aux_panda_pad_geometry(self.args):
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
        if _uses_rope_cradle_geometry(self.args):
            self._add_rope_cradle_geometry(builder, shape_cfg_primitives, sdf_max_resolution, sdf_narrow_band_range)

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
                shape_type = builder.shape_type[shape_idx]
                label = str(builder.shape_label[shape_idx])
                is_cradle_primitive = shape_type == newton.GeoType.BOX and "rope_cradle" in label
                if shape_type != newton.GeoType.MESH and not is_cradle_primitive:
                    continue
                mesh = builder.shape_source[shape_idx]
                if shape_type == newton.GeoType.MESH and mesh is None:
                    continue
                is_pad = "pad" in label
                local_xform = builder.shape_transform[shape_idx]
                scale = tuple(float(v) for v in builder.shape_scale[shape_idx])
                self.contact_shape_specs.append(
                    FingerShapeSpec(
                        body_name=body_name,
                        shape_type=shape_type,
                        mesh=mesh,
                        local_xform=local_xform,
                        scale=scale,
                        label=label,
                        is_pad=is_pad,
                    )
                )
                if is_pad:
                    self.pad_center_locals[body_name] = np.asarray(local_xform[:3], dtype=np.float32)
        if not self.pad_center_locals:
            native_tip_local = np.asarray(self.pad_xform[:3], dtype=np.float32)
            self.pad_center_locals = {
                "left": native_tip_local.copy(),
                "right": native_tip_local.copy(),
            }
        if not self.grasp_contact_locals:
            self.grasp_contact_locals = {
                side: self.pad_center_locals[side].copy()
                for side in ("left", "right")
                if side in self.pad_center_locals
            }

        self.model_single = copy.deepcopy(builder).finalize()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        sdf_hydroelastic_config = HydroelasticSDF.Config(
            buffer_mult_iso=int(self.args.rigid_hydro_buffer_mult_iso),
            buffer_mult_contact=int(self.args.rigid_hydro_buffer_mult_contact),
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
            njmax=int(self.args.rigid_njmax),
            nconmax=int(self.args.rigid_nconmax),
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )
        self.control = self.model.control()
        self.viewer.set_model(self.model)

    def _add_rope_cradle_geometry(
        self,
        builder: Any,
        shape_cfg_meshes: Any,
        sdf_max_resolution: int,
        sdf_narrow_band_range: tuple[float, float],
    ) -> None:
        """Add visible, physical finger extensions that can support a rope segment.

        The Panda finger meshes are mostly flat pads; they contact and push the
        rope but do not create a lower support surface.  The cradle is explicit
        gripper hardware: two small L-shaped mesh additions, mirrored into the
        rope model as ordinary collision shapes.
        """

        lip_hx = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_lip_length))
        lip_hy = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_lip_depth))
        lip_hz = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_lip_thickness))
        wall_hx = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_lip_length))
        wall_hy = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_wall_thickness))
        wall_hz = max(1.0e-4, 0.5 * float(self.args.presentation_cradle_wall_height))

        lip_local_y = -0.5 * float(self.args.presentation_cradle_lip_depth)
        lip_local_z = float(self.args.presentation_cradle_lip_local_z)
        wall_local_y = -float(self.args.presentation_cradle_lip_depth) - 0.5 * float(
            self.args.presentation_cradle_wall_thickness
        )
        wall_local_z = lip_local_z - lip_hz + wall_hz
        bite_local = np.array([0.0, lip_local_y, lip_local_z - lip_hz], dtype=np.float32)

        cradle_color = (0.05, 0.68, 0.62)
        for side, body_idx in (("left", self.left_finger_body_idx), ("right", self.right_finger_body_idx)):
            lip_shape = builder.add_shape_box(
                body=body_idx,
                xform=wp.transform(wp.vec3(0.0, float(lip_local_y), float(lip_local_z)), wp.quat_identity()),
                hx=lip_hx,
                hy=lip_hy,
                hz=lip_hz,
                cfg=shape_cfg_meshes,
                color=cradle_color,
                label=f"{side}_rope_cradle_lip",
            )
            wall_shape = builder.add_shape_box(
                body=body_idx,
                xform=wp.transform(wp.vec3(0.0, float(wall_local_y), float(wall_local_z)), wp.quat_identity()),
                hx=wall_hx,
                hy=wall_hy,
                hz=wall_hz,
                cfg=shape_cfg_meshes,
                color=cradle_color,
                label=f"{side}_rope_cradle_wall",
            )
            self.finger_shape_indices.extend([lip_shape, wall_shape])
            self.gripper_geometry_shape_indices.extend([lip_shape, wall_shape])
            self.grasp_contact_locals[side] = bite_local.copy()
        self.presentation_gripper_geometry = "rope_cradle"

    def _setup_motion_defaults(self) -> None:
        if _is_presentation_mode(self.args):
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
        self.gripper_opening_open = float(self.args.presentation_grasp_opening)
        self.gripper_opening_closed = float(self.args.presentation_grasp_closed_opening)
        if _is_pick_place_mode(self.args):
            self._setup_pick_place_timing()

    def _setup_pick_place_timing(self) -> None:
        self.pick_place_phase_frames = {
            "opening": max(1, int(round(float(self.args.presentation_opening_seconds) / self.frame_dt))),
            "approach": max(1, int(round(float(self.args.presentation_approach_seconds) / self.frame_dt))),
            "lower": max(1, int(round(float(self.args.presentation_grasp_lower_seconds) / self.frame_dt))),
            "close": max(1, int(round(float(self.args.presentation_grasp_close_seconds) / self.frame_dt))),
            "lift": max(1, int(round(float(self.args.presentation_lift_seconds) / self.frame_dt))),
            "transfer": max(1, int(round(float(self.args.presentation_transfer_seconds) / self.frame_dt))),
            "place_lower": max(1, int(round(float(self.args.presentation_place_lower_seconds) / self.frame_dt))),
            "release": max(1, int(round(float(self.args.presentation_release_seconds) / self.frame_dt))),
            "retract": max(1, int(round(float(self.args.presentation_retract_seconds) / self.frame_dt))),
        }
        self.settle_frames = self.pick_place_phase_frames["opening"]
        self.approach_frames = self.pick_place_phase_frames["approach"]
        self.retract_frames = self.pick_place_phase_frames["retract"]
        self.push_frames = (
            self.pick_place_phase_frames["lower"]
            + self.pick_place_phase_frames["close"]
            + self.pick_place_phase_frames["lift"]
            + self.pick_place_phase_frames["transfer"]
            + self.pick_place_phase_frames["place_lower"]
            + self.pick_place_phase_frames["release"]
        )
        self.total_frames = int(sum(self.pick_place_phase_frames.values()))

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
        if _is_pick_place_mode(self.args):
            self._configure_pick_place_for_rope(points, rope_radius_m)
            return

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

    def _configure_grasp_center_target(self) -> None:
        body_q_np = self.state_single.body_q.numpy()
        left_local = self.grasp_contact_locals.get("left", self.pad_center_locals["left"])
        right_local = self.grasp_contact_locals.get("right", self.pad_center_locals["right"])
        left_center = _transform_point_np(body_q_np[self.left_finger_body_idx], left_local)
        right_center = _transform_point_np(body_q_np[self.right_finger_body_idx], right_local)
        center_world = 0.5 * (left_center + right_center)
        self.grasp_center_local_offset = _inverse_transform_point_np(body_q_np[self.ee_index], center_world).astype(
            np.float32,
            copy=False,
        )
        self._configure_position_target_ik(self.ee_index, self.grasp_center_local_offset)

    def _configure_pick_place_for_rope(self, points: np.ndarray, rope_radius_m: float) -> None:
        table_min = self.table_pos - self.table_half_extents
        table_max = self.table_pos + self.table_half_extents
        table_edge_y = float(table_min[1])
        table_mask = (
            (points[:, 0] >= table_min[0] - 0.01)
            & (points[:, 0] <= table_max[0] + 0.01)
            & (points[:, 1] >= table_min[1] - 0.01)
            & (points[:, 1] <= table_max[1] + 0.01)
        )
        edge_band = np.abs(points[:, 1] - table_edge_y) <= max(0.025, float(rope_radius_m) * 4.0)
        grasp_cloud = points[edge_band] if np.any(edge_band) else (points[table_mask] if np.any(table_mask) else points)
        grasp_x = float(np.median(grasp_cloud[:, 0]) + float(self.args.presentation_grasp_x_offset))
        # A real gripper cannot close around a rope segment that is fully backed by
        # the table. The presentation grasp targets the table-edge segment so the
        # lower jaw has free space outside the support.
        grasp_y = float(
            table_edge_y
            - float(self.args.presentation_edge_grasp_outset_y)
            + float(self.args.presentation_grasp_y_offset)
        )
        grasp_z_clearance = (
            float(self.args.presentation_cradle_grasp_z_clearance)
            if _uses_rope_cradle_geometry(self.args)
            else float(self.args.presentation_grasp_z_clearance)
        )
        grasp_z = float(
            self.table_top_z
            + float(rope_radius_m)
            + grasp_z_clearance
            + float(self.args.presentation_grasp_z_offset)
        )
        grasp_center = np.array([grasp_x, grasp_y, grasp_z], dtype=np.float32)

        carry_z = float(self.table_top_z + float(self.args.presentation_carry_height))
        ground_place_y = float(table_edge_y - abs(float(self.args.presentation_place_y_offset)))
        place_center = np.array(
            [
                float(grasp_center[0] + float(self.args.presentation_place_x_offset)),
                ground_place_y,
                float(float(rope_radius_m) + float(self.args.presentation_place_ground_clearance)),
            ],
            dtype=np.float32,
        )

        pregrasp = grasp_center + np.array(
            [
                0.0,
                -abs(float(self.args.presentation_pregrasp_y_standoff)),
                float(self.args.presentation_pregrasp_z_offset),
            ],
            dtype=np.float32,
        )
        approach_offset = np.array(
            [
                -abs(float(self.args.presentation_approach_x_offset)),
                -abs(float(self.args.presentation_approach_y_offset)),
                float(self.args.presentation_approach_z_offset),
            ],
            dtype=np.float32,
        )
        above_grasp = pregrasp
        far = pregrasp + approach_offset
        lifted = np.array([grasp_center[0], grasp_center[1], carry_z], dtype=np.float32)
        carry = np.array([place_center[0], place_center[1], carry_z], dtype=np.float32)
        # After release, retreat from the placement pose.  First clear vertically
        # before moving back across the table edge; a single low diagonal retract
        # can make the open cradle sweep the table and re-hook the rope.
        retract_clear = place_center + np.array(
            [
                0.0,
                0.0,
                float(self.args.presentation_retract_z_offset),
            ],
            dtype=np.float32,
        )
        retract = place_center + np.array(
            [
                -abs(float(self.args.presentation_retract_x_offset)),
                float(self.args.presentation_retract_y_offset),
                float(self.args.presentation_retract_z_offset),
            ],
            dtype=np.float32,
        )

        self.pick_place_targets = {
            "far": far,
            "above_grasp": above_grasp,
            "grasp": grasp_center,
            "lifted": lifted,
            "carry": carry,
            "place": place_center,
            "retract_clear": retract_clear,
            "retract": retract,
        }
        self.presentation_contact_point = grasp_center.copy()
        self.presentation_scene_center = 0.5 * (grasp_center + place_center)
        self.pad_far_xyz = far.copy()
        self.pad_push_start_xyz = grasp_center.copy()
        self.pad_push_end_xyz = place_center.copy()
        self.prepush_z = float(grasp_center[2])
        self.retract_z = float(retract[2])
        self.push_path_length_m = float(np.linalg.norm(place_center - grasp_center))
        self.push_speed_mps = float(self.args.presentation_push_speed_mps)
        self._setup_pick_place_timing()
        self._configure_grasp_center_target()
        self._snap_to_target_position(far)
        self._set_camera()

    def _solve_single_fk(
        self, ee_target_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.pos_obj.set_target_positions(wp.array([ee_target_xyz], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        q_np = self.joint_q_ik.numpy().astype(np.float32, copy=False).reshape(-1)
        if q_np.shape[0] >= 9:
            q_np[7:9] = float(self.current_gripper_opening())
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
        if _is_presentation_mode(self.args) and not _is_pick_place_mode(self.args):
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
        if q_np.shape[0] >= 9:
            q_np[7:9] = float(self.current_gripper_opening())
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
        self.control.joint_target_pos[7:9].fill_(float(self.current_gripper_opening()))
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
        if _is_presentation_mode(self.args) and self.presentation_contact_point is not None:
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
            self.viewer.camera.fov = 82.0 if _is_presentation_mode(self.args) else 68.0

    def _pick_place_phase(self) -> tuple[str, float]:
        if not self.pick_place_phase_frames:
            return "opening", 0.0
        cursor = 0
        for name, count in self.pick_place_phase_frames.items():
            end = cursor + int(count)
            if self.frame_idx < end:
                alpha = (self.frame_idx - cursor) / max(int(count) - 1, 1)
                return name, float(np.clip(alpha, 0.0, 1.0))
            cursor = end
        return "tail", 1.0

    def current_motion_phase(self) -> tuple[str, float]:
        if _is_pick_place_mode(self.args):
            return self._pick_place_phase()
        if self.frame_idx < self.settle_frames:
            return "opening", 0.0
        if self.frame_idx < self.settle_frames + self.approach_frames:
            alpha = (self.frame_idx - self.settle_frames) / max(self.approach_frames - 1, 1)
            return "approach", float(np.clip(alpha, 0.0, 1.0))
        if self.frame_idx < self.settle_frames + self.approach_frames + self.push_frames:
            alpha = (self.frame_idx - self.settle_frames - self.approach_frames) / max(self.push_frames - 1, 1)
            return "push", float(np.clip(alpha, 0.0, 1.0))
        alpha = (self.frame_idx - self.settle_frames - self.approach_frames - self.push_frames) / max(
            self.retract_frames - 1,
            1,
        )
        return "retract", float(np.clip(alpha, 0.0, 1.0))

    def _current_pick_place_target(self) -> np.ndarray:
        if not self.pick_place_targets:
            return self.pad_far_xyz.copy()
        phase, alpha = self._pick_place_phase()
        targets = self.pick_place_targets
        if phase == "opening":
            return targets["far"].copy()
        if phase == "approach":
            return _lerp_np(targets["far"], targets["above_grasp"], alpha)
        if phase == "lower":
            return _lerp_np(targets["above_grasp"], targets["grasp"], alpha)
        if phase == "close":
            return targets["grasp"].copy()
        if phase == "lift":
            return _lerp_np(targets["grasp"], targets["lifted"], alpha)
        if phase == "transfer":
            return _lerp_np(targets["lifted"], targets["carry"], alpha)
        if phase == "place_lower":
            return _lerp_np(targets["carry"], targets["place"], alpha)
        if phase == "release":
            return targets["place"].copy()
        if phase == "retract":
            if "retract_clear" in targets and float(targets["retract_clear"][2]) > float(targets["place"][2]):
                if alpha < 0.5:
                    return _lerp_np(targets["place"], targets["retract_clear"], alpha * 2.0)
                return _lerp_np(targets["retract_clear"], targets["retract"], (alpha - 0.5) * 2.0)
            return _lerp_np(targets["place"], targets["retract"], alpha)
        return targets["retract"].copy()

    def current_gripper_opening(self) -> float:
        if not _is_pick_place_mode(self.args):
            return float(self.gripper_opening)
        phase, alpha = self._pick_place_phase()
        open_q = float(self.gripper_opening_open)
        closed_q = float(self.gripper_opening_closed)
        if phase in {"opening", "approach", "lower", "retract", "tail"}:
            return open_q
        if phase == "close":
            return float((1.0 - alpha) * open_q + alpha * closed_q)
        if phase in {"lift", "transfer", "place_lower"}:
            return closed_q
        if phase == "release":
            return float((1.0 - alpha) * closed_q + alpha * open_q)
        return open_q

    def current_target_position(self) -> np.ndarray:
        if _is_pick_place_mode(self.args):
            return self._current_pick_place_target()
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
        motion_phase, motion_phase_alpha = self.current_motion_phase()
        target_position = self.current_target_position()
        gripper_opening = float(self.current_gripper_opening())
        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        self.control.joint_target_pos.zero_()
        wp.copy(self.control.joint_target_pos, self.joint_q_ik.flatten())
        self.control.joint_target_pos[7:9].fill_(gripper_opening)

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
            motion_phase=motion_phase,
            motion_phase_alpha=float(motion_phase_alpha),
            target_xyz=target_position,
            actual_target_xyz=actual_target,
            target_error_m=target_error,
            actual_ee_xyz=actual_ee,
            ee_error_m=ee_error,
            gripper_opening_m=gripper_opening,
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

    def grasp_contact_points_world(self) -> dict[str, np.ndarray]:
        body_q = self.state_0.body_q.numpy().astype(np.float32, copy=False)
        return {
            side: _transform_point_np(body_q[body_idx], self.grasp_contact_locals.get(side, self.pad_center_locals[side]))
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
        self.grasp_assist_indices: np.ndarray | None = None
        self.grasp_assist_offsets: np.ndarray | None = None
        self.grasp_assist_indices_wp: wp.array | None = None
        self.grasp_assist_offsets_wp: wp.array | None = None
        self.grasp_metric_indices: np.ndarray | None = None

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
                    if _is_presentation_mode(self.args)
                    else float(self.rigid.prepush_x + 0.05)
                ),
                self.rigid.table_pos[1]
                - self.rigid.table_half_extents[1]
                + (
                    float(self.args.presentation_table_edge_inset_y)
                    if _is_presentation_mode(self.args)
                    else float(self.args.table_edge_inset_y)
                ),
                center_z_for_table_clearance,
            ],
            dtype=np.float32,
        )
        if str(self.args.video_mode) == "presentation_lifted" and not _is_pick_place_mode(self.args):
            target_center[2] += float(self.args.presentation_lift_height)
        shift = target_center - center
        shifted = (x0 + shift).astype(np.float32, copy=False)
        table_edge_y = float(self.rigid.table_pos[1] - self.rigid.table_half_extents[1])
        overhang_mask = shifted[:, 1] < table_edge_y
        if np.any(overhang_mask):
            hang = (table_edge_y - shifted[overhang_mask, 1]).astype(np.float32, copy=False)
            drop_factor = (
                float(self.args.presentation_overhang_drop_factor)
                if _is_presentation_mode(self.args)
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
            if spec.shape_type == newton.GeoType.MESH:
                if spec.mesh is None:
                    continue
                mesh_copy = spec.mesh.copy()
                shape_idx = builder.add_shape_mesh(
                    body=body_id,
                    mesh=mesh_copy,
                    xform=spec.local_xform,
                    scale=spec.scale,
                    cfg=shape_cfg,
                    label=f"mirror_{spec.body_name}_{spec.label}",
                )
            elif spec.shape_type == newton.GeoType.BOX:
                shape_idx = builder.add_shape_box(
                    body=body_id,
                    xform=spec.local_xform,
                    hx=float(spec.scale[0]),
                    hy=float(spec.scale[1]),
                    hz=float(spec.scale[2]),
                    cfg=shape_cfg,
                    label=f"mirror_{spec.body_name}_{spec.label}",
                )
            else:
                continue
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
        requested_rigid_contact_max = max(0, int(self.args.rope_rigid_contact_max))
        requested_soft_contact_max = max(0, int(self.args.rope_soft_contact_max))
        default_soft_contact_max = int(self.model.shape_count) * int(self.model.particle_count)
        if requested_rigid_contact_max > 0 or requested_soft_contact_max > 0:
            pipeline_kwargs: dict[str, int] = {}
            if requested_rigid_contact_max > 0:
                pipeline_kwargs["rigid_contact_max"] = requested_rigid_contact_max
            if requested_soft_contact_max > 0:
                pipeline_kwargs["soft_contact_max"] = max(requested_soft_contact_max, default_soft_contact_max)
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                **pipeline_kwargs,
            )
            self.rope_rigid_contact_max_effective = int(self.collision_pipeline.rigid_contact_max)
            self.rope_soft_contact_max_effective = int(self.collision_pipeline.soft_contact_max)
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = None
            self.contacts = self.model.contacts()
            default_pipeline = getattr(self.model, "_collision_pipeline", None)
            self.rope_rigid_contact_max_effective = int(
                getattr(default_pipeline, "rigid_contact_max", getattr(self.model, "rigid_contact_max", 0)) or 0
            )
            self.rope_soft_contact_max_effective = int(
                getattr(default_pipeline, "soft_contact_max", default_soft_contact_max)
            )
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

    def configure_grasp_region(self, grasp_center: np.ndarray | None) -> None:
        if grasp_center is None:
            return
        points = np.asarray(self.rope_ir["x0"], dtype=np.float32)[: self.n_obj]
        center = np.asarray(grasp_center, dtype=np.float32).reshape(3)
        count = max(1, min(int(self.args.presentation_grasp_particle_count), int(points.shape[0])))
        # Prefer a compact edge segment: rank mostly by xy distance so the selected
        # particles are near the visible gripper window even if gravity has already
        # started to sag the rope.
        xy_dist = np.linalg.norm(points[:, :2] - center[None, :2], axis=1)
        z_dist = np.abs(points[:, 2] - center[2])
        score = xy_dist + 0.25 * z_dist
        indices = np.argsort(score)[:count].astype(np.int32, copy=False)
        self.grasp_metric_indices = indices
        if not bool(self.args.presentation_grasp_assist):
            return
        offsets = (points[indices] - center[None, :]).astype(np.float32, copy=False)
        self.grasp_assist_indices = indices
        self.grasp_assist_offsets = offsets
        self.grasp_assist_indices_wp = wp.array(indices, dtype=wp.int32, device=self.model.device)
        self.grasp_assist_offsets_wp = wp.array(offsets, dtype=wp.vec3, device=self.model.device)

    def _apply_grasp_assist(self, target_center: np.ndarray, strength: float) -> None:
        if (
            self.grasp_assist_indices_wp is None
            or self.grasp_assist_offsets_wp is None
            or self.state_in.particle_f is None
            or float(strength) <= 0.0
        ):
            return
        target = np.asarray(target_center, dtype=np.float32).reshape(3)
        wp.launch(
            _apply_grasp_assist_kernel,
            dim=int(self.grasp_assist_indices_wp.shape[0]),
            inputs=[
                self.state_in.particle_q,
                self.state_in.particle_qd,
                self.grasp_assist_indices_wp,
                self.grasp_assist_offsets_wp,
                wp.vec3(float(target[0]), float(target[1]), float(target[2])),
                float(self.args.presentation_grasp_assist_ke),
                float(self.args.presentation_grasp_assist_kd),
                float(self.args.presentation_grasp_assist_max_force),
                float(strength),
            ],
            outputs=[self.state_in.particle_f],
            device=self.model.device,
        )

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

    def step_frame(
        self,
        mirror_state: dict[str, dict[str, np.ndarray]],
        *,
        grasp_assist_target: np.ndarray | None = None,
        grasp_assist_strength: float = 0.0,
    ) -> dict[str, Any]:
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
            if grasp_assist_target is not None and float(grasp_assist_strength) > 0.0:
                self._apply_grasp_assist(grasp_assist_target, float(grasp_assist_strength))
            if self.collision_pipeline is None:
                self.model.collide(self.state_in, self.contacts)
            else:
                self.collision_pipeline.collide(self.state_in, self.contacts)
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
        if _is_presentation_mode(self.args):
            self.rigid.configure_presentation_for_rope(
                rope_points=np.asarray(self.rope.rope_ir["x0"], dtype=np.float32),
                rope_radius_m=float(np.mean(self.rope.physical_radius)),
            )
            if _is_pick_place_mode(self.args):
                self.rope.configure_grasp_region(self.rigid.presentation_contact_point)
        try:
            apply_viewer_shape_colors(
                self.viewer,
                self.rigid.model,
                extra_rules=[
                    (lambda name: "rope_cradle" in name, (0.05, 0.68, 0.62)),
                    (lambda name: "split_probe_table" in name, (0.58, 0.46, 0.34)),
                ],
            )
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
        self.first_grasp_assist_frame: int | None = None
        self.grasp_assist_frames = 0
        self.baseline_com_x = None
        self.preview_frames: list[np.ndarray] = []
        self.first_30_frames: list[np.ndarray] = []
        self.render_frame_count = self._resolve_render_frame_count()
        self.preview_indices = sorted(set(np.linspace(0, max(self.render_frame_count - 1, 0), 6, dtype=int).tolist()))

    def _record_start_mode(self) -> str:
        return "visible_opening" if _is_presentation_mode(self.args) else "post_settle"

    def _effective_preroll_seconds(self) -> float:
        return 0.0 if _is_presentation_mode(self.args) else float(self.args.rope_preroll_seconds)

    def _resolve_render_frame_count(self) -> int:
        requested = max(0, int(self.args.num_frames))
        if _is_presentation_mode(self.args):
            full_cycle = int(self.rigid.total_frames + int(self.args.presentation_tail_frames))
            return max(requested, full_cycle)
        return requested

    def _current_mirror_state(self, control_frame: int) -> dict[str, dict[str, np.ndarray]]:
        if _is_presentation_mode(self.args):
            return self.rigid.mirror_body_state()
        return self.rigid.parked_mirror_state() if control_frame < self.rigid.settle_frames else self.rigid.mirror_body_state()

    def _grasp_assist_strength(self, rigid_metrics: RigidStepMetrics) -> float:
        if not (_is_pick_place_mode(self.args) and bool(self.args.presentation_grasp_assist)):
            return 0.0
        phase = str(rigid_metrics.motion_phase)
        alpha = float(rigid_metrics.motion_phase_alpha)
        if phase == "close":
            return float(np.clip((alpha - 0.25) / 0.75, 0.0, 1.0))
        if phase in {"lift", "transfer", "place_lower"}:
            return 1.0
        if phase == "release":
            return float(1.0 - np.clip(alpha, 0.0, 1.0))
        return 0.0

    def _advance_one_frame(
        self, pending_world_wrench: dict[str, np.ndarray]
    ) -> tuple[RigidStepMetrics, dict[str, Any], np.ndarray, float, dict[str, np.ndarray], int, dict[str, Any]]:
        control_frame = int(self.rigid.frame_idx)
        rigid_metrics = self.rigid.step(
            external_world_wrenches=(pending_world_wrench if self.args.coupling_mode == "two_way" else None)
        )
        mirror_state = self._current_mirror_state(control_frame)
        grasp_strength = self._grasp_assist_strength(rigid_metrics)
        rope_metrics = self.rope.step_frame(
            mirror_state,
            grasp_assist_target=rigid_metrics.actual_target_xyz,
            grasp_assist_strength=grasp_strength,
        )
        rope_q = rope_metrics["particle_q"]
        rope_com = rope_q.mean(axis=0)
        rope_max_z = float(np.max(rope_q[:, 2])) if rope_q.size else 0.0
        if self.rope.grasp_metric_indices is not None and self.rope.grasp_metric_indices.size > 0:
            grasp_q = rope_q[self.rope.grasp_metric_indices]
            grasp_particle_com = grasp_q.mean(axis=0)
            grasp_particle_max_z = float(np.max(grasp_q[:, 2]))
        else:
            grasp_particle_com = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
            grasp_particle_max_z = 0.0
        pad_centers = self.rigid.pad_centers_world()
        left_pad_center = np.asarray(pad_centers["left"], dtype=np.float32)
        right_pad_center = np.asarray(pad_centers["right"], dtype=np.float32)
        left_pad_min_distance = float(np.min(np.linalg.norm(rope_q - left_pad_center[None, :], axis=1)))
        right_pad_min_distance = float(np.min(np.linalg.norm(rope_q - right_pad_center[None, :], axis=1)))
        leading_pad_center = left_pad_center if self.leading_side == "left" else right_pad_center
        leading_pad_min_distance = left_pad_min_distance if self.leading_side == "left" else right_pad_min_distance
        contact_points = self.rigid.grasp_contact_points_world()
        left_contactor = np.asarray(contact_points["left"], dtype=np.float32)
        right_contactor = np.asarray(contact_points["right"], dtype=np.float32)
        left_contactor_min_distance = float(np.min(np.linalg.norm(rope_q - left_contactor[None, :], axis=1)))
        right_contactor_min_distance = float(np.min(np.linalg.norm(rope_q - right_contactor[None, :], axis=1)))
        leading_contactor = left_contactor if self.leading_side == "left" else right_contactor
        leading_contactor_min_distance = (
            left_contactor_min_distance if self.leading_side == "left" else right_contactor_min_distance
        )
        pad_metrics = {
            "left_pad_center_xyz": left_pad_center,
            "right_pad_center_xyz": right_pad_center,
            "leading_pad_center_xyz": leading_pad_center,
            "left_pad_to_rope_min_distance_m": left_pad_min_distance,
            "right_pad_to_rope_min_distance_m": right_pad_min_distance,
            "leading_pad_to_rope_min_distance_m": float(leading_pad_min_distance),
            "left_contactor_xyz": left_contactor,
            "right_contactor_xyz": right_contactor,
            "leading_contactor_xyz": leading_contactor,
            "left_contactor_to_rope_min_distance_m": left_contactor_min_distance,
            "right_contactor_to_rope_min_distance_m": right_contactor_min_distance,
            "leading_contactor_to_rope_min_distance_m": float(leading_contactor_min_distance),
            "grasp_assist_strength": float(grasp_strength),
            "rope_max_z_m": rope_max_z,
            "grasp_particle_com_xyz": grasp_particle_com,
            "grasp_particle_max_z_m": grasp_particle_max_z,
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

        grasp_assist_strength = float(pad_metrics.get("grasp_assist_strength", 0.0))
        if grasp_assist_strength > 0.0:
            self.grasp_assist_frames += 1
            if self.first_grasp_assist_frame is None:
                self.first_grasp_assist_frame = frame_idx

        self.timeseries.append(
            {
                "frame": int(frame_idx),
                "time_s": float(rigid_metrics.time_s),
                "motion_phase": str(rigid_metrics.motion_phase),
                "motion_phase_alpha": float(rigid_metrics.motion_phase_alpha),
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
                "gripper_opening_m": float(rigid_metrics.gripper_opening_m),
                "robot_table_contact_frames": int(rigid_metrics.robot_table_contacts > 0),
                "finger_table_contact_frames": int(rigid_metrics.finger_table_contacts > 0),
                "nonfinger_table_contact_frames": int(rigid_metrics.nonfinger_table_contacts > 0),
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
                "left_contactor_x_m": float(pad_metrics["left_contactor_xyz"][0]),
                "left_contactor_y_m": float(pad_metrics["left_contactor_xyz"][1]),
                "left_contactor_z_m": float(pad_metrics["left_contactor_xyz"][2]),
                "right_contactor_x_m": float(pad_metrics["right_contactor_xyz"][0]),
                "right_contactor_y_m": float(pad_metrics["right_contactor_xyz"][1]),
                "right_contactor_z_m": float(pad_metrics["right_contactor_xyz"][2]),
                "leading_contactor_x_m": float(pad_metrics["leading_contactor_xyz"][0]),
                "leading_contactor_y_m": float(pad_metrics["leading_contactor_xyz"][1]),
                "leading_contactor_z_m": float(pad_metrics["leading_contactor_xyz"][2]),
                "left_contactor_to_rope_min_distance_m": float(
                    pad_metrics["left_contactor_to_rope_min_distance_m"]
                ),
                "right_contactor_to_rope_min_distance_m": float(
                    pad_metrics["right_contactor_to_rope_min_distance_m"]
                ),
                "leading_contactor_to_rope_min_distance_m": float(
                    pad_metrics["leading_contactor_to_rope_min_distance_m"]
                ),
                "grasp_assist_strength": grasp_assist_strength,
                "rope_com_x_m": float(rope_com[0]),
                "rope_com_y_m": float(rope_com[1]),
                "rope_com_z_m": float(rope_com[2]),
                "rope_max_z_m": float(pad_metrics["rope_max_z_m"]),
                "grasp_particle_com_x_m": float(pad_metrics["grasp_particle_com_xyz"][0]),
                "grasp_particle_com_y_m": float(pad_metrics["grasp_particle_com_xyz"][1]),
                "grasp_particle_com_z_m": float(pad_metrics["grasp_particle_com_xyz"][2]),
                "grasp_particle_max_z_m": float(pad_metrics["grasp_particle_max_z_m"]),
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
        finger_table_contact_frames = int(sum(int(row["finger_table_contact_frames"]) for row in arr))
        nonfinger_table_contact_frames = int(sum(int(row["nonfinger_table_contact_frames"]) for row in arr))
        rope_table_contact_frames = int(sum(int(row["rope_table_contact_frames"]) for row in arr))
        rope_ground_contact_frames = int(sum(int(row["rope_ground_contact_frames"]) for row in arr))
        left_finger_rope_contacts = int(sum(int(row["left_finger_rope_contacts"]) for row in arr))
        right_finger_rope_contacts = int(sum(int(row["right_finger_rope_contacts"]) for row in arr))
        first_30 = arr[: min(30, len(arr))]
        final_30 = arr[-min(30, len(arr)) :] if arr else []
        rope_table_contact_frames_first_30 = int(sum(int(row["rope_table_contact_frames"]) for row in first_30))
        rope_ground_contact_frames_first_30 = int(sum(int(row["rope_ground_contact_frames"]) for row in first_30))
        robot_table_contact_frames_first_30 = int(sum(int(row["robot_table_contact_frames"]) for row in first_30))
        finger_table_contact_frames_first_30 = int(sum(int(row["finger_table_contact_frames"]) for row in first_30))
        nonfinger_table_contact_frames_first_30 = int(sum(int(row["nonfinger_table_contact_frames"]) for row in first_30))
        rope_ground_contact_frames_final_30 = int(sum(int(row["rope_ground_contact_frames"]) for row in final_30))
        rope_z_first_30 = np.asarray([row["rope_com_z_m"] for row in first_30], dtype=np.float32) if first_30 else np.zeros((0,), dtype=np.float32)
        rope_z_all = np.asarray([row["rope_com_z_m"] for row in arr], dtype=np.float32) if arr else np.zeros((0,), dtype=np.float32)
        rope_max_z_all = np.asarray([row["rope_max_z_m"] for row in arr], dtype=np.float32) if arr else np.zeros((0,), dtype=np.float32)
        grasp_z_all = (
            np.asarray([row["grasp_particle_com_z_m"] for row in arr], dtype=np.float32) if arr else np.zeros((0,), dtype=np.float32)
        )
        max_rope_com_z_first_30 = float(np.max(rope_z_first_30)) if rope_z_first_30.size else 0.0
        max_abs_delta_rope_com_z_first_30 = (
            float(np.max(np.abs(np.diff(rope_z_first_30)))) if rope_z_first_30.size > 1 else 0.0
        )
        initial_rope_com_z_m = float(rope_z_all[0]) if rope_z_all.size else 0.0
        max_rope_com_z_m = float(np.max(rope_z_all)) if rope_z_all.size else 0.0
        final_rope_com_z_m = float(rope_z_all[-1]) if rope_z_all.size else 0.0
        rope_lift_height_m = float(max_rope_com_z_m - initial_rope_com_z_m)
        initial_rope_max_z_m = float(rope_max_z_all[0]) if rope_max_z_all.size else 0.0
        max_rope_max_z_m = float(np.max(rope_max_z_all)) if rope_max_z_all.size else 0.0
        rope_max_z_lift_height_m = float(max_rope_max_z_m - initial_rope_max_z_m)
        finite_grasp_z = grasp_z_all[np.isfinite(grasp_z_all)] if grasp_z_all.size else np.zeros((0,), dtype=np.float32)
        initial_grasp_particle_com_z_m = float(finite_grasp_z[0]) if finite_grasp_z.size else 0.0
        max_grasp_particle_com_z_m = float(np.max(finite_grasp_z)) if finite_grasp_z.size else 0.0
        grasp_particle_lift_height_m = float(max_grasp_particle_com_z_m - initial_grasp_particle_com_z_m)
        if arr and self.first_finger_contact_frame is not None:
            grasp_after_contact = np.asarray(
                [
                    row["grasp_particle_com_z_m"]
                    for row in arr
                    if int(row["frame"]) >= int(self.first_finger_contact_frame)
                    and np.isfinite(float(row["grasp_particle_com_z_m"]))
                ],
                dtype=np.float32,
            )
        else:
            grasp_after_contact = np.zeros((0,), dtype=np.float32)
        if arr and self.first_finger_contact_frame is not None:
            pre_contact_rows = [row for row in arr if int(row["frame"]) < int(self.first_finger_contact_frame)]
        else:
            pre_contact_rows = arr
        pre_contact_grasp_z = (
            np.asarray(
                [
                    row["grasp_particle_com_z_m"]
                    for row in pre_contact_rows
                    if np.isfinite(float(row["grasp_particle_com_z_m"]))
                ],
                dtype=np.float32,
            )
            if pre_contact_rows
            else np.zeros((0,), dtype=np.float32)
        )
        pre_contact_grasp_drop_m = (
            float(initial_grasp_particle_com_z_m - np.min(pre_contact_grasp_z)) if pre_contact_grasp_z.size else 0.0
        )
        pre_contact_ground_contact_frames = int(
            sum(int(row["rope_ground_contact_frames"]) for row in pre_contact_rows)
        )
        first_contact_grasp_particle_com_z_m = float(grasp_after_contact[0]) if grasp_after_contact.size else 0.0
        min_grasp_particle_com_z_after_contact_m = float(np.min(grasp_after_contact)) if grasp_after_contact.size else 0.0
        max_grasp_particle_com_z_after_contact_m = float(np.max(grasp_after_contact)) if grasp_after_contact.size else 0.0
        grasp_particle_lift_after_contact_m = float(
            max_grasp_particle_com_z_after_contact_m - first_contact_grasp_particle_com_z_m
        )
        grasp_particle_lift_from_post_contact_min_m = float(
            max_grasp_particle_com_z_after_contact_m - min_grasp_particle_com_z_after_contact_m
        )
        lift_transfer_grasp_after_contact = (
            np.asarray(
                [
                    row["grasp_particle_com_z_m"]
                    for row in arr
                    if str(row["motion_phase"]) in {"lift", "transfer"}
                    and self.first_finger_contact_frame is not None
                    and int(row["frame"]) >= int(self.first_finger_contact_frame)
                    and np.isfinite(float(row["grasp_particle_com_z_m"]))
                ],
                dtype=np.float32,
            )
            if arr
            else np.zeros((0,), dtype=np.float32)
        )
        min_grasp_particle_com_z_during_lift_transfer_m = (
            float(np.min(lift_transfer_grasp_after_contact)) if lift_transfer_grasp_after_contact.size else 0.0
        )
        max_grasp_particle_com_z_during_lift_transfer_m = (
            float(np.max(lift_transfer_grasp_after_contact)) if lift_transfer_grasp_after_contact.size else 0.0
        )
        grasp_particle_lift_during_lift_transfer_m = float(
            max_grasp_particle_com_z_during_lift_transfer_m - first_contact_grasp_particle_com_z_m
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
        min_leading_contactor_to_rope_distance_m = (
            float(min(float(row["leading_contactor_to_rope_min_distance_m"]) for row in arr)) if arr else 0.0
        )
        min_gripper_opening_m = float(min(float(row["gripper_opening_m"]) for row in arr)) if arr else 0.0
        max_gripper_opening_m = float(max(float(row["gripper_opening_m"]) for row in arr)) if arr else 0.0
        lift_window = [row for row in arr if str(row["motion_phase"]) in {"lift", "transfer"}]
        lift_window_left_finger_rope_contact_frames = int(
            sum(1 for row in lift_window if int(row["left_finger_rope_contacts"]) > 0)
        )
        lift_window_right_finger_rope_contact_frames = int(
            sum(1 for row in lift_window if int(row["right_finger_rope_contacts"]) > 0)
        )
        lift_window_left_finger_rope_contacts = int(
            sum(int(row["left_finger_rope_contacts"]) for row in lift_window)
        )
        lift_window_right_finger_rope_contacts = int(
            sum(int(row["right_finger_rope_contacts"]) for row in lift_window)
        )
        lift_window_both_finger_rope_contact_frames = int(
            sum(
                1
                for row in lift_window
                if int(row["left_finger_rope_contacts"]) > 0 and int(row["right_finger_rope_contacts"]) > 0
            )
        )
        lift_window_unilateral_finger_rope_contact_frames = int(
            sum(
                1
                for row in lift_window
                if (int(row["left_finger_rope_contacts"]) > 0)
                != (int(row["right_finger_rope_contacts"]) > 0)
            )
        )
        lift_window_contact_balance_ratio = (
            float(
                min(lift_window_left_finger_rope_contacts, lift_window_right_finger_rope_contacts)
                / max(lift_window_left_finger_rope_contacts, lift_window_right_finger_rope_contacts)
            )
            if max(lift_window_left_finger_rope_contacts, lift_window_right_finger_rope_contacts) > 0
            else 0.0
        )
        final_15 = arr[-min(15, len(arr)) :] if arr else []
        final_finger_rope_contact_frames = int(
            sum(
                1
                for row in final_15
                if int(row["left_finger_rope_contacts"]) > 0 or int(row["right_finger_rope_contacts"]) > 0
            )
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
        strict_fail_reasons: list[str] = []
        if bool(self.args.presentation_grasp_assist) or self.first_grasp_assist_frame is not None or self.grasp_assist_frames > 0:
            strict_fail_reasons.append("grasp assist must be disabled and unused")
        if bool(self.args.strict_require_native_panda_fingers) and str(self.args.presentation_gripper_geometry) != "panda_fingers":
            strict_fail_reasons.append("final strict route must use native Panda finger pads without helper geometry")
        if self.first_finger_contact_frame is None:
            strict_fail_reasons.append("no finger-rope contact frame was detected")
        if not rope_motion_after_contact:
            strict_fail_reasons.append("rope motion after finger contact was not detected")
        if pre_contact_grasp_drop_m > float(self.args.strict_max_precontact_grasp_drop):
            strict_fail_reasons.append("local grasp segment dropped before finger contact")
        if pre_contact_ground_contact_frames > 0:
            strict_fail_reasons.append("rope touched the ground before finger contact")
        if lift_window_both_finger_rope_contact_frames < int(self.args.strict_both_finger_contact_frames):
            strict_fail_reasons.append("both fingers did not maintain contact during the lift/transfer window")
        if lift_window_unilateral_finger_rope_contact_frames > int(self.args.strict_max_unilateral_lift_contact_frames):
            strict_fail_reasons.append("lift/transfer contact was unilateral, which reads as side pickup")
        if lift_window_contact_balance_ratio < float(self.args.strict_min_lift_contact_balance_ratio):
            strict_fail_reasons.append("lift/transfer contact was too one-sided, which reads as side suction")
        if grasp_particle_lift_during_lift_transfer_m < float(self.args.strict_grasp_lift_height):
            strict_fail_reasons.append("local grasp segment did not lift enough during lift/transfer")
        if rope_lift_height_m < float(self.args.strict_rope_lift_height):
            strict_fail_reasons.append("whole-rope COM did not rise enough for visible carry")
        if rope_max_z_lift_height_m < float(self.args.strict_rope_max_z_lift_height):
            strict_fail_reasons.append("no visible rope point rose enough for carry")
        if max_rope_max_z_m > float(self.args.strict_max_rope_height):
            strict_fail_reasons.append("rope flew away above the strict max height")
        if max_grasp_particle_com_z_during_lift_transfer_m < (
            float(self.rigid.table_top_z) + float(self.args.strict_lift_transfer_clearance)
        ):
            strict_fail_reasons.append("local grasp segment was not carried visibly above the table")
        if final_finger_rope_contact_frames > int(self.args.strict_final_finger_contact_max_frames):
            strict_fail_reasons.append("finger/cradle contact remained after release")
        if nonfinger_table_contact_frames != 0:
            strict_fail_reasons.append("non-finger robot body touched the table")
        if max_support_penetration_m > float(self.args.strict_max_support_penetration):
            strict_fail_reasons.append("support penetration exceeded strict threshold")
        if rope_ground_contact_frames_final_30 < int(self.args.strict_final_ground_contact_frames):
            strict_fail_reasons.append("rope did not settle on the ground at the end")
        if not rope_render_matches_physics:
            strict_fail_reasons.append("rope render radius differs from physical radius")
        strict_contact_only_pass = _is_pick_place_mode(self.args) and not strict_fail_reasons

        summary = {
            "demo": "robot_table_rope_split_mujoco_semiimplicit",
            "coupling_mode": str(self.args.coupling_mode),
            "rigid_frame_dt_s": float(self.rigid.frame_dt),
            "rigid_substep_dt_s": float(self.rigid.sim_dt),
            "rigid_njmax": int(self.args.rigid_njmax),
            "rigid_nconmax": int(self.args.rigid_nconmax),
            "rigid_hydro_buffer_mult_iso": int(self.args.rigid_hydro_buffer_mult_iso),
            "rigid_hydro_buffer_mult_contact": int(self.args.rigid_hydro_buffer_mult_contact),
            "arm_joint_target_ke": float(self.args.arm_joint_target_ke),
            "arm_joint_target_kd": float(self.args.arm_joint_target_kd),
            "arm_joint_effort_limit": float(self.args.arm_joint_effort_limit),
            "finger_joint_target_ke": float(self.args.finger_joint_target_ke),
            "finger_joint_target_kd": float(self.args.finger_joint_target_kd),
            "finger_joint_effort_limit": float(self.args.finger_joint_effort_limit),
            "gripper_yaw_rad": float(self.args.gripper_yaw),
            "rope_substep_dt_s": float(
                self.args.rope_sim_dt
                if self.args.rope_sim_dt is not None
                else (self.rigid.frame_dt / float(self.args.sim_substeps_rope))
            ),
            "rope_rigid_contact_max_requested": int(self.args.rope_rigid_contact_max),
            "rope_rigid_contact_max_effective": int(self.rope.rope_rigid_contact_max_effective),
            "rope_soft_contact_max_requested": int(self.args.rope_soft_contact_max),
            "rope_soft_contact_max_effective": int(self.rope.rope_soft_contact_max_effective),
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
            "first_grasp_assist_frame": self.first_grasp_assist_frame,
            "grasp_assist_enabled": bool(self.args.presentation_grasp_assist),
            "grasp_assist_frames": int(self.grasp_assist_frames),
            "grasp_assist_particle_count": (
                0 if self.rope.grasp_assist_indices is None else int(self.rope.grasp_assist_indices.shape[0])
            ),
            "grasp_metric_particle_count": (
                0 if self.rope.grasp_metric_indices is None else int(self.rope.grasp_metric_indices.shape[0])
            ),
            "strict_contact_only_pass": bool(strict_contact_only_pass),
            "strict_contact_only_fail_reasons": strict_fail_reasons,
            "rope_motion_after_contact": rope_motion_after_contact,
            "robot_table_contact_frames": robot_table_contact_frames,
            "finger_table_contact_frames": finger_table_contact_frames,
            "nonfinger_table_contact_frames": nonfinger_table_contact_frames,
            "strict_table_contact_policy": "edge-pick allows finger/cradle table contact but fails non-finger robot-table contact",
            "rope_table_contact_frames": rope_table_contact_frames,
            "rope_ground_contact_frames": rope_ground_contact_frames,
            "left_finger_rope_contacts": left_finger_rope_contacts,
            "right_finger_rope_contacts": right_finger_rope_contacts,
            "robot_table_contact_frames_first_30": robot_table_contact_frames_first_30,
            "finger_table_contact_frames_first_30": finger_table_contact_frames_first_30,
            "nonfinger_table_contact_frames_first_30": nonfinger_table_contact_frames_first_30,
            "rope_table_contact_frames_first_30": rope_table_contact_frames_first_30,
            "rope_ground_contact_frames_first_30": rope_ground_contact_frames_first_30,
            "rope_ground_contact_frames_final_30": rope_ground_contact_frames_final_30,
            "lift_window_left_finger_rope_contact_frames": lift_window_left_finger_rope_contact_frames,
            "lift_window_right_finger_rope_contact_frames": lift_window_right_finger_rope_contact_frames,
            "lift_window_left_finger_rope_contacts": lift_window_left_finger_rope_contacts,
            "lift_window_right_finger_rope_contacts": lift_window_right_finger_rope_contacts,
            "lift_window_both_finger_rope_contact_frames": lift_window_both_finger_rope_contact_frames,
            "lift_window_unilateral_finger_rope_contact_frames": lift_window_unilateral_finger_rope_contact_frames,
            "lift_window_contact_balance_ratio": float(lift_window_contact_balance_ratio),
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
            "effective_table_shape_contact_scale": float(_effective_table_shape_contact_scale(self.args)),
            "effective_table_shape_contact_damping_multiplier": float(
                _effective_table_shape_contact_damping_multiplier(self.args)
            ),
            "finger_shape_contact_scale": float(self.args.finger_shape_contact_scale),
            "finger_shape_contact_damping_multiplier": float(self.args.finger_shape_contact_damping_multiplier),
            "finger_shape_friction_multiplier": float(self.args.finger_shape_friction_multiplier),
            "presentation_gripper_geometry": str(self.rigid.presentation_gripper_geometry),
            "presentation_gripper_geometry_shape_count": int(len(self.rigid.gripper_geometry_shape_indices)),
            "presentation_aux_panda_pad_geometry_enabled": bool(_uses_aux_panda_pad_geometry(self.args)),
            "presentation_native_panda_finger_grasp_local_m": [
                float(self.args.presentation_panda_finger_grasp_local_x),
                float(self.args.presentation_panda_finger_grasp_local_y),
                float(self.args.presentation_panda_finger_grasp_local_z),
            ],
            "presentation_cradle_lip_length_m": float(self.args.presentation_cradle_lip_length),
            "presentation_cradle_lip_depth_m": float(self.args.presentation_cradle_lip_depth),
            "presentation_cradle_lip_thickness_m": float(self.args.presentation_cradle_lip_thickness),
            "presentation_cradle_lip_local_z_m": float(self.args.presentation_cradle_lip_local_z),
            "presentation_cradle_wall_height_m": float(self.args.presentation_cradle_wall_height),
            "presentation_cradle_wall_thickness_m": float(self.args.presentation_cradle_wall_thickness),
            "presentation_cradle_grasp_z_clearance_m": float(self.args.presentation_cradle_grasp_z_clearance),
            "presentation_grasp_opening_m": float(self.args.presentation_grasp_opening),
            "presentation_grasp_closed_opening_m": float(self.args.presentation_grasp_closed_opening),
            "presentation_release_seconds": float(self.args.presentation_release_seconds),
            "presentation_place_y_offset_m": float(self.args.presentation_place_y_offset),
            "presentation_retract_x_offset_m": float(self.args.presentation_retract_x_offset),
            "presentation_retract_y_offset_m": float(self.args.presentation_retract_y_offset),
            "presentation_retract_z_offset_m": float(self.args.presentation_retract_z_offset),
            "table_edge_inset_y": float(self.args.table_edge_inset_y),
            "overhang_drop_factor": float(self.args.overhang_drop_factor),
            "min_clearance_extra": float(self.args.min_clearance_extra),
            "presentation_table_edge_inset_y": float(self.args.presentation_table_edge_inset_y),
            "presentation_edge_grasp_outset_y_m": float(self.args.presentation_edge_grasp_outset_y),
            "max_rope_com_z_first_30": float(max_rope_com_z_first_30),
            "max_abs_delta_rope_com_z_first_30": float(max_abs_delta_rope_com_z_first_30),
            "initial_rope_com_z_m": float(initial_rope_com_z_m),
            "max_rope_com_z_m": float(max_rope_com_z_m),
            "final_rope_com_z_m": float(final_rope_com_z_m),
            "rope_lift_height_m": float(rope_lift_height_m),
            "initial_rope_max_z_m": float(initial_rope_max_z_m),
            "max_rope_max_z_m": float(max_rope_max_z_m),
            "rope_max_z_lift_height_m": float(rope_max_z_lift_height_m),
            "initial_grasp_particle_com_z_m": float(initial_grasp_particle_com_z_m),
            "max_grasp_particle_com_z_m": float(max_grasp_particle_com_z_m),
            "grasp_particle_lift_height_m": float(grasp_particle_lift_height_m),
            "pre_contact_grasp_particle_drop_m": float(pre_contact_grasp_drop_m),
            "pre_contact_ground_contact_frames": int(pre_contact_ground_contact_frames),
            "first_contact_grasp_particle_com_z_m": float(first_contact_grasp_particle_com_z_m),
            "min_grasp_particle_com_z_after_contact_m": float(min_grasp_particle_com_z_after_contact_m),
            "max_grasp_particle_com_z_after_contact_m": float(max_grasp_particle_com_z_after_contact_m),
            "grasp_particle_lift_after_contact_m": float(grasp_particle_lift_after_contact_m),
            "grasp_particle_lift_from_post_contact_min_m": float(grasp_particle_lift_from_post_contact_min_m),
            "min_grasp_particle_com_z_during_lift_transfer_m": float(
                min_grasp_particle_com_z_during_lift_transfer_m
            ),
            "max_grasp_particle_com_z_during_lift_transfer_m": float(
                max_grasp_particle_com_z_during_lift_transfer_m
            ),
            "grasp_particle_lift_during_lift_transfer_m": float(grasp_particle_lift_during_lift_transfer_m),
            "max_ground_penetration_m": float(max_ground_penetration_m),
            "max_table_penetration_m": float(max_table_penetration_m),
            "max_support_penetration_m": float(max_support_penetration_m),
            "final_ground_penetration_p99_m": float(final_ground_penetration_p99_m),
            "final_table_penetration_p99_m": float(final_table_penetration_p99_m),
            "final_support_penetration_p99_m": float(final_support_penetration_p99_m),
            "min_leading_pad_to_rope_distance_m": float(min_leading_pad_to_rope_distance_m),
            "min_leading_contactor_to_rope_distance_m": float(min_leading_contactor_to_rope_distance_m),
            "min_gripper_opening_m": float(min_gripper_opening_m),
            "max_gripper_opening_m": float(max_gripper_opening_m),
            "final_finger_rope_contact_frames": int(final_finger_rope_contact_frames),
            "strict_grasp_lift_height_m": float(self.args.strict_grasp_lift_height),
            "strict_rope_lift_height_m": float(self.args.strict_rope_lift_height),
            "strict_rope_max_z_lift_height_m": float(self.args.strict_rope_max_z_lift_height),
            "strict_max_rope_height_m": float(self.args.strict_max_rope_height),
            "strict_both_finger_contact_frames": int(self.args.strict_both_finger_contact_frames),
            "strict_require_native_panda_fingers": bool(self.args.strict_require_native_panda_fingers),
            "strict_min_lift_contact_balance_ratio": float(self.args.strict_min_lift_contact_balance_ratio),
            "strict_max_unilateral_lift_contact_frames": int(self.args.strict_max_unilateral_lift_contact_frames),
            "strict_max_support_penetration_m": float(self.args.strict_max_support_penetration),
            "strict_final_ground_contact_frames": int(self.args.strict_final_ground_contact_frames),
            "strict_max_precontact_grasp_drop_m": float(self.args.strict_max_precontact_grasp_drop),
            "strict_lift_transfer_clearance_m": float(self.args.strict_lift_transfer_clearance),
            "strict_final_finger_contact_max_frames": int(self.args.strict_final_finger_contact_max_frames),
            "presentation_contact_target_x_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[0])
            ),
            "presentation_contact_target_y_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[1])
            ),
            "presentation_contact_target_z_m": (
                None if self.rigid.presentation_contact_point is None else float(self.rigid.presentation_contact_point[2])
            ),
            "presentation_grasp_x_offset_m": float(self.args.presentation_grasp_x_offset),
            "presentation_grasp_y_offset_m": float(self.args.presentation_grasp_y_offset),
            "presentation_grasp_z_offset_m": float(self.args.presentation_grasp_z_offset),
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
            gripper_opening=np.asarray([row["gripper_opening_m"] for row in arr], dtype=np.float32),
            grasp_assist_strength=np.asarray([row["grasp_assist_strength"] for row in arr], dtype=np.float32),
            rope_com_xyz=np.asarray([[row["rope_com_x_m"], row["rope_com_y_m"], row["rope_com_z_m"]] for row in arr], dtype=np.float32),
            rope_max_z=np.asarray([row["rope_max_z_m"] for row in arr], dtype=np.float32),
            grasp_particle_com_xyz=np.asarray(
                [[row["grasp_particle_com_x_m"], row["grasp_particle_com_y_m"], row["grasp_particle_com_z_m"]] for row in arr],
                dtype=np.float32,
            ),
            grasp_particle_max_z=np.asarray([row["grasp_particle_max_z_m"] for row in arr], dtype=np.float32),
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
                    f"- Strict contact-only pass: `{strict_contact_only_pass}`",
                    f"- Min leading-pad distance to rope: `{min_leading_pad_to_rope_distance_m:.6f}`",
                    f"- Grasp assist enabled: `{bool(self.args.presentation_grasp_assist)}`",
                    f"- Grasp-particle lift height (m): `{grasp_particle_lift_height_m:.6f}`",
                    f"- Grasp-particle lift during lift/transfer (m): `{grasp_particle_lift_during_lift_transfer_m:.6f}`",
                    f"- Rope render matches physics: `{rope_render_matches_physics}`",
                    f"- Record start mode: `{self._record_start_mode()}`",
                    f"- Rope pre-roll seconds: `{self._effective_preroll_seconds():.3f}`",
                    f"- Rope rigid-contact max effective: `{int(self.rope.rope_rigid_contact_max_effective)}`",
                    f"- Rope soft-contact max effective: `{int(self.rope.rope_soft_contact_max_effective)}`",
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
    parser.add_argument("--motion-pattern", choices=["side_finger_push", "grasp_lift_place"], default="side_finger_push")
    parser.add_argument("--rope-render-mode", choices=["physical_only"], default="physical_only")
    parser.add_argument(
        "--video-mode",
        choices=["support_default", "presentation_lifted", "presentation_pick_place"],
        default="support_default",
    )
    parser.add_argument("--finger-contact-set", choices=["fingers_plus_pads"], default="fingers_plus_pads")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=170)
    parser.add_argument("--sim-substeps-rigid", type=int, default=10)
    parser.add_argument("--rigid-njmax", type=int, default=2048)
    parser.add_argument("--rigid-nconmax", type=int, default=2048)
    parser.add_argument("--rigid-hydro-buffer-mult-iso", type=int, default=4)
    parser.add_argument("--rigid-hydro-buffer-mult-contact", type=int, default=4)
    parser.add_argument("--arm-joint-target-ke", type=float, default=650.0)
    parser.add_argument("--arm-joint-target-kd", type=float, default=100.0)
    parser.add_argument("--arm-joint-effort-limit", type=float, default=80.0)
    parser.add_argument("--finger-joint-target-ke", type=float, default=650.0)
    parser.add_argument("--finger-joint-target-kd", type=float, default=100.0)
    parser.add_argument("--finger-joint-effort-limit", type=float, default=20.0)
    parser.add_argument("--sim-substeps-rope", type=int, default=667)
    parser.add_argument("--calibration-sim-substeps-rope", type=int, default=32)
    parser.add_argument("--rope-sim-dt", type=float, default=None)
    parser.add_argument(
        "--rope-rigid-contact-max",
        type=int,
        default=0,
        help="Override the rope-side rigid contact buffer capacity. 0 keeps Newton's default estimate.",
    )
    parser.add_argument(
        "--rope-soft-contact-max",
        type=int,
        default=0,
        help="Override the rope-side soft-contact buffer capacity. 0 keeps Newton's default shape_count*particle_count.",
    )
    parser.add_argument("--rope-preroll-seconds", type=float, default=2.0)
    parser.add_argument("--presentation-lift-height", type=float, default=0.0)
    parser.add_argument("--presentation-table-edge-inset-y", type=float, default=0.04)
    parser.add_argument("--presentation-overhang-drop-factor", type=float, default=0.0)
    parser.add_argument("--presentation-opening-seconds", type=float, default=0.05)
    parser.add_argument("--presentation-approach-seconds", type=float, default=0.12)
    parser.add_argument("--presentation-retract-seconds", type=float, default=0.6)
    parser.add_argument("--presentation-push-distance", type=float, default=0.05)
    parser.add_argument("--presentation-push-speed-mps", type=float, default=0.02)
    parser.add_argument("--presentation-tail-frames", type=int, default=15)
    parser.add_argument("--presentation-proximity-threshold", type=float, default=0.015)
    parser.add_argument("--presentation-grasp-lower-seconds", type=float, default=0.10)
    parser.add_argument("--presentation-grasp-close-seconds", type=float, default=0.20)
    parser.add_argument("--presentation-lift-seconds", type=float, default=0.8)
    parser.add_argument("--presentation-transfer-seconds", type=float, default=1.0)
    parser.add_argument("--presentation-place-lower-seconds", type=float, default=0.7)
    parser.add_argument("--presentation-release-seconds", type=float, default=0.4)
    parser.add_argument("--presentation-grasp-opening", type=float, default=0.04)
    parser.add_argument("--presentation-grasp-closed-opening", type=float, default=0.025)
    parser.add_argument("--presentation-grasp-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--presentation-grasp-particle-count", type=int, default=24)
    parser.add_argument("--presentation-grasp-assist-ke", type=float, default=10.0)
    parser.add_argument("--presentation-grasp-assist-kd", type=float, default=0.1)
    parser.add_argument("--presentation-grasp-assist-max-force", type=float, default=1.0)
    parser.add_argument("--presentation-gripper-geometry", choices=["panda_fingers", "panda_pads", "rope_cradle"], default="panda_fingers")
    parser.add_argument("--presentation-panda-finger-grasp-local-x", type=float, default=0.0)
    parser.add_argument("--presentation-panda-finger-grasp-local-y", type=float, default=0.005)
    parser.add_argument("--presentation-panda-finger-grasp-local-z", type=float, default=0.045)
    parser.add_argument("--presentation-cradle-lip-length", type=float, default=0.045)
    parser.add_argument("--presentation-cradle-lip-depth", type=float, default=0.020)
    parser.add_argument("--presentation-cradle-lip-thickness", type=float, default=0.006)
    parser.add_argument("--presentation-cradle-lip-local-z", type=float, default=0.055)
    parser.add_argument("--presentation-cradle-wall-height", type=float, default=0.018)
    parser.add_argument("--presentation-cradle-wall-thickness", type=float, default=0.005)
    parser.add_argument("--presentation-cradle-grasp-z-clearance", type=float, default=0.010)
    parser.add_argument("--presentation-grasp-x-offset", type=float, default=0.0)
    parser.add_argument("--presentation-grasp-y-offset", type=float, default=0.0)
    parser.add_argument("--presentation-grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--presentation-grasp-z-clearance", type=float, default=0.060)
    parser.add_argument("--presentation-edge-grasp-outset-y", type=float, default=0.060)
    parser.add_argument("--presentation-table-shape-contact-scale", type=float, default=0.2)
    parser.add_argument("--presentation-table-shape-contact-damping-multiplier", type=float, default=64.0)
    parser.add_argument("--presentation-carry-height", type=float, default=0.16)
    parser.add_argument("--presentation-place-ground-clearance", type=float, default=0.018)
    parser.add_argument("--presentation-place-x-offset", type=float, default=0.04)
    parser.add_argument("--presentation-place-y-offset", type=float, default=0.12)
    parser.add_argument("--presentation-pregrasp-y-standoff", type=float, default=0.035)
    parser.add_argument("--presentation-pregrasp-z-offset", type=float, default=0.015)
    parser.add_argument("--presentation-approach-x-offset", type=float, default=0.12)
    parser.add_argument("--presentation-approach-y-offset", type=float, default=0.05)
    parser.add_argument("--presentation-approach-z-offset", type=float, default=0.015)
    parser.add_argument("--presentation-retract-x-offset", type=float, default=0.08)
    parser.add_argument("--presentation-retract-y-offset", type=float, default=0.06)
    parser.add_argument("--presentation-retract-z-offset", type=float, default=0.05)
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
    parser.add_argument("--finger-shape-friction-multiplier", type=float, default=1.0)
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
    parser.add_argument("--strict-grasp-lift-height", type=float, default=0.02)
    parser.add_argument("--strict-rope-lift-height", type=float, default=0.025)
    parser.add_argument("--strict-rope-max-z-lift-height", type=float, default=0.025)
    parser.add_argument("--strict-max-rope-height", type=float, default=1.0)
    parser.add_argument("--strict-both-finger-contact-frames", type=int, default=10)
    parser.add_argument("--strict-require-native-panda-fingers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-min-lift-contact-balance-ratio", type=float, default=0.20)
    parser.add_argument("--strict-max-unilateral-lift-contact-frames", type=int, default=0)
    parser.add_argument("--strict-max-support-penetration", type=float, default=0.0025)
    parser.add_argument("--strict-final-ground-contact-frames", type=int, default=20)
    parser.add_argument("--strict-max-precontact-grasp-drop", type=float, default=0.035)
    parser.add_argument("--strict-lift-transfer-clearance", type=float, default=0.04)
    parser.add_argument("--strict-final-finger-contact-max-frames", type=int, default=0)
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
