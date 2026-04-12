#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image, ImageDraw
from pxr import Usd

from bridge_bootstrap import newton
from newton._src.viewer.viewer_gl import ViewerGL
from newton.geometry import HydroelasticSDF

import newton.ik as ik
import newton.usd
import newton.utils


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
        draw.rectangle((x, y, x + 150, y + 28), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), f"frame {idx}", fill=(255, 255, 255))
    canvas.save(output_path)


class PenetrationProbe:
    def __init__(self, viewer: ViewerGL, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.fps = int(args.fps)
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = int(args.sim_substeps)
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.frame_idx = 0

        self.table_half_extents = np.array([0.10, 0.10, 0.05], dtype=np.float32)
        self.table_pos = np.array([0.08, -0.50, 0.05], dtype=np.float32)
        self.table_top_z = float(self.table_pos[2] + self.table_half_extents[2])
        self.attempt_target_xy = np.array([0.08, -0.50], dtype=np.float32)
        self.approach_z = float(args.approach_z)
        self.penetration_target_z = float(args.penetration_target_z)
        self.hold_frames = int(args.hold_frames)
        self.descend_frames = int(args.descend_frames)
        self.total_frames = int(args.num_frames)

        if self.penetration_target_z >= self.table_top_z:
            raise ValueError(
                f"--penetration-target-z must be below the table top ({self.table_top_z:.4f} m), "
                f"got {self.penetration_target_z:.4f}"
            )

        self.pad_xform = wp.transform(
            wp.vec3(0.0, 0.005, 0.045),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )

        self._build_scene()
        self._setup_ik()
        self._set_camera()
        self._reset_metrics()

    def _build_scene(self) -> None:
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        mesh_shape_cfg = copy.deepcopy(shape_cfg)
        mesh_shape_cfg.sdf_max_resolution = None
        mesh_shape_cfg.sdf_target_voxel_size = None
        mesh_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        hydro_mesh_sdf_max_resolution = 64

        builder = newton.ModelBuilder()

        urdf_shape_cfg = copy.deepcopy(shape_cfg)
        urdf_shape_cfg.is_hydroelastic = False
        urdf_shape_cfg.sdf_max_resolution = None
        urdf_shape_cfg.sdf_target_voxel_size = None
        urdf_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        builder.default_shape_cfg = urdf_shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )
        builder.default_shape_cfg = shape_cfg

        def find_body(name: str) -> int:
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        self.left_finger_body_idx = find_body("fr3_leftfinger")
        self.right_finger_body_idx = find_body("fr3_rightfinger")
        self.hand_body_idx = find_body("fr3_hand")
        self.ee_index = 10

        self.finger_body_indices = {
            self.left_finger_body_idx,
            self.right_finger_body_idx,
            self.hand_body_idx,
        }

        self.finger_shape_indices: list[int] = []
        self.nonfinger_robot_shape_indices: list[int] = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in self.finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=hydro_mesh_sdf_max_resolution,
                        narrow_band_range=shape_cfg.sdf_narrow_band_range,
                        margin=shape_cfg.gap if shape_cfg.gap is not None else 0.05,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
                self.finger_shape_indices.append(shape_idx)
            elif body_idx >= 0:
                builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                self.nonfinger_robot_shape_indices.append(shape_idx)

        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=self.nonfinger_robot_shape_indices,
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
        builder.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder.joint_target_pos[:9] = [*init_q, 0.05, 0.05]
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
            max_resolution=hydro_mesh_sdf_max_resolution,
            narrow_band_range=shape_cfg.sdf_narrow_band_range,
            margin=shape_cfg.gap if shape_cfg.gap is not None else 0.05,
        )
        left_pad_shape = builder.add_shape_mesh(
            body=self.left_finger_body_idx,
            mesh=pad_mesh,
            xform=self.pad_xform,
            cfg=mesh_shape_cfg,
            label="left_pad",
        )
        right_pad_shape = builder.add_shape_mesh(
            body=self.right_finger_body_idx,
            mesh=pad_mesh,
            xform=self.pad_xform,
            cfg=mesh_shape_cfg,
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
            max_resolution=hydro_mesh_sdf_max_resolution,
            narrow_band_range=shape_cfg.sdf_narrow_band_range,
            margin=shape_cfg.gap if shape_cfg.gap is not None else 0.05,
        )
        self.table_shape_idx = builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(*self.table_pos.tolist()), wp.quat_identity()),
            cfg=mesh_shape_cfg,
            label="probe_table",
        )

        builder.add_ground_plane()

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

    def _setup_ik(self) -> None:
        self.state_single = self.model_single.state()
        newton.eval_fk(self.model_single, self.model_single.joint_q, self.model_single.joint_qd, self.state_single)
        body_q_np = self.state_single.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(
                [wp.transform_get_translation(self.ee_tf)],
                dtype=wp.vec3,
            ),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [quat_to_vec4(wp.transform_get_rotation(self.ee_tf))],
                dtype=wp.vec4,
            ),
        )
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_single.joint_limit_lower,
            joint_limit_upper=self.model_single.joint_limit_upper,
        )
        self.joint_q_ik = wp.array(
            self.model_single.joint_q,
            shape=(1, self.model_single.joint_coord_count),
        )
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _set_camera(self) -> None:
        self.viewer.set_camera(wp.vec3(0.42, -0.02, 0.38), -18.0, -125.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 70.0

    def _reset_metrics(self) -> None:
        self.timeseries: list[dict[str, float | int | bool]] = []
        self.any_finger_contact = False
        self.any_nonfinger_contact = False

    def _target_position(self) -> np.ndarray:
        if self.frame_idx < self.hold_frames:
            z = self.approach_z
        elif self.frame_idx < self.hold_frames + self.descend_frames:
            alpha = (self.frame_idx - self.hold_frames) / max(self.descend_frames - 1, 1)
            z = float((1.0 - alpha) * self.approach_z + alpha * self.penetration_target_z)
        else:
            z = self.penetration_target_z
        return np.array([self.attempt_target_xy[0], self.attempt_target_xy[1], z], dtype=np.float32)

    def _contact_counts(self) -> tuple[int, int]:
        count = int(np.asarray(self.contacts.rigid_contact_count.numpy()).reshape(-1)[0])
        if count <= 0:
            return 0, 0
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count].astype(np.int32, copy=False)
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count].astype(np.int32, copy=False)
        finger = 0
        nonfinger = 0
        finger_shapes = set(self.finger_shape_indices)
        nonfinger_shapes = set(self.nonfinger_robot_shape_indices)
        for s0, s1 in zip(shape0, shape1, strict=False):
            pair = {int(s0), int(s1)}
            if self.table_shape_idx in pair:
                other = int(s1) if int(s0) == self.table_shape_idx else int(s0)
                if other in finger_shapes:
                    finger += 1
                elif other in nonfinger_shapes:
                    nonfinger += 1
        return finger, nonfinger

    def step(self) -> None:
        target_position = self._target_position()
        target_rotation = wp.mul(
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0),
        )

        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        wp.copy(self.control.joint_target_pos, self.joint_q_ik.flatten())

        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        body_q = self.state_0.body_q.numpy()
        actual_ee = np.asarray(body_q[self.ee_index][:3], dtype=np.float32)
        finger_contacts, nonfinger_contacts = self._contact_counts()
        self.any_finger_contact = self.any_finger_contact or finger_contacts > 0
        self.any_nonfinger_contact = self.any_nonfinger_contact or nonfinger_contacts > 0
        ee_error = float(np.linalg.norm(target_position - actual_ee))
        target_below_table = bool(target_position[2] < self.table_top_z)

        self.timeseries.append(
            {
                "frame": int(self.frame_idx),
                "time_s": float(self.sim_time),
                "target_x_m": float(target_position[0]),
                "target_y_m": float(target_position[1]),
                "target_z_m": float(target_position[2]),
                "actual_ee_x_m": float(actual_ee[0]),
                "actual_ee_y_m": float(actual_ee[1]),
                "actual_ee_z_m": float(actual_ee[2]),
                "ee_error_m": ee_error,
                "target_below_table": int(target_below_table),
                "finger_table_contacts": int(finger_contacts),
                "nonfinger_table_contacts": int(nonfinger_contacts),
            }
        )

        self.frame_idx += 1
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def build_summary(self, out_dir: Path) -> dict[str, object]:
        if not self.timeseries:
            raise RuntimeError("No timeseries recorded")

        target_z = np.array([row["target_z_m"] for row in self.timeseries], dtype=np.float64)
        actual_z = np.array([row["actual_ee_z_m"] for row in self.timeseries], dtype=np.float64)
        ee_error = np.array([row["ee_error_m"] for row in self.timeseries], dtype=np.float64)
        finger_contacts = np.array([row["finger_table_contacts"] for row in self.timeseries], dtype=np.int32)
        nonfinger_contacts = np.array([row["nonfinger_table_contacts"] for row in self.timeseries], dtype=np.int32)
        attempt_mask = target_z < self.table_top_z

        attempt_frames = int(attempt_mask.sum())
        mean_error_during_attempt = float(ee_error[attempt_mask].mean()) if attempt_frames > 0 else 0.0
        max_error_during_attempt = float(ee_error[attempt_mask].max()) if attempt_frames > 0 else 0.0
        min_actual_ee_z = float(actual_z.min())
        final_actual_ee_z = float(actual_z[-1])
        finger_contact_frames = int((finger_contacts > 0).sum())
        nonfinger_contact_frames = int((nonfinger_contacts > 0).sum())
        actual_stays_above_target = float(final_actual_ee_z - self.penetration_target_z)

        attempt_blocked = bool(
            attempt_frames > 0
            and finger_contact_frames > 0
            and mean_error_during_attempt > 0.03
            and actual_stays_above_target > 0.05
        )

        if attempt_blocked:
            verdict = "target_went_below_table_but_robot_remained_above_with_contact_and_large_error"
        elif finger_contact_frames > 0:
            verdict = "contact_detected_but_blocking_signal_is_weaker_than_threshold"
        else:
            verdict = "no_finger_table_contact_detected"

        summary = {
            "probe": "native_robot_table_penetration_probe",
            "reference_style": "robot_panda_hydro",
            "solver": 'SolverMuJoCo(use_mujoco_contacts=False, solver="newton", integrator="implicitfast", cone="elliptic")',
            "robot_drive_pattern": "IK_to_control_joint_target_pos_to_solver_owned_state",
            "table_top_z_m": self.table_top_z,
            "approach_z_m": self.approach_z,
            "penetration_target_z_m": self.penetration_target_z,
            "target_below_table_by_m": float(self.table_top_z - self.penetration_target_z),
            "frames": len(self.timeseries),
            "attempt_frames": attempt_frames,
            "finger_table_contact_frames": finger_contact_frames,
            "nonfinger_table_contact_frames": nonfinger_contact_frames,
            "mean_ee_error_during_attempt_m": mean_error_during_attempt,
            "max_ee_error_during_attempt_m": max_error_during_attempt,
            "min_actual_ee_z_m": min_actual_ee_z,
            "final_actual_ee_z_m": final_actual_ee_z,
            "final_actual_minus_target_z_m": actual_stays_above_target,
            "attempt_blocked": attempt_blocked,
            "verdict": verdict,
            "artifacts": {
                "scene_npz": str(out_dir / "scene.npz"),
                "timeseries_csv": str(out_dir / "timeseries.csv"),
                "mp4": str(out_dir / "penetration_probe.mp4"),
                "gif": str(out_dir / "penetration_probe.gif"),
                "contact_sheet": str(out_dir / "contact_sheet.jpg"),
            },
        }
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal native Panda + rigid-table probe that intentionally targets below the table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--approach-z", type=float, default=0.26)
    parser.add_argument("--penetration-target-z", type=float, default=0.04)
    parser.add_argument("--hold-frames", type=int, default=60)
    parser.add_argument("--descend-frames", type=int, default=120)
    return parser.parse_args()


def write_timeseries_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Native Robot Table Penetration Probe",
        "",
        "- Scene: `Franka Panda + native rigid table`",
        "- Reference style: `robot_panda_hydro`",
        f"- Solver: `{summary['solver']}`",
        f"- Table top z: `{summary['table_top_z_m']:.4f} m`",
        f"- Target z: `{summary['penetration_target_z_m']:.4f} m`",
        f"- Attempt blocked: `{summary['attempt_blocked']}`",
        f"- Verdict: `{summary['verdict']}`",
        "",
        "Artifacts:",
        "- `summary.json`",
        "- `scene.npz`",
        "- `timeseries.csv`",
        "- `penetration_probe.mp4`",
        "- `penetration_probe.gif`",
        "- `contact_sheet.jpg`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        wp.set_device(args.device)

    viewer = ViewerGL(width=args.width, height=args.height, headless=True)
    probe = PenetrationProbe(viewer, args)

    mp4_path = args.out_dir / "penetration_probe.mp4"
    gif_path = args.out_dir / "penetration_probe.gif"
    sheet_path = args.out_dir / "contact_sheet.jpg"
    ffmpeg = _ffmpeg_encode_mp4(args.width, args.height, args.fps, mp4_path)
    if ffmpeg.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin for probe video")

    preview_frames: list[np.ndarray] = []
    preview_indices = sorted(set(np.linspace(0, max(args.num_frames - 1, 0), 6, dtype=int).tolist()))

    try:
        for frame_idx in range(args.num_frames):
            probe.step()
            probe.render()
            frame = viewer.get_frame(render_ui=False).numpy()
            ffmpeg.stdin.write(frame.tobytes())
            if frame_idx in preview_indices:
                preview_frames.append(frame.copy())
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        viewer.close()

    _make_gif(mp4_path, gif_path)
    _save_contact_sheet(preview_frames, sheet_path)

    summary = probe.build_summary(args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.savez(
        args.out_dir / "scene.npz",
        frame=np.array([row["frame"] for row in probe.timeseries], dtype=np.int32),
        time_s=np.array([row["time_s"] for row in probe.timeseries], dtype=np.float32),
        target_xyz=np.array(
            [[row["target_x_m"], row["target_y_m"], row["target_z_m"]] for row in probe.timeseries],
            dtype=np.float32,
        ),
        actual_ee_xyz=np.array(
            [[row["actual_ee_x_m"], row["actual_ee_y_m"], row["actual_ee_z_m"]] for row in probe.timeseries],
            dtype=np.float32,
        ),
        ee_error_m=np.array([row["ee_error_m"] for row in probe.timeseries], dtype=np.float32),
        finger_table_contacts=np.array([row["finger_table_contacts"] for row in probe.timeseries], dtype=np.int32),
        nonfinger_table_contacts=np.array(
            [row["nonfinger_table_contacts"] for row in probe.timeseries],
            dtype=np.int32,
        ),
    )
    write_timeseries_csv(args.out_dir / "timeseries.csv", probe.timeseries)
    write_readme(args.out_dir / "README.md", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
