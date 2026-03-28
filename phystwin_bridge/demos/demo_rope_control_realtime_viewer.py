#!/usr/bin/env python3
"""Realtime Newton viewer for the original PhysTwin rope controller trajectory.

This viewer keeps the PhysTwin controller points intact and replays the exact
controller trajectory stored in the strict IR:

- object particles stay deformable
- controller particles stay kinematic
- every substep writes the interpolated controller positions into Newton state
- the rope follows through the original controller-object spring graph

The goal is not to create a new interaction task. The goal is to answer a much
more basic question first:

1. can Newton replay the original rope control trajectory in a realtime viewer?
2. what is the actual per-step cost of this controller-driven version?
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from bridge_bootstrap import newton, newton_import_ir, path_defaults
import newton.examples
from semiimplicit_bridge_kernels import (
    eval_body_contact_forces,
    eval_body_joint_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
DEFAULT_IR = WORKSPACE_ROOT / "Newton/phystwin_bridge/ir/rope_double_hand/phystwin_ir_v2_bf_strict.npz"
REQUIRED_RUNTIME_DEVICE = "cuda:0"
PROFILE_OP_NAMES = (
    "write_kinematic_state",
    "particle_grid_build",
    "model_collide",
    "spring_forces",
    "triangle_forces",
    "bending_forces",
    "tetra_forces",
    "body_joint_forces",
    "particle_contact_forces",
    "triangle_contact_forces",
    "body_contact_forces",
    "particle_body_contact_forces",
    "integrate_particles",
    "integrate_bodies",
    "solver_step",
    "drag_correction",
    "total_substep",
    "total_step",
)


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Realtime Newton viewer for the original PhysTwin rope controller trajectory."

    parser.add_argument("--ir", type=Path, default=DEFAULT_IR, help="Path to PhysTwin strict rope IR.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WORKSPACE_ROOT / "tmp/rope_control_realtime_viewer",
        help="Directory for optional profiling outputs.",
    )
    parser.add_argument("--prefix", default="rope_control_realtime")
    parser.add_argument(
        "--runtime-device",
        default=REQUIRED_RUNTIME_DEVICE,
        help=(
            "Newton runtime device. This realtime viewer is pinned to the RTX 4090 "
            f"workstation path and must run on {REQUIRED_RUNTIME_DEVICE}."
        ),
    )

    parser.add_argument(
        "--sim-dt",
        type=float,
        default=None,
        help="Override physics dt. If omitted, reuse the dt stored in the IR.",
    )
    parser.add_argument(
        "--segment-substeps",
        type=int,
        default=None,
        help="Override the number of Newton substeps between two controller trajectory frames.",
    )
    parser.add_argument(
        "--steps-per-render",
        type=int,
        default=1,
        help=(
            "Physics substeps advanced before each viewer draw. Keep this small to "
            "preserve responsiveness while leaving dt and controller interpolation unchanged."
        ),
    )
    parser.add_argument(
        "--trajectory-frame-limit",
        type=int,
        default=None,
        help="Optional cap on how many controller trajectory frames to replay.",
    )

    parser.add_argument("--gravity-mag", type=float, default=9.8)
    parser.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drag-damping-scale", type=float, default=1.0)
    parser.add_argument(
        "--interpolate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly interpolate controller positions inside each controller frame interval.",
    )
    parser.add_argument(
        "--add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the Newton z=0 ground plane. Disabled by default to preserve the original rope trajectory setup.",
    )
    parser.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable particle-vs-shape contact pipeline. Disabled by default in the pure rope trajectory viewer.",
    )

    parser.add_argument("--camera-pos", type=float, nargs=3, default=(-0.55, 1.20, 0.72), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-pitch", type=float, default=-14.0)
    parser.add_argument("--camera-yaw", type=float, default=-92.0)
    parser.add_argument("--camera-fov", type=float, default=50.0)
    parser.add_argument("--particle-radius-vis-scale", type=float, default=1.35)
    parser.add_argument("--particle-radius-vis-cap", type=float, default=0.028)

    parser.add_argument(
        "--profile-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable rendering and profile a no-render rollout.",
    )
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--profile-warmup-runs", type=int, default=1)
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Optional path for JSON profiling summary. Defaults to <out-dir>/<prefix>_profile.json.",
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=None,
        help="Optional path for CSV profiling summary. Defaults to <out-dir>/<prefix>_profile.csv.",
    )
    return parser


def _resolve_workspace_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (WORKSPACE_ROOT / raw).resolve()


def _require_runtime_device(requested: str | None) -> str:
    raw = str(requested or path_defaults.default_device())
    resolved = newton_import_ir.resolve_device(raw)
    if resolved != REQUIRED_RUNTIME_DEVICE:
        raise RuntimeError(
            "This realtime viewer is pinned to the local RTX 4090 path and must run "
            f"on {REQUIRED_RUNTIME_DEVICE}. Requested device resolved to {resolved!r} "
            f"from {raw!r}. Re-run with --runtime-device {REQUIRED_RUNTIME_DEVICE}."
        )
    return resolved


def _load_runtime_ir(path: Path) -> dict[str, np.ndarray]:
    ir = newton_import_ir.load_ir(path.resolve())
    required = [
        "x0",
        "v0",
        "mass",
        "spring_edges",
        "spring_ke",
        "spring_kd",
        "controller_idx",
        "controller_traj",
        "num_object_points",
    ]
    missing = [name for name in required if name not in ir]
    if missing:
        raise KeyError(f"{path} missing required rope realtime fields: {missing}")

    ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int32).ravel()
    ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
    x0 = np.asarray(ir["x0"], dtype=np.float32)
    mass = np.asarray(ir["mass"], dtype=np.float32).ravel()
    if ctrl_idx.size == 0:
        raise ValueError("Rope realtime viewer requires non-empty controller_idx.")
    if ctrl_traj.ndim != 3 or ctrl_traj.shape[1] != ctrl_idx.size or ctrl_traj.shape[2] != 3:
        raise ValueError(f"controller_traj must be [F, C, 3], got {ctrl_traj.shape}")
    if float(np.max(np.abs(x0[ctrl_idx] - ctrl_traj[0]))) > 1.0e-7:
        raise ValueError("controller_traj[0] must match x0 at controller_idx for a faithful replay.")
    if float(np.max(np.abs(mass[ctrl_idx]))) > float(newton_import_ir.KINEMATIC_MASS_TOL):
        raise ValueError(
            "Realtime rope viewer requires controller particles to remain kinematic. "
            f"Found max |controller mass| = {float(np.max(np.abs(mass[ctrl_idx]))):.3e}."
        )
    return ir


class NewtonRopeControlViewer:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.args.ir = _resolve_workspace_path(self.args.ir)
        self.args.out_dir = _resolve_workspace_path(self.args.out_dir)
        self.args.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = _require_runtime_device(self.args.runtime_device)
        self.ir = _load_runtime_ir(self.args.ir)

        self.sim_dt = (
            float(self.args.sim_dt)
            if self.args.sim_dt is not None
            else float(newton_import_ir.ir_scalar(self.ir, "sim_dt"))
        )
        self.segment_substeps = (
            max(1, int(self.args.segment_substeps))
            if self.args.segment_substeps is not None
            else max(1, int(newton_import_ir.ir_scalar(self.ir, "sim_substeps")))
        )
        self.steps_per_render = max(1, int(self.args.steps_per_render))

        traj = np.asarray(self.ir["controller_traj"], dtype=np.float32)
        if self.args.trajectory_frame_limit is not None:
            limit = max(1, min(int(self.args.trajectory_frame_limit), int(traj.shape[0])))
            traj = traj[:limit].copy()
        self.ctrl_traj = traj
        self.ctrl_idx = np.asarray(self.ir["controller_idx"], dtype=np.int32).ravel()
        self.n_obj = int(np.asarray(self.ir["num_object_points"]).ravel()[0])
        self.n_traj_frames = int(self.ctrl_traj.shape[0])
        self.total_substeps = max(0, (self.n_traj_frames - 1) * self.segment_substeps)

        self.cfg = newton_import_ir.SimConfig(
            ir_path=self.args.ir.resolve(),
            out_dir=self.args.out_dir,
            output_prefix=self.args.prefix,
            angular_damping=0.05,
            friction_smoothing=1.0,
            enable_tri_contact=True,
            disable_particle_contact_kernel=True,
            num_frames=self.n_traj_frames,
            substeps_per_frame=self.segment_substeps,
            sim_dt=self.sim_dt,
            gravity=-float(self.args.gravity_mag),
            gravity_from_reverse_z=False,
            up_axis="Z",
            device=self.device,
            shape_contacts=bool(self.args.shape_contacts),
            add_ground_plane=bool(self.args.add_ground_plane),
            particle_contacts=None,
            interpolate_controls=bool(self.args.interpolate_controls),
            strict_physics_checks=False,
            apply_drag=bool(self.args.apply_drag),
            drag_damping_scale=float(self.args.drag_damping_scale),
        )

        model_result = newton_import_ir.build_model(self.ir, self.cfg, self.device)
        self.model = model_result.model
        self.model_checks = model_result.checks
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            angular_damping=self.cfg.angular_damping,
            friction_smoothing=self.cfg.friction_smoothing,
            enable_tri_contact=self.cfg.enable_tri_contact,
        )

        self.control = self.model.control()
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        collision_pipeline_enabled = newton_import_ir._use_collision_pipeline(self.cfg, self.ir)
        self.contacts = self.model.contacts() if collision_pipeline_enabled else None
        particle_contacts = newton_import_ir._resolve_particle_contacts(self.cfg, self.ir)
        self.particle_grid = self.model.particle_grid if (particle_contacts and not self.cfg.disable_particle_contact_kernel) else None
        self.search_radius = 0.0
        if self.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.particle_grid.reserve(self.model.particle_count)
            self.search_radius = float(self.model.particle_max_radius) * 2.0 + float(self.model.particle_cohesion)
            self.search_radius = max(self.search_radius, float(getattr(newton_import_ir, "EPSILON", 1.0e-6)))

        self.ctrl_idx_wp = wp.array(self.ctrl_idx.astype(np.int32), dtype=wp.int32, device=self.device)
        self.ctrl_target_wp = wp.empty(self.ctrl_idx.size, dtype=wp.vec3, device=self.device)
        self.ctrl_vel_wp = wp.empty(self.ctrl_idx.size, dtype=wp.vec3, device=self.device)
        self.ctrl_vel_zero = np.zeros((self.ctrl_idx.size, 3), dtype=np.float32)

        self.drag = 0.0
        if self.cfg.apply_drag and "drag_damping" in self.ir:
            self.drag = float(newton_import_ir.ir_scalar(self.ir, "drag_damping")) * float(self.cfg.drag_damping_scale)

        self._physical_particle_radius = self.model.particle_radius.numpy().astype(np.float32, copy=False).copy()
        self._visual_particle_radius = np.minimum(
            self._physical_particle_radius * float(self.args.particle_radius_vis_scale),
            float(self.args.particle_radius_vis_cap),
        ).astype(np.float32)

        self.pending_reset = False
        self.finished = False
        self.global_substep = 0
        self.sim_time = 0.0
        self.frame_index = 0
        self.last_step_wall_ms = 0.0
        self.last_render_wall_ms = 0.0
        self.profile_enabled = False
        self._profile_run_samples: dict[str, list[float]] = {}

        self._configure_viewer()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        print("Newton rope realtime viewer controls:")
        print("- Space: pause / resume (built into ViewerGL)")
        print("- H: show / hide UI")
        print("- F: frame camera on model")
        print("- ESC: exit")

    def _configure_viewer(self) -> None:
        self.viewer.set_model(self.model)
        for attr, value in (
            ("show_particles", True),
            ("show_triangles", False),
            ("show_visual", True),
            ("show_static", True),
            ("show_collision", False),
            ("show_contacts", False),
            ("picking_enabled", False),
        ):
            if hasattr(self.viewer, attr):
                setattr(self.viewer, attr, value)
        if hasattr(self.viewer, "show_ui"):
            self.viewer.show_ui = True
        try:
            self.viewer.camera.fov = float(self.args.camera_fov)
            cam_pos = np.asarray(self.args.camera_pos, dtype=np.float32)
            self.viewer.set_camera(
                wp.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
                float(self.args.camera_pitch),
                float(self.args.camera_yaw),
            )
        except Exception:
            pass

    def _profile_call(self, name: str, fn, *args, **kwargs):
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        wp.synchronize_device(self.device)
        self._profile_run_samples.setdefault(name, []).append(1000.0 * (time.perf_counter() - t0))
        return result

    def _record_profile_sample(self, name: str, duration_ms: float) -> None:
        self._profile_run_samples.setdefault(name, []).append(float(duration_ms))

    def _solver_step_granular(self) -> None:
        wp.synchronize_device(self.device)
        solver_t0 = time.perf_counter()

        model = self.model
        particle_f = self.state_in.particle_f if self.state_in.particle_count else None
        body_f = self.state_in.body_f if self.state_in.body_count else None
        control = self.control

        body_f_work = body_f
        if body_f is not None and model.joint_count and control.joint_f is not None:
            body_f_work = wp.clone(body_f)

        self._profile_call("spring_forces", eval_spring_forces, model, self.state_in, particle_f)
        self._profile_call("triangle_forces", eval_triangle_forces, model, self.state_in, control, particle_f)
        self._profile_call("bending_forces", eval_bending_forces, model, self.state_in, particle_f)
        self._profile_call("tetra_forces", eval_tetrahedra_forces, model, self.state_in, control, particle_f)
        self._profile_call(
            "body_joint_forces",
            eval_body_joint_forces,
            model,
            self.state_in,
            control,
            body_f_work,
            self.solver.joint_attach_ke,
            self.solver.joint_attach_kd,
        )
        self._profile_call("particle_contact_forces", eval_particle_contact_forces, model, self.state_in, particle_f)
        if self.solver.enable_tri_contact:
            self._profile_call("triangle_contact_forces", eval_triangle_contact_forces, model, self.state_in, particle_f)
        self._profile_call(
            "body_contact_forces",
            eval_body_contact_forces,
            model,
            self.state_in,
            self.contacts,
            friction_smoothing=self.solver.friction_smoothing,
            body_f_out=body_f_work,
        )
        self._profile_call(
            "particle_body_contact_forces",
            eval_particle_body_contact_forces,
            model,
            self.state_in,
            self.contacts,
            particle_f,
            body_f_work,
            body_f_in_world_frame=False,
        )
        self._profile_call("integrate_particles", self.solver.integrate_particles, model, self.state_in, self.state_out, self.sim_dt)
        if body_f_work is body_f:
            self._profile_call(
                "integrate_bodies",
                self.solver.integrate_bodies,
                model,
                self.state_in,
                self.state_out,
                self.sim_dt,
                self.solver.angular_damping,
            )
        else:
            body_f_prev = self.state_in.body_f
            self.state_in.body_f = body_f_work
            try:
                self._profile_call(
                    "integrate_bodies",
                    self.solver.integrate_bodies,
                    model,
                    self.state_in,
                    self.state_out,
                    self.sim_dt,
                    self.solver.angular_damping,
                )
            finally:
                self.state_in.body_f = body_f_prev

        wp.synchronize_device(self.device)
        self._record_profile_sample("solver_step", 1000.0 * (time.perf_counter() - solver_t0))

    def _controller_target_for_substep(self, global_substep: int) -> np.ndarray:
        if self.n_traj_frames <= 1:
            return self.ctrl_traj[0]
        frame = global_substep // self.segment_substeps + 1
        frame = min(frame, self.n_traj_frames - 1)
        sub = global_substep % self.segment_substeps
        return newton_import_ir.interpolate_controller(
            self.ctrl_traj, frame, sub, self.segment_substeps, bool(self.args.interpolate_controls)
        ).astype(np.float32, copy=False)

    def _trajectory_progress(self) -> tuple[int, int]:
        if self.n_traj_frames <= 1:
            return 0, 0
        frame = min(self.global_substep // self.segment_substeps + 1, self.n_traj_frames - 1)
        sub = self.global_substep % self.segment_substeps
        return int(frame), int(sub)

    def _write_controller_state(self) -> None:
        target = self._controller_target_for_substep(self.global_substep)
        self.ctrl_target_wp.assign(target)
        self.ctrl_vel_wp.assign(self.ctrl_vel_zero)
        if self.profile_enabled:
            self._profile_call(
                "write_kinematic_state",
                wp.launch,
                newton_import_ir._write_kinematic_state,
                dim=self.ctrl_idx.size,
                inputs=[
                    self.state_in.particle_q,
                    self.state_in.particle_qd,
                    self.ctrl_idx_wp,
                    self.ctrl_target_wp,
                    self.ctrl_vel_wp,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                newton_import_ir._write_kinematic_state,
                dim=self.ctrl_idx.size,
                inputs=[
                    self.state_in.particle_q,
                    self.state_in.particle_qd,
                    self.ctrl_idx_wp,
                    self.ctrl_target_wp,
                    self.ctrl_vel_wp,
                ],
                device=self.device,
            )

    def _advance_one_substep(self) -> None:
        if self.finished:
            return
        if self.profile_enabled:
            wp.synchronize_device(self.device)
            t_sub = time.perf_counter()

        self.state_in.clear_forces()
        self._write_controller_state()

        if self.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                if self.profile_enabled:
                    self._profile_call(
                        "particle_grid_build",
                        self.particle_grid.build,
                        self.state_in.particle_q,
                        radius=self.search_radius,
                    )
                else:
                    self.particle_grid.build(self.state_in.particle_q, radius=self.search_radius)

        if self.contacts is not None:
            if self.profile_enabled:
                self._profile_call("model_collide", self.model.collide, self.state_in, self.contacts)
            else:
                self.model.collide(self.state_in, self.contacts)

        if self.profile_enabled:
            self._solver_step_granular()
        else:
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, self.sim_dt)

        self.state_in, self.state_out = self.state_out, self.state_in

        if self.drag > 0.0:
            if self.profile_enabled:
                self._profile_call(
                    "drag_correction",
                    wp.launch,
                    newton_import_ir._apply_drag_correction,
                    dim=self.n_obj,
                    inputs=[self.state_in.particle_q, self.state_in.particle_qd, self.n_obj, self.sim_dt, self.drag],
                    device=self.device,
                )
            else:
                wp.launch(
                    newton_import_ir._apply_drag_correction,
                    dim=self.n_obj,
                    inputs=[self.state_in.particle_q, self.state_in.particle_qd, self.n_obj, self.sim_dt, self.drag],
                    device=self.device,
                )

        self.global_substep += 1
        self.sim_time += self.sim_dt
        if self.global_substep >= self.total_substeps:
            self.finished = True

        if self.profile_enabled:
            wp.synchronize_device(self.device)
            self._profile_run_samples.setdefault("total_substep", []).append(
                1000.0 * (time.perf_counter() - t_sub)
            )

    def step(self) -> None:
        if self.pending_reset:
            self._reset_runtime()
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self.model.particle_radius.assign(self._physical_particle_radius)
        for _ in range(self.steps_per_render):
            if self.finished:
                break
            self._advance_one_substep()
        wp.synchronize_device(self.device)
        self.last_step_wall_ms = 1000.0 * (time.perf_counter() - t0)
        if self.profile_enabled:
            self._profile_run_samples.setdefault("total_step", []).append(self.last_step_wall_ms)
        self.frame_index += 1

    def render(self) -> None:
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self.model.particle_radius.assign(self._visual_particle_radius)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_in)
        self.viewer.end_frame()
        self.model.particle_radius.assign(self._physical_particle_radius)
        wp.synchronize_device(self.device)
        self.last_render_wall_ms = 1000.0 * (time.perf_counter() - t0)

    def _reset_runtime(self) -> None:
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = (
            self.model.contacts()
            if newton_import_ir._use_collision_pipeline(self.cfg, self.ir)
            else None
        )
        self.global_substep = 0
        self.sim_time = 0.0
        self.frame_index = 0
        self.finished = False
        self.pending_reset = False

    def render_ui(self, imgui) -> None:
        if not getattr(self.viewer, "ui", None) or not self.viewer.ui.is_available:
            return

        frame, sub = self._trajectory_progress()
        flags = imgui.WindowFlags_.no_resize.value
        imgui.set_next_window_size(imgui.ImVec2(420, 320), imgui.Cond_.first_use_ever)
        if imgui.begin("Rope Trajectory Viewer", flags=flags):
            imgui.text("Backend: Newton only")
            imgui.text(f"IR: {Path(self.args.ir).name}")
            imgui.text(f"runtime_device: {self.device}")
            imgui.text(f"trajectory frame: {frame}/{max(0, self.n_traj_frames - 1)}")
            imgui.text(f"segment substep: {sub}/{max(0, self.segment_substeps - 1)}")
            imgui.text(f"global_substep: {self.global_substep}/{self.total_substeps}")
            imgui.text(f"finished: {self.finished}")
            imgui.separator()
            imgui.text(f"sim_dt: {self.sim_dt:.2e}")
            imgui.text(f"steps_per_render: {self.steps_per_render}")
            imgui.text(f"sim time / viewer frame: {self.sim_dt * self.steps_per_render:.3e} s")
            imgui.text(f"last step wall time: {self.last_step_wall_ms:.2f} ms")
            imgui.text(f"last render wall time: {self.last_render_wall_ms:.2f} ms")
            changed, new_steps = imgui.slider_int(
                "Substeps per render",
                self.steps_per_render,
                1,
                max(1, self.segment_substeps),
            )
            if changed:
                self.steps_per_render = int(new_steps)
            if imgui.button("Reset Trajectory"):
                self.pending_reset = True
            imgui.text("Space: pause/resume")
            imgui.text("H: toggle UI, F: frame camera, ESC: quit")
        imgui.end()

    def test_final(self) -> None:
        pass

    def _summarize_profile_runs(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        op_names = list(PROFILE_OP_NAMES)
        summary: dict[str, Any] = {}
        for name in op_names:
            run_means = []
            all_calls = []
            counts = []
            for run in runs:
                samples = run["samples"].get(name, [])
                counts.append(len(samples))
                if samples:
                    run_means.append(float(np.mean(samples)))
                    all_calls.extend(samples)
            summary[name] = {
                "call_count_total": int(len(all_calls)),
                "call_count_per_run": counts,
                "run_mean_ms": run_means,
                "mean_of_run_means_ms": float(np.mean(run_means)) if run_means else 0.0,
                "std_of_run_means_ms": float(np.std(run_means)) if len(run_means) > 1 else 0.0,
                "mean_over_all_calls_ms": float(np.mean(all_calls)) if all_calls else 0.0,
                "std_over_all_calls_ms": float(np.std(all_calls)) if len(all_calls) > 1 else 0.0,
            }
        return summary

    def _rank_profile_ops(self, aggregate: dict[str, Any]) -> list[dict[str, Any]]:
        excluded = {"solver_step", "total_substep", "total_step"}
        ranking = []
        for name, stats in aggregate.items():
            if name in excluded:
                continue
            ranking.append(
                {
                    "op_name": str(name),
                    "mean_of_run_means_ms": float(stats["mean_of_run_means_ms"]),
                    "call_count_total": int(stats["call_count_total"]),
                }
            )
        ranking.sort(key=lambda item: item["mean_of_run_means_ms"], reverse=True)
        return ranking

    def _write_profile_outputs(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        json_path = (
            self.args.profile_json.resolve()
            if self.args.profile_json is not None
            else (self.args.out_dir / f"{self.args.prefix}_profile.json").resolve()
        )
        csv_path = (
            self.args.profile_csv.resolve()
            if self.args.profile_csv is not None
            else (self.args.out_dir / f"{self.args.prefix}_profile.csv").resolve()
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "op_name",
                    "call_count_total",
                    "mean_of_run_means_ms",
                    "std_of_run_means_ms",
                    "mean_over_all_calls_ms",
                    "std_over_all_calls_ms",
                ]
            )
            for name, stats in payload["aggregate"].items():
                writer.writerow(
                    [
                        name,
                        stats["call_count_total"],
                        stats["mean_of_run_means_ms"],
                        stats["std_of_run_means_ms"],
                        stats["mean_over_all_calls_ms"],
                        stats["std_over_all_calls_ms"],
                    ]
                )
        return json_path, csv_path

    def profile_only(self) -> dict[str, Any]:
        warmup_runs = max(0, int(self.args.profile_warmup_runs))
        prof_runs = max(1, int(self.args.profile_runs))

        def run_episode(record: bool) -> dict[str, Any]:
            self._reset_runtime()
            self.profile_enabled = record
            self._profile_run_samples = {}
            t0 = time.perf_counter()
            while not self.finished:
                self.step()
            wp.synchronize_device(self.device)
            wall_ms = 1000.0 * (time.perf_counter() - t0)
            samples = {k: [float(v) for v in vals] for k, vals in self._profile_run_samples.items()}
            self.profile_enabled = False
            return {
                "wall_ms": float(wall_ms),
                "sim_time_sec": float(self.sim_time),
                "viewer_frames": int(self.frame_index),
                "trajectory_frames": int(self.n_traj_frames),
                "total_substeps": int(self.total_substeps),
                "samples": samples,
            }

        for _ in range(warmup_runs):
            run_episode(record=False)

        runs = [run_episode(record=True) for _ in range(prof_runs)]
        aggregate = self._summarize_profile_runs(runs)
        payload = {
            "ir": str(self.args.ir),
            "runtime_device": self.device,
            "sim_dt": float(self.sim_dt),
            "segment_substeps": int(self.segment_substeps),
            "steps_per_render": int(self.steps_per_render),
            "trajectory_frames": int(self.n_traj_frames),
            "total_substeps": int(self.total_substeps),
            "granular_solver_profile": True,
            "profile_runs": int(prof_runs),
            "warmup_runs": int(warmup_runs),
            "runs": runs,
            "aggregate": aggregate,
            "bottleneck_ranked_ops": self._rank_profile_ops(aggregate),
        }
        json_path, csv_path = self._write_profile_outputs(payload)
        print(f"Profile JSON: {json_path}", flush=True)
        print(f"Profile CSV: {csv_path}", flush=True)
        for name, stats in sorted(payload["aggregate"].items()):
            print(
                f"{name}: mean={stats['mean_of_run_means_ms']:.3f} ms, std={stats['std_of_run_means_ms']:.3f} ms, calls={stats['call_count_total']}",
                flush=True,
            )
        if payload["bottleneck_ranked_ops"]:
            top = payload["bottleneck_ranked_ops"][:5]
            print("Top bottlenecks:", flush=True)
            for item in top:
                print(
                    f"  {item['op_name']}: mean={item['mean_of_run_means_ms']:.3f} ms, calls={item['call_count_total']}",
                    flush=True,
                )
        return payload


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    wp.init()
    example = NewtonRopeControlViewer(viewer, args)
    if bool(args.profile_only):
        example.profile_only()
        if hasattr(viewer, "close"):
            viewer.close()
        return 0
    newton.examples.run(example, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
