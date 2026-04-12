#!/usr/bin/env python3
"""Newton-only real-time viewer for cloth-rigid demos.

This script converts the current two-stage offline workflow

    build_model() -> simulate() -> render_video()

into a single real-time loop:

    build_model() -> step() / render()

The backend is fully Newton. We do not call PhysTwin's Warp simulator,
PhysTwin trainer, or Gaussian-splatting renderer.

Usage examples
--------------

Stable OFF viewer (recommended):
    python demo_cloth_bunny_realtime_viewer.py --viewer gl

Quick validation without opening a window:
    python demo_cloth_bunny_realtime_viewer.py --viewer null --num-frames 5

Experimental ON viewer:
    python demo_cloth_bunny_realtime_viewer.py --mode on --viewer gl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

if __package__ in {None, ""}:
    DEMOS_DIR = Path(__file__).resolve().parent.parent
    if str(DEMOS_DIR) not in sys.path:
        sys.path.insert(0, str(DEMOS_DIR))

from bridge_bootstrap import newton
import newton.examples
from bridge_shared import BRIDGE_ROOT, apply_viewer_shape_colors
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

if __package__ in {None, ""}:
    import cloth_bunny.offline as off_case
    from cloth_bunny.profiling import (
        PROFILE_GROUPS,
        PROFILE_OP_NAMES,
        compute_group_shares,
        rank_profile_ops,
        summarize_profile_runs,
        write_profile_outputs,
    )
    from cloth_bunny.runtime import (
        REQUIRED_RUNTIME_DEVICE,
        build_example_runtime,
    )
else:
    from . import offline as off_case
    from .profiling import (
        PROFILE_GROUPS,
        PROFILE_OP_NAMES,
        compute_group_shares,
        rank_profile_ops,
        summarize_profile_runs,
        write_profile_outputs,
    )
    from .runtime import (
        REQUIRED_RUNTIME_DEVICE,
        build_example_runtime,
    )
import demo_cloth_box_drop_with_self_contact as on_case


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
DEFAULT_IR = WORKSPACE_ROOT / "Newton/phystwin_bridge/ir/blue_cloth_double_lift_around/phystwin_ir_v2_bf_strict.npz"
DEFAULT_TARGET_TOTAL_MASS = 0.1
DEFAULT_VBD_ITERATIONS = 15
DEFAULT_STEPS_PER_RENDER = 1


def _create_parser_impl() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Real-time Newton-only cloth-rigid viewer."

    parser.add_argument("--mode", choices=["off", "on"], default="off", help="Playground mode.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR, help="Path to PhysTwin strict IR.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WORKSPACE_ROOT / "tmp/interactive_playground",
        help="Kept for compatibility with the existing build_model helpers.",
    )
    parser.add_argument("--prefix", default="cloth_bunny_playground")
    parser.add_argument(
        "--runtime-device",
        default=REQUIRED_RUNTIME_DEVICE,
        help=(
            "Newton runtime device. This realtime viewer is pinned to the RTX 4090 "
            f"workstation path and must run on {REQUIRED_RUNTIME_DEVICE}."
        ),
    )

    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sim-dt", type=float, default=5.0e-5)
    parser.add_argument("--substeps", type=int, default=667)
    parser.add_argument(
        "--steps-per-render",
        type=int,
        default=DEFAULT_STEPS_PER_RENDER,
        help=(
            "Simulation steps executed before each viewer frame. Keep this small "
            "(for example 1-8) to preserve the validated per-step dt while making "
            "the viewer responsive. Increasing it speeds up simulated time but also "
            "reintroduces the offline-style long stall before each draw."
        ),
    )
    parser.add_argument("--gravity-mag", type=float, default=9.8)
    parser.add_argument("--drop-height", type=float, default=0.5)
    parser.add_argument("--cloth-shift-x", type=float, default=0.0)
    parser.add_argument("--cloth-shift-y", type=float, default=0.0)

    parser.add_argument("--object-mass", type=float, default=None)
    parser.add_argument("--target-total-mass", type=float, default=DEFAULT_TARGET_TOTAL_MASS)
    parser.add_argument(
        "--mass-spring-scale", type=float, default=None
    )
    parser.add_argument("--spring-ke-scale", type=float, default=1.0)
    parser.add_argument("--spring-kd-scale", type=float, default=1.0)

    parser.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drag-damping-scale", type=float, default=1.0)
    parser.add_argument("--drag-ignore-gravity-axis", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--shape-contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-bunny", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-ground-plane", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-particle-contact-kernel", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--contact-dist-scale", type=float, default=1.0)
    parser.add_argument(
        "--shape-contact-scale",
        type=float,
        default=None,
        help="If omitted, defaults to the same alpha as mass-spring-scale.",
    )
    parser.add_argument("--shape-contact-damping-multiplier", type=float, default=15.0)
    parser.add_argument("--ground-shape-contact-scale", type=float, default=None)
    parser.add_argument("--ground-shape-contact-damping-multiplier", type=float, default=None)
    parser.add_argument("--bunny-shape-contact-scale", type=float, default=None)
    parser.add_argument("--bunny-shape-contact-damping-multiplier", type=float, default=None)
    parser.add_argument(
        "--decouple-shape-materials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rigid-rigid support at baseline while particle-shape contact uses scaled materials.",
    )

    parser.add_argument("--rigid-mass", type=float, default=0.5)
    parser.add_argument("--rigid-shape", choices=["bunny", "box"], default="bunny")
    parser.add_argument("--body-mu", type=float, default=0.5)
    parser.add_argument("--body-ke", type=float, default=5.0e4)
    parser.add_argument("--body-kd", type=float, default=5.0e2)
    parser.add_argument("--bunny-scale", type=float, default=0.12)
    parser.add_argument("--bunny-asset", default="bunny.usd")
    parser.add_argument("--bunny-prim", default="/root/bunny")
    parser.add_argument(
        "--bunny-quat-xyzw",
        type=float,
        nargs=4,
        default=(0.70710678, 0.0, 0.0, 0.70710678),
        metavar=("X", "Y", "Z", "W"),
    )
    parser.add_argument("--box-hx", type=float, default=0.12)
    parser.add_argument("--box-hy", type=float, default=0.12)
    parser.add_argument("--box-hz", type=float, default=0.08)
    parser.add_argument("--dynamic-box", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
    parser.add_argument("--vbd-tri-ke", type=float, default=1.0e4)
    parser.add_argument("--vbd-tri-ka", type=float, default=1.0e4)
    parser.add_argument("--vbd-tri-kd", type=float, default=1.0e-1)
    parser.add_argument("--vbd-edge-ke", type=float, default=1.0e4)
    parser.add_argument("--vbd-edge-kd", type=float, default=1.0e0)
    parser.add_argument("--self-contact-radius-scale", type=float, default=0.35)
    parser.add_argument("--self-contact-margin-scale", type=float, default=0.6)
    parser.add_argument("--soft-contact-margin-scale", type=float, default=1.0)

    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument("--friction-smoothing", type=float, default=1.0)

    parser.add_argument("--on-contact-dist-scale", type=float, default=1.0)
    parser.add_argument("--particle-self-contact-scale", type=float, default=0.1)

    parser.add_argument("--camera-pos", type=float, nargs=3, default=(-1.55, 1.35, 1.18), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-pitch", type=float, default=-10.0)
    parser.add_argument("--camera-yaw", type=float, default=-40.0)
    parser.add_argument("--camera-fov", type=float, default=55.0)
    parser.add_argument("--render-springs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spring-stride", type=int, default=8)
    parser.add_argument("--particle-radius-vis-scale", type=float, default=2.5)
    parser.add_argument("--particle-radius-vis-min", type=float, default=0.005)
    parser.add_argument(
        "--profile-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable rendering and run repeated step-only profiling episodes.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["throughput", "attribution"],
        default="attribution",
        help=(
            "throughput: keep the normal solver path and only measure end-to-end no-render wall time. "
            "attribution: enable synchronized per-op instrumentation for bottleneck attribution."
        ),
    )
    parser.add_argument(
        "--freeze-profile-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When profiling mode=on, keep the caller-provided dt/substeps/rigid parameters "
            "instead of applying the interactive ON-mode auto-retune."
        ),
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=5,
        help="Number of measured no-render profiling episodes.",
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=1,
        help="Number of warmup episodes excluded from the final statistics.",
    )
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


def create_parser() -> argparse.ArgumentParser:
    return _create_parser_impl()


def _resolve_demo_module(mode: str):
    return off_case if mode == "off" else on_case


class ClothBunnyExample:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.module = _resolve_demo_module(args.mode)
        runtime_assets = build_example_runtime(
            self.module,
            self.args,
            workspace_root=WORKSPACE_ROOT,
            required_runtime_device=REQUIRED_RUNTIME_DEVICE,
        )

        self.device = runtime_assets.device
        self.ir_obj = runtime_assets.ir_obj
        self.model = runtime_assets.model
        self.meta = runtime_assets.meta
        self.n_obj = runtime_assets.n_obj
        self.cfg = runtime_assets.cfg
        self.solver = runtime_assets.solver
        self.collision_pipeline = None

        self.steps_per_render = max(1, int(self.args.steps_per_render))
        self.sim_dt = float(self.args.sim_dt)
        self.drag = runtime_assets.drag
        self.gravity_axis = runtime_assets.gravity_axis
        self.particle_grid = runtime_assets.particle_grid
        self.search_radius = runtime_assets.search_radius
        self._physical_particle_radius = runtime_assets.physical_particle_radius
        self._visual_particle_radius = runtime_assets.visual_particle_radius

        self.pending_reset = False
        self.sim_time = 0.0
        self.frame_index = 0
        self.last_step_wall_ms = 0.0
        self.last_render_wall_ms = 0.0
        self.profile_enabled = False
        self.profile_mode = str(self.args.profile_mode)
        self._profile_run_samples: dict[str, list[float]] = {}

        self._reset_runtime()
        self._configure_viewer()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        print("Newton-only realtime viewer controls:")
        print("- Space: pause / resume (built into ViewerGL)")
        print("- H: show / hide UI")
        print("- F: frame camera on model")
        print("- ESC: exit")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        return _create_parser_impl()

    def _profile_call(self, name: str, fn, *args, **kwargs):
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        wp.synchronize_device(self.device)
        self._profile_run_samples.setdefault(name, []).append(1000.0 * (time.perf_counter() - t0))
        return result

    def _configure_viewer(self) -> None:
        self.viewer.set_model(self.model)
        for attr, value in (
            ("show_particles", True),
            ("show_triangles", True),
            ("show_visual", True),
            ("show_static", True),
            ("show_collision", True),
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
        try:
            apply_viewer_shape_colors(self.viewer, self.model)
        except Exception:
            pass

    def _reset_runtime(self) -> None:
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = (
            self.model.contacts()
            if self.module.newton_import_ir._use_collision_pipeline(
                self.cfg, self.ir_obj
            )
            else None
        )
        self.sim_time = 0.0
        self.frame_index = 0
        self.pending_reset = False

    def _step_substep(self, state_in, state_out):
        if self.profile_enabled:
            wp.synchronize_device(self.device)
            t_sub = time.perf_counter()

        state_in.clear_forces()

        if self.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                if self.profile_enabled:
                    self._profile_call(
                        "particle_grid_build",
                        self.particle_grid.build,
                        state_in.particle_q,
                        radius=self.search_radius,
                    )
                else:
                    self.particle_grid.build(state_in.particle_q, radius=self.search_radius)

        if self.contacts is not None:
            if self.profile_enabled:
                self._profile_call("model_collide", self.model.collide, state_in, self.contacts)
            else:
                self.model.collide(state_in, self.contacts)

        if not bool(self.args.decouple_shape_materials):
            if self.profile_enabled:
                self._profile_call(
                    "solver_step",
                    self.solver.step,
                    state_in,
                    state_out,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )
            else:
                self.solver.step(state_in, state_out, self.control, self.contacts, self.sim_dt)
        else:
            particle_f = state_in.particle_f if state_in.particle_count else None
            body_f = state_in.body_f if state_in.body_count else None
            body_f_work = body_f
            if body_f is not None and self.model.joint_count and self.control.joint_f is not None:
                body_f_work = wp.clone(body_f)

            if self.profile_enabled:
                self._profile_call("eval_spring_forces", eval_spring_forces, self.model, state_in, particle_f)
                self._profile_call("eval_triangle_forces", eval_triangle_forces, self.model, state_in, self.control, particle_f)
                self._profile_call("eval_bending_forces", eval_bending_forces, self.model, state_in, particle_f)
                self._profile_call("eval_tetrahedra_forces", eval_tetrahedra_forces, self.model, state_in, self.control, particle_f)
                self._profile_call(
                    "eval_body_joint_forces",
                    eval_body_joint_forces,
                    self.model,
                    state_in,
                    self.control,
                    body_f_work,
                    self.solver.joint_attach_ke,
                    self.solver.joint_attach_kd,
                )
                if self.args.mode == "on":
                    self._profile_call("eval_particle_contact_forces", eval_particle_contact_forces, self.model, state_in, particle_f)
                if self.solver.enable_tri_contact:
                    self._profile_call("eval_triangle_contact_forces", eval_triangle_contact_forces, self.model, state_in, particle_f)

                self.module._assign_shape_material_triplet(
                    self.model,
                    self.meta["shape_material_ke_base"],
                    self.meta["shape_material_kd_base"],
                    self.meta["shape_material_kf_base"],
                )
                self._profile_call(
                    "eval_body_contact_forces",
                    eval_body_contact_forces,
                    self.model,
                    state_in,
                    self.contacts,
                    friction_smoothing=self.solver.friction_smoothing,
                    body_f_out=body_f_work,
                )

                self.module._assign_shape_material_triplet(
                    self.model,
                    self.meta["shape_material_ke_scaled"],
                    self.meta["shape_material_kd_scaled"],
                    self.meta["shape_material_kf_scaled"],
                )
                self._profile_call(
                    "eval_particle_body_contact_forces",
                    eval_particle_body_contact_forces,
                    self.model,
                    state_in,
                    self.contacts,
                    particle_f,
                    body_f_work,
                    body_f_in_world_frame=False,
                )

                self._profile_call(
                    "integrate_particles",
                    self.solver.integrate_particles,
                    self.model,
                    state_in,
                    state_out,
                    self.sim_dt,
                )
                if body_f_work is body_f:
                    self._profile_call(
                        "integrate_bodies",
                        self.solver.integrate_bodies,
                        self.model,
                        state_in,
                        state_out,
                        self.sim_dt,
                        self.solver.angular_damping,
                    )
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_work
                    self._profile_call(
                        "integrate_bodies",
                        self.solver.integrate_bodies,
                        self.model,
                        state_in,
                        state_out,
                        self.sim_dt,
                        self.solver.angular_damping,
                    )
                    state_in.body_f = body_f_prev
            else:
                eval_spring_forces(self.model, state_in, particle_f)
                eval_triangle_forces(self.model, state_in, self.control, particle_f)
                eval_bending_forces(self.model, state_in, particle_f)
                eval_tetrahedra_forces(self.model, state_in, self.control, particle_f)
                eval_body_joint_forces(
                    self.model, state_in, self.control, body_f_work, self.solver.joint_attach_ke, self.solver.joint_attach_kd
                )
                if self.args.mode == "on":
                    eval_particle_contact_forces(self.model, state_in, particle_f)
                if self.solver.enable_tri_contact:
                    eval_triangle_contact_forces(self.model, state_in, particle_f)

                self.module._assign_shape_material_triplet(
                    self.model,
                    self.meta["shape_material_ke_base"],
                    self.meta["shape_material_kd_base"],
                    self.meta["shape_material_kf_base"],
                )
                eval_body_contact_forces(
                    self.model,
                    state_in,
                    self.contacts,
                    friction_smoothing=self.solver.friction_smoothing,
                    body_f_out=body_f_work,
                )

                self.module._assign_shape_material_triplet(
                    self.model,
                    self.meta["shape_material_ke_scaled"],
                    self.meta["shape_material_kd_scaled"],
                    self.meta["shape_material_kf_scaled"],
                )
                eval_particle_body_contact_forces(
                    self.model, state_in, self.contacts, particle_f, body_f_work, body_f_in_world_frame=False
                )

                self.solver.integrate_particles(self.model, state_in, state_out, self.sim_dt)
                if body_f_work is body_f:
                    self.solver.integrate_bodies(
                        self.model, state_in, state_out, self.sim_dt, self.solver.angular_damping
                    )
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_work
                    self.solver.integrate_bodies(
                        self.model, state_in, state_out, self.sim_dt, self.solver.angular_damping
                    )
                    state_in.body_f = body_f_prev

        state_in, state_out = state_out, state_in

        if self.drag > 0.0:
            if self.gravity_axis is not None:
                if self.profile_enabled:
                    self._profile_call(
                        "drag_correction",
                        wp.launch,
                        self.module._apply_drag_correction_ignore_axis,
                        dim=self.n_obj,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.n_obj,
                            self.sim_dt,
                            self.drag,
                            wp.vec3(*self.gravity_axis.tolist()),
                        ],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        self.module._apply_drag_correction_ignore_axis,
                        dim=self.n_obj,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.n_obj,
                            self.sim_dt,
                            self.drag,
                            wp.vec3(*self.gravity_axis.tolist()),
                        ],
                        device=self.device,
                    )
            else:
                if self.profile_enabled:
                    self._profile_call(
                        "drag_correction",
                        wp.launch,
                        self.module.newton_import_ir._apply_drag_correction,
                        dim=self.n_obj,
                        inputs=[state_in.particle_q, state_in.particle_qd, self.n_obj, self.sim_dt, self.drag],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        self.module.newton_import_ir._apply_drag_correction,
                        dim=self.n_obj,
                        inputs=[state_in.particle_q, state_in.particle_qd, self.n_obj, self.sim_dt, self.drag],
                        device=self.device,
                    )

        if self.profile_enabled:
            wp.synchronize_device(self.device)
            self._profile_run_samples.setdefault("total_substep", []).append(
                1000.0 * (time.perf_counter() - t_sub)
            )

        return state_in, state_out

    def step(self) -> None:
        if self.pending_reset:
            self._reset_runtime()

        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self.model.particle_radius.assign(self._physical_particle_radius)
        state_in = self.state_in
        state_out = self.state_out
        for _ in range(self.steps_per_render):
            state_in, state_out = self._step_substep(state_in, state_out)
        self.state_in, self.state_out = state_in, state_out
        wp.synchronize_device(self.device)
        self.last_step_wall_ms = 1000.0 * (time.perf_counter() - t0)
        if self.profile_enabled:
            self._profile_run_samples.setdefault("total_step", []).append(self.last_step_wall_ms)
        self.sim_time += self.sim_dt * self.steps_per_render
        self.frame_index += 1

    def step_no_sync(self) -> None:
        if self.pending_reset:
            self._reset_runtime()

        self.model.particle_radius.assign(self._physical_particle_radius)
        state_in = self.state_in
        state_out = self.state_out
        for _ in range(self.steps_per_render):
            state_in, state_out = self._step_substep(state_in, state_out)
        self.state_in, self.state_out = state_in, state_out
        self.sim_time += self.sim_dt * self.steps_per_render
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

    def render_ui(self, imgui) -> None:
        if not getattr(self.viewer, "ui", None) or not self.viewer.ui.is_available:
            return

        flags = imgui.WindowFlags_.no_resize.value
        imgui.set_next_window_size(imgui.ImVec2(390, 320), imgui.Cond_.first_use_ever)
        if imgui.begin("Newton Viewer", flags=flags):
            imgui.text(f"Backend: Newton only")
            imgui.text(f"Mode: {self.args.mode.upper()}")
            imgui.text(f"IR: {Path(self.args.ir).name}")
            imgui.text(f"Rigid: {self.args.rigid_shape}, mass={self.args.rigid_mass:.3f} kg")
            imgui.text(f"Cloth total mass target: {float(self.args.target_total_mass):.3f} kg")
            imgui.text(f"mass-spring-scale: {float(self.args.mass_spring_scale):.6e}")
            imgui.text(f"shape-contact-scale: {float(self.args.shape_contact_scale):.6e}")
            imgui.text(f"sim_dt: {float(self.sim_dt):.2e}")
            imgui.text(f"steps_per_render: {self.steps_per_render}")
            imgui.text(f"sim time / viewer frame: {self.sim_dt * self.steps_per_render:.3e} s")
            if self.args.mode == "on":
                imgui.text("Solver: SemiImplicit")
                imgui.text(f"self_contact_radius_scale: {float(self.args.self_contact_radius_scale):.3f}")
                imgui.text(f"self_contact_margin_scale: {float(self.args.self_contact_margin_scale):.3f}")
            imgui.separator()
            imgui.text(f"sim_time: {self.sim_time:.4f} s")
            imgui.text(f"frame_index: {self.frame_index}")
            imgui.text(f"last step wall time: {self.last_step_wall_ms:.2f} ms")
            imgui.text(f"last render wall time: {self.last_render_wall_ms:.2f} ms")
            changed, new_steps = imgui.slider_int("Substeps per render", self.steps_per_render, 1, max(1, int(self.args.substeps)))
            if changed:
                self.steps_per_render = int(new_steps)
            if imgui.button("Reset Scene"):
                self.pending_reset = True
            imgui.text("Space: pause/resume")
            imgui.text("H: toggle UI, F: frame camera, ESC: quit")
        imgui.end()

    def test_final(self):
        pass

    def profile_only(self) -> dict[str, Any]:
        num_frames = int(getattr(self.args, "num_frames", 0) or 0)
        if num_frames <= 0:
            num_frames = max(1, int(self.args.frames))
        warmup_runs = max(0, int(self.args.profile_warmup_runs))
        prof_runs = max(1, int(self.args.profile_runs))

        def run_episode(record: bool) -> dict[str, Any]:
            self._reset_runtime()
            attribution_mode = record and self.profile_mode == "attribution"
            self.profile_enabled = attribution_mode
            self._profile_run_samples = {}
            wp.synchronize_device(self.device)
            t0 = time.perf_counter()
            for _ in range(num_frames):
                if attribution_mode:
                    self.step()
                else:
                    self.step_no_sync()
            wp.synchronize_device(self.device)
            wall_ms = 1000.0 * (time.perf_counter() - t0)
            samples = {k: [float(v) for v in vals] for k, vals in self._profile_run_samples.items()}
            self.profile_enabled = False
            return {
                "wall_ms": float(wall_ms),
                "sim_time_sec": float(self.sim_time),
                "frame_index": int(self.frame_index),
                "samples": samples,
            }

        for _ in range(warmup_runs):
            run_episode(record=False)

        runs = [run_episode(record=True) for _ in range(prof_runs)]
        aggregate = summarize_profile_runs(runs, op_names=PROFILE_OP_NAMES) if self.profile_mode == "attribution" else {}
        bottleneck_ranking = rank_profile_ops(aggregate) if aggregate else []
        group_shares = compute_group_shares(aggregate, groups=PROFILE_GROUPS) if aggregate else {}
        payload = {
            "mode": str(self.args.mode),
            "viewer": str(getattr(self.args, "viewer", "unknown")),
            "profile_mode": str(self.profile_mode),
            "freeze_profile_config": bool(self.args.freeze_profile_config),
            "num_frames": int(num_frames),
            "steps_per_render": int(self.steps_per_render),
            "sim_dt": float(self.sim_dt),
            "substeps": int(self.args.substeps),
            "rigid_mass": float(self.args.rigid_mass),
            "body_ke": float(self.args.body_ke),
            "body_kd": float(self.args.body_kd),
            "warmup_runs": int(warmup_runs),
            "profile_runs": int(prof_runs),
            "runs": runs,
            "aggregate": aggregate,
            "bottleneck_ranking": bottleneck_ranking,
            "group_shares": group_shares,
        }
        json_path, csv_path = write_profile_outputs(self.args, payload)
        print(f"Profile JSON: {json_path}", flush=True)
        print(f"Profile CSV: {csv_path}", flush=True)
        if aggregate:
            for rank in bottleneck_ranking[:5]:
                print(
                    f"top op: {rank['op_name']} mean={rank['mean_of_run_means_ms']:.3f} ms calls={rank['call_count_total']}",
                    flush=True,
                )
            print(f"group shares: {group_shares}", flush=True)
        return payload


def main() -> int:
    parser = ClothBunnyExample.create_parser()
    viewer, args = newton.examples.init(parser)
    wp.init()
    example = ClothBunnyExample(viewer, args)
    if bool(args.profile_only):
        example.profile_only()
        if hasattr(viewer, "close"):
            viewer.close()
        return 0
    newton.examples.run(example, args)
    return 0


NewtonClothBunnyViewer = ClothBunnyExample


if __name__ == "__main__":
    raise SystemExit(main())
