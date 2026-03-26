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

import newton
import newton.examples
from newton._src.solvers.semi_implicit.kernels_body import eval_body_joint_forces
from newton._src.solvers.semi_implicit.kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from newton._src.solvers.semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import demo_cloth_bunny_drop_without_self_contact as off_case
import demo_cloth_box_drop_with_self_contact as on_case


DEFAULT_IR = WORKSPACE_ROOT / "Newton/phystwin_bridge/ir/blue_cloth_double_lift_around/phystwin_ir_v2_bf_strict.npz"
DEFAULT_TARGET_TOTAL_MASS = 0.1
DEFAULT_VBD_ITERATIONS = 15
DEFAULT_STEPS_PER_RENDER = 1
REQUIRED_RUNTIME_DEVICE = "cuda:0"


def create_parser() -> argparse.ArgumentParser:
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

    return parser


def _resolve_demo_module(mode: str):
    return off_case if mode == "off" else on_case


def _resolve_workspace_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    workspace_candidate = (WORKSPACE_ROOT / raw).resolve()
    return workspace_candidate


def _maybe_autoset_alpha(args: argparse.Namespace, raw_ir: dict[str, np.ndarray]) -> None:
    if args.mass_spring_scale is not None:
        return
    n_obj = int(np.asarray(raw_ir["num_object_points"]).ravel()[0])
    total_mass = float(np.asarray(raw_ir["mass"], dtype=np.float32)[:n_obj].sum())
    if total_mass <= 0.0:
        raise ValueError(f"IR total object mass must be > 0, got {total_mass}")
    args.mass_spring_scale = float(args.target_total_mass) / total_mass


def _prepare_ir_for_runtime(module, args: argparse.Namespace) -> dict[str, Any]:
    raw_ir = module.load_ir(args.ir)
    _maybe_autoset_alpha(args, raw_ir)
    if args.shape_contact_scale is None:
        args.shape_contact_scale = float(args.mass_spring_scale)
    ir_obj = module._copy_object_only_ir(raw_ir, args)
    if "contact_collision_dist" in ir_obj:
        scaled_dist = float(np.asarray(ir_obj["contact_collision_dist"]).ravel()[0]) * float(args.contact_dist_scale)
        if args.mode == "on":
            scaled_dist *= float(args.on_contact_dist_scale)
        ir_obj["contact_collision_dist"] = np.asarray(scaled_dist, dtype=np.float32)
    return ir_obj


def _require_runtime_device(module, args: argparse.Namespace) -> str:
    requested = str(
        args.runtime_device or args.device or module.path_defaults.default_device()
    )
    resolved = module.newton_import_ir.resolve_device(requested)
    if resolved != REQUIRED_RUNTIME_DEVICE:
        raise RuntimeError(
            "This realtime viewer is pinned to the local RTX 4090 path and must run "
            f"on {REQUIRED_RUNTIME_DEVICE}. Requested device resolved to {resolved!r} "
            f"from {requested!r}. Re-run with --runtime-device {REQUIRED_RUNTIME_DEVICE}."
        )
    return resolved


class NewtonClothBunnyViewer:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.module = _resolve_demo_module(args.mode)
        self.args.ir = _resolve_workspace_path(self.args.ir)
        self.args.out_dir = _resolve_workspace_path(self.args.out_dir)
        self.args.out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "on":
            self.args.rigid_shape = "box"
            self.args.add_box = bool(getattr(self.args, "add_bunny", True))
            if self.args.prefix == "cloth_bunny_playground":
                self.args.prefix = "cloth_box_playground"
            if float(self.args.sim_dt) == 5.0e-5:
                self.args.sim_dt = 1.0 / 600.0
            if int(self.args.substeps) == 667:
                self.args.substeps = 10
            if float(self.args.rigid_mass) == 0.5:
                self.args.rigid_mass = 4000.0
            if float(self.args.body_ke) == 5.0e4:
                self.args.body_ke = 1.0e4
            if float(self.args.body_kd) == 5.0e2:
                self.args.body_kd = 1.0e-2

        self.module._validate_scaling_args(self.args)
        self.device = _require_runtime_device(self.module, self.args)
        self.ir_obj = _prepare_ir_for_runtime(self.module, self.args)
        self.model, self.meta, self.n_obj = self.module.build_model(self.ir_obj, self.args, self.device)

        add_rigid_shape = bool(getattr(self.args, "add_box", getattr(self.args, "add_bunny", True)))
        self.shape_contacts_enabled = bool(self.args.shape_contacts) and add_rigid_shape
        self.cfg = self.module.newton_import_ir.SimConfig(
            ir_path=self.args.ir.resolve(),
            out_dir=self.args.out_dir,
            output_prefix=self.args.prefix,
            spring_ke_scale=float(self.args.spring_ke_scale),
            spring_kd_scale=float(self.args.spring_kd_scale),
            angular_damping=float(self.args.angular_damping),
            friction_smoothing=float(self.args.friction_smoothing),
            enable_tri_contact=True,
            disable_particle_contact_kernel=(self.args.mode == "off"),
            shape_contacts=self.shape_contacts_enabled,
            add_ground_plane=bool(self.args.add_ground_plane),
            strict_physics_checks=False,
            apply_drag=bool(self.args.apply_drag),
            drag_damping_scale=float(self.args.drag_damping_scale),
            gravity=-float(self.args.gravity_mag),
            gravity_from_reverse_z=False,
            up_axis="Z",
            particle_contacts=True,
            device=self.device,
        )

        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            angular_damping=self.cfg.angular_damping,
            friction_smoothing=self.cfg.friction_smoothing,
            enable_tri_contact=self.cfg.enable_tri_contact,
        )
        self.collision_pipeline = None

        self.steps_per_render = max(1, int(self.args.steps_per_render))
        self.sim_dt = float(self.args.sim_dt)
        self.drag = 0.0
        if self.args.apply_drag and "drag_damping" in self.ir_obj:
            self.drag = float(self.module.newton_import_ir.ir_scalar(self.ir_obj, "drag_damping")) * float(
                self.args.drag_damping_scale
            )
        _, gravity_vec = self.module.newton_import_ir.resolve_gravity(self.cfg, self.ir_obj)
        gravity_norm = float(np.linalg.norm(gravity_vec))
        self.gravity_axis = None
        if bool(self.args.drag_ignore_gravity_axis) and gravity_norm > 1.0e-12:
            self.gravity_axis = (-np.asarray(gravity_vec, dtype=np.float32) / gravity_norm).astype(np.float32)

        self.particle_grid = self.model.particle_grid
        self.search_radius = 0.0
        if self.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.particle_grid.reserve(self.model.particle_count)
            self.search_radius = float(self.model.particle_max_radius) * 2.0 + float(self.model.particle_cohesion)
            self.search_radius = max(
                self.search_radius, float(getattr(self.module.newton_import_ir, "EPSILON", 1.0e-6))
            )

        self._physical_particle_radius = self.model.particle_radius.numpy().astype(np.float32, copy=False).copy()
        self._visual_particle_radius = np.minimum(
            self._physical_particle_radius * float(self.args.particle_radius_vis_scale),
            float(self.args.particle_radius_vis_min),
        ).astype(np.float32)

        self.pending_reset = False
        self.sim_time = 0.0
        self.frame_index = 0
        self.last_step_wall_ms = 0.0
        self.last_render_wall_ms = 0.0

        self._reset_runtime()
        self._configure_viewer()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        print("Newton-only realtime viewer controls:")
        print("- Space: pause / resume (built into ViewerGL)")
        print("- H: show / hide UI")
        print("- F: frame camera on model")
        print("- ESC: exit")

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
        state_in.clear_forces()

        if self.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.particle_grid.build(state_in.particle_q, radius=self.search_radius)

        if self.contacts is not None:
            self.model.collide(state_in, self.contacts)

        if not bool(self.args.decouple_shape_materials):
            self.solver.step(state_in, state_out, self.control, self.contacts, self.sim_dt)
        else:
            particle_f = state_in.particle_f if state_in.particle_count else None
            body_f = state_in.body_f if state_in.body_count else None
            body_f_work = body_f
            if body_f is not None and self.model.joint_count and self.control.joint_f is not None:
                body_f_work = wp.clone(body_f)

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
                wp.launch(
                    self.module.newton_import_ir._apply_drag_correction,
                    dim=self.n_obj,
                    inputs=[state_in.particle_q, state_in.particle_qd, self.n_obj, self.sim_dt, self.drag],
                    device=self.device,
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


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    wp.init()
    example = NewtonClothBunnyViewer(viewer, args)
    newton.examples.run(example, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
