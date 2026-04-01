#!/usr/bin/env python3
"""Import a PhysTwin IR into Newton and run a semi-implicit simulation.

IR requirements
---------------
This importer intentionally targets **IR v2+** exported by `export_ir.py` and
requires an explicit per-particle `collision_radius`. We do **not** support
legacy heuristics where a generic `radius` field could ambiguously mean either
topology radius or contact radius.

Architecture
------------
The pipeline has four phases, each handled by a dedicated function:

1. **load_ir** → raw IR dict from disk
2. **build_model** → Newton ``Model`` from IR (particles, springs, ground, …)
3. **simulate** → rollout loop producing per-frame positions
4. **save_results** → ``.npz`` + ``.json`` outputs

Extensibility: ``build_model`` returns a standard ``newton.Model``.  To add
new objects (rigid bodies, meshes, other particle systems), extend
``build_model`` with additional ``builder.add_*`` calls before ``finalize()``.
They will automatically interact through Newton's collision system.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

DEMOS_DIR = Path(__file__).resolve().parents[2] / "demos"
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

import newton
from path_defaults import bridge_root, default_device

# ═══════════════════════════════════════════════════════════════════════════
# Constants – no magic numbers anywhere else in this file.
# ═══════════════════════════════════════════════════════════════════════════

# IR
IR_VERSION_KEY = "ir_version"
IR_COLLISION_RADIUS_KEY = "collision_radius"

# Numerical tolerances
EPSILON = 1e-12  # safe-division floor
STRICT_KE_REL_TOL = 1e-4  # max relative error for ke consistency
KINEMATIC_MASS_TOL = 1e-8  # controllers must have |mass| below this
RESTITUTION_EPS = 1e-4  # avoid log(0) in restitution -> damping-ratio mapping

# Contact clamping
MU_MAX = 2.0
RESTITUTION_MAX = 1.0

# Ground restitution mapping modes
GROUND_RESTITUTION_MODE_STRICT_NATIVE = "strict-native"
GROUND_RESTITUTION_MODE_APPROXIMATE_NATIVE = "approximate-native"

# Bridge self-contact modes
SELF_CONTACT_MODE_OFF = "off"
SELF_CONTACT_MODE_NATIVE = "native"
SELF_CONTACT_MODE_CUSTOM = "custom"
SELF_CONTACT_MODE_PHYSTWIN = "phystwin"
SELF_CONTACT_MODES = (
    SELF_CONTACT_MODE_OFF,
    SELF_CONTACT_MODE_NATIVE,
    SELF_CONTACT_MODE_CUSTOM,
    SELF_CONTACT_MODE_PHYSTWIN,
)


def _load_self_contact_kernels():
    from self_contact_bridge_kernels import (
        apply_velocity_update_from_force,
        build_filtered_self_contact_tables,
        eval_filtered_self_contact_forces,
        eval_filtered_self_contact_phystwin_velocity,
    )

    return {
        "apply_velocity_update_from_force": apply_velocity_update_from_force,
        "build_filtered_self_contact_tables": build_filtered_self_contact_tables,
        "eval_filtered_self_contact_forces": eval_filtered_self_contact_forces,
        "eval_filtered_self_contact_phystwin_velocity": eval_filtered_self_contact_phystwin_velocity,
    }


def _load_semiimplicit_bridge_kernels():
    from semiimplicit_bridge_kernels import (
        eval_bending_forces,
        eval_spring_forces,
        eval_tetrahedra_forces,
        eval_triangle_contact_forces,
        eval_triangle_forces,
    )

    return {
        "eval_bending_forces": eval_bending_forces,
        "eval_spring_forces": eval_spring_forces,
        "eval_tetrahedra_forces": eval_tetrahedra_forces,
        "eval_triangle_contact_forces": eval_triangle_contact_forces,
        "eval_triangle_forces": eval_triangle_forces,
    }


def _load_phystwin_contact_stack():
    from phystwin_contact_stack import (
        build_strict_phystwin_contact_context,
        is_strict_phystwin_mode,
        step_strict_phystwin_contact_stack,
        validate_strict_phystwin_mode,
    )

    return {
        "build_strict_phystwin_contact_context": build_strict_phystwin_contact_context,
        "is_strict_phystwin_mode": is_strict_phystwin_mode,
        "step_strict_phystwin_contact_stack": step_strict_phystwin_contact_stack,
        "validate_strict_phystwin_mode": validate_strict_phystwin_mode,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimConfig:
    """All simulation parameters in one typed, inspectable object.

    Created once from CLI args; threaded through every function that needs it.
    No function reads ``argparse.Namespace`` directly.
    """

    # I/O
    ir_path: Path = Path(".")
    out_dir: Path = Path(".")
    output_prefix: str = "newton_rollout"

    # Spring tuning
    spring_ke_scale: float = 1.0
    spring_kd_scale: float = 1.0

    # Solver
    angular_damping: float = 0.05
    friction_smoothing: float = 1.0
    enable_tri_contact: bool = True
    # Default to disabling Newton's particle-particle contact kernel: it is expensive and
    # not PhysTwin-aligned, so enabling it can easily break parity.
    disable_particle_contact_kernel: bool = True

    # Simulation
    num_frames: int = 20
    substeps_per_frame: int | None = None
    sim_dt: float | None = None

    # Gravity
    gravity: float | None = None
    gravity_mag: float = 9.8
    gravity_from_reverse_z: bool = True
    up_axis: str = "Z"

    # Device
    device: str = "cuda:0"

    # Contacts
    shape_contacts: bool = False
    add_ground_plane: bool = True
    self_contact_mode: str | None = None
    custom_self_contact_hops: int = 0
    # Override for IR `self_collision` (which we treat as the PhysTwin collision enable).
    # Note: enabling particle self-collision in SemiImplicit also requires enabling the
    # particle contact kernel (see `disable_particle_contact_kernel`).
    particle_contacts: bool | None = None
    particle_contact_radius: float = 1e-5
    object_contact_radius: float | None = None
    particle_contact_ke: float | None = None
    particle_contact_kf_scale: float = 1.0
    shape_contact_scale: float | None = None
    shape_contact_damping_multiplier: float = 1.0

    # Controls
    interpolate_controls: bool = True

    # Validation
    strict_physics_checks: bool = True

    # Drag
    # Bridge-side PhysTwin drag emulation (applied to spring-mass object particles only).
    apply_drag: bool = True
    drag_damping_scale: float = 1.0

    # Friction / restitution
    particle_mu_override: float | None = None
    ground_mu_scale: float = 1.0
    ground_restitution_scale: float = 1.0
    ground_restitution_mode: str = GROUND_RESTITUTION_MODE_STRICT_NATIVE

    # Baseline
    inference_path: Path | None = None

    # Optional two-way interaction probe (importer-native CLI mode).
    rigid_probe: bool = False
    rigid_probe_object_mass: float = 1.0
    rigid_probe_drop_controller_springs: bool = True
    rigid_probe_freeze_controllers: bool = True
    rigid_probe_mass: float = 5.0
    rigid_probe_inertia_diag: float = 0.05
    rigid_probe_hx: float = 0.02
    rigid_probe_hy: float = 0.02
    rigid_probe_hz: float = 0.02
    rigid_probe_offset: tuple[float, float, float] = (-0.18, 0.0, 0.02)
    rigid_probe_velocity: tuple[float, float, float] = (1.2, 0.0, 0.0)
    rigid_probe_mu: float = 0.4
    rigid_probe_ke: float = 5.0e4
    rigid_probe_kd: float = 5.0e2
    rigid_probe_shape: str = "box"  # choices: box | bunny
    rigid_probe_bunny_scale: float = 0.12
    rigid_probe_bunny_asset: str = "bunny.usd"
    rigid_probe_bunny_prim: str = "/root/bunny"
    rigid_probe_bunny_edge_stride: int = 40
    rigid_probe_use_scene_gravity: bool = False
    rigid_probe_add_ground_plane: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SimConfig:
        return cls(
            ir_path=args.ir.resolve(),
            out_dir=args.out_dir.resolve(),
            output_prefix=args.output_prefix,
            spring_ke_scale=args.spring_ke_scale,
            spring_kd_scale=args.spring_kd_scale,
            angular_damping=args.angular_damping,
            friction_smoothing=args.friction_smoothing,
            enable_tri_contact=args.enable_tri_contact,
            disable_particle_contact_kernel=args.disable_particle_contact_kernel,
            num_frames=args.num_frames,
            substeps_per_frame=args.substeps_per_frame,
            sim_dt=args.sim_dt,
            gravity=args.gravity,
            gravity_mag=args.gravity_mag,
            gravity_from_reverse_z=args.gravity_from_reverse_z,
            up_axis=args.up_axis,
            device=args.device,
            shape_contacts=args.shape_contacts,
            add_ground_plane=args.add_ground_plane,
            self_contact_mode=args.self_contact_mode,
            custom_self_contact_hops=args.custom_self_contact_hops,
            particle_contacts=args.particle_contacts,
            particle_contact_radius=args.particle_contact_radius,
            object_contact_radius=args.object_contact_radius,
            particle_contact_ke=args.particle_contact_ke,
            particle_contact_kf_scale=args.particle_contact_kf_scale,
            shape_contact_scale=args.shape_contact_scale,
            shape_contact_damping_multiplier=args.shape_contact_damping_multiplier,
            interpolate_controls=args.interpolate_controls,
            strict_physics_checks=args.strict_physics_checks,
            apply_drag=args.apply_drag,
            drag_damping_scale=args.drag_damping_scale,
            particle_mu_override=args.particle_mu_override,
            ground_mu_scale=args.ground_mu_scale,
            ground_restitution_scale=args.ground_restitution_scale,
            ground_restitution_mode=args.ground_restitution_mode,
            inference_path=args.inference,
            rigid_probe=args.rigid_probe,
            rigid_probe_object_mass=args.rigid_probe_object_mass,
            rigid_probe_drop_controller_springs=args.rigid_probe_drop_controller_springs,
            rigid_probe_freeze_controllers=args.rigid_probe_freeze_controllers,
            rigid_probe_mass=args.rigid_probe_mass,
            rigid_probe_inertia_diag=args.rigid_probe_inertia_diag,
            rigid_probe_hx=args.rigid_probe_hx,
            rigid_probe_hy=args.rigid_probe_hy,
            rigid_probe_hz=args.rigid_probe_hz,
            rigid_probe_offset=tuple(args.rigid_probe_offset),
            rigid_probe_velocity=tuple(args.rigid_probe_velocity),
            rigid_probe_mu=args.rigid_probe_mu,
            rigid_probe_ke=args.rigid_probe_ke,
            rigid_probe_kd=args.rigid_probe_kd,
            rigid_probe_shape=args.rigid_probe_shape,
            rigid_probe_bunny_scale=args.rigid_probe_bunny_scale,
            rigid_probe_bunny_asset=args.rigid_probe_bunny_asset,
            rigid_probe_bunny_prim=args.rigid_probe_bunny_prim,
            rigid_probe_bunny_edge_stride=args.rigid_probe_bunny_edge_stride,
            rigid_probe_use_scene_gravity=args.rigid_probe_use_scene_gravity,
            rigid_probe_add_ground_plane=args.rigid_probe_add_ground_plane,
        )


@dataclass
class ModelResult:
    """Everything produced by ``build_model``."""

    model: newton.Model
    radius: np.ndarray
    ir_version: int
    self_contact_mode: str = SELF_CONTACT_MODE_OFF
    bridge_self_contact_grid: Any = None
    bridge_neighbor_table: Any = None
    bridge_neighbor_count: Any = None
    phystwin_contact_context: Any = None
    checks: dict = field(default_factory=dict)


@dataclass
class SimResult:
    """Everything produced by ``simulate``."""

    particle_q_all: np.ndarray  # [frames+1, N, 3]
    particle_q_object: np.ndarray  # [frames+1, n_obj, 3]
    wall_time_sec: float


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import a PhysTwin IR into Newton (SolverSemiImplicit)."
    )

    # I/O
    p.add_argument("--ir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--output-prefix", default="newton_rollout")

    # Spring tuning
    p.add_argument("--spring-ke-scale", type=float, default=1.0)
    p.add_argument("--spring-kd-scale", type=float, default=1.0)

    # Solver
    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)
    p.add_argument(
        "--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--disable-particle-contact-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force-disable particle-particle contact kernel in SemiImplicit by setting "
            "model.particle_grid=None. Useful for debugging and to avoid costly/unstable "
            "particle self-collisions. Default: disabled (use --no-disable-particle-contact-kernel to enable)."
        ),
    )

    # Simulation
    p.add_argument("--num-frames", type=int, default=20)
    p.add_argument("--substeps-per-frame", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=None)

    # Gravity
    p.add_argument("--gravity", type=float, default=None)
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument(
        "--gravity-from-reverse-z", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")

    # Device
    p.add_argument("--device", default=default_device())

    # Contacts
    p.add_argument(
        "--shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable particle-vs-shape contacts (e.g. ground plane) by running Newton "
            "collision detection each substep. Ground-plane contacts also enable this "
            "pipeline automatically when --add-ground-plane is set."
        ),
    )
    p.add_argument(
        "--add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add a ground plane shape to the model. Particle-vs-plane forces require running the "
            "collision pipeline; this importer enables that pipeline automatically when a ground "
            "plane is present (or when --shape-contacts / IR self_collision is enabled)."
        ),
    )
    p.add_argument(
        "--self-contact-mode",
        choices=list(SELF_CONTACT_MODES),
        default=None,
        help=(
            "Optional explicit self-contact mode. "
            "`off` disables particle self-contact, "
            "`native` uses Newton's built-in particle kernel, "
            "`custom` uses bridge-side filtered penalty self-contact, "
            "`phystwin` uses the strict bridge-side PhysTwin contact stack "
            "(pairwise self-collision + implicit z=0 ground plane only)."
        ),
    )
    p.add_argument(
        "--custom-self-contact-hops",
        type=int,
        default=0,
        help=(
            "Graph-hop exclusion radius for bridge-side self-contact tables. "
            "0 keeps all non-self pairs eligible; 1 excludes direct spring neighbors."
        ),
    )
    p.add_argument(
        "--particle-contacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override IR self_collision (PhysTwin collision enable). "
            "When enabled and `--no-disable-particle-contact-kernel` is set, we rebuild the "
            "HashGrid each substep to support SemiImplicit particle-particle contacts (expensive). "
            "If the kernel remains disabled, this flag only affects radius mapping/metadata."
        ),
    )
    p.add_argument(
        "--particle-contact-radius",
        type=float,
        default=1e-5,
        help=(
            "Controller particle radius used for contacts. Default is tiny to avoid "
            "kinematic controllers acting like 'bulldozers' in collisions."
        ),
    )
    p.add_argument(
        "--object-contact-radius",
        type=float,
        default=None,
        help="Optional override for object particle collision radius (does not affect controllers).",
    )
    p.add_argument(
        "--particle-contact-ke",
        type=float,
        default=None,
        help=(
            "Optional override for Newton particle contact stiffness (model.particle_ke) when "
            "mapping PhysTwin collide_object_* parameters."
        ),
    )
    p.add_argument(
        "--particle-contact-kf-scale",
        type=float,
        default=1.0,
        help=(
            "Scales Newton particle tangential friction stiffness: particle_kf = "
            "particle_contact_kf_scale * particle_kd."
        ),
    )
    p.add_argument(
        "--shape-contact-scale",
        type=float,
        default=None,
        help=(
            "Scale factor applied to the actual particle-shape contact chain used by "
            "SemiImplicit: model.soft_contact_ke and model.shape_material_ke. "
            "Use together with low-mass experiments when trying to preserve baseline contact behavior."
        ),
    )
    p.add_argument(
        "--shape-contact-damping-multiplier",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier applied on top of --shape-contact-scale for kd/kf terms in the "
            "particle-shape contact chain. Values >1 are useful when minimizing rollout RMSE under "
            "discrete penalty contact."
        ),
    )

    # Controls / validation / drag
    p.add_argument(
        "--interpolate-controls", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--strict-physics-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable PhysTwin-parity validation (missing parity fields cause errors). "
            "Use --no-strict-physics-checks for explicit native-Newton experimentation."
        ),
    )
    p.add_argument(
        "--apply-drag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply PhysTwin-style drag damping on spring-mass object particles only. "
            "Use --no-apply-drag to disable."
        ),
    )
    p.add_argument("--drag-damping-scale", type=float, default=1.0)

    # Friction / restitution
    p.add_argument("--particle-mu-override", type=float, default=None)
    p.add_argument("--ground-mu-scale", type=float, default=1.0)
    p.add_argument("--ground-restitution-scale", type=float, default=1.0)
    p.add_argument(
        "--ground-restitution-mode",
        choices=[
            GROUND_RESTITUTION_MODE_STRICT_NATIVE,
            GROUND_RESTITUTION_MODE_APPROXIMATE_NATIVE,
        ],
        default=GROUND_RESTITUTION_MODE_STRICT_NATIVE,
        help=(
            "How to handle PhysTwin ground collide_elas in native Newton SemiImplicit. "
            "'strict-native' keeps native Newton ground contact defaults and treats "
            "collide_elas as unsupported for parity. "
            "'approximate-native' calibrates native ground damping from collide_elas "
            "without changing Newton core."
        ),
    )

    # Baseline
    p.add_argument("--inference", type=Path, default=None)

    # Optional rigid-body two-way probe (integrated in importer CLI).
    p.add_argument(
        "--rigid-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run built-in two-way interaction probe mode: imported spring-mass object "
            "vs one native Newton rigid object (box or bunny). "
            "Outputs probe diagnostics instead of parity baseline files."
        ),
    )
    p.add_argument("--rigid-probe-object-mass", type=float, default=1.0)
    p.add_argument(
        "--rigid-probe-drop-controller-springs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop springs connected to controller particles in probe mode.",
    )
    p.add_argument(
        "--rigid-probe-freeze-controllers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze controller particles at frame-0 targets in probe mode.",
    )
    p.add_argument("--rigid-probe-mass", type=float, default=5.0)
    p.add_argument("--rigid-probe-inertia-diag", type=float, default=0.05)
    p.add_argument("--rigid-probe-hx", type=float, default=0.02)
    p.add_argument("--rigid-probe-hy", type=float, default=0.02)
    p.add_argument("--rigid-probe-hz", type=float, default=0.02)
    p.add_argument(
        "--rigid-probe-offset",
        type=float,
        nargs=3,
        default=(-0.18, 0.0, 0.02),
        metavar=("DX", "DY", "DZ"),
    )
    p.add_argument(
        "--rigid-probe-velocity",
        type=float,
        nargs=3,
        default=(1.2, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
    )
    p.add_argument("--rigid-probe-mu", type=float, default=0.4)
    p.add_argument("--rigid-probe-ke", type=float, default=5e4)
    p.add_argument("--rigid-probe-kd", type=float, default=5e2)
    p.add_argument(
        "--rigid-probe-shape",
        choices=["box", "bunny"],
        default="box",
        help="Rigid probe shape type.",
    )
    p.add_argument("--rigid-probe-bunny-scale", type=float, default=0.12)
    p.add_argument("--rigid-probe-bunny-asset", type=str, default="bunny.usd")
    p.add_argument("--rigid-probe-bunny-prim", type=str, default="/root/bunny")
    p.add_argument("--rigid-probe-bunny-edge-stride", type=int, default=40)
    p.add_argument(
        "--rigid-probe-use-scene-gravity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use resolve_gravity(cfg, ir) in rigid probe mode instead of zero gravity.",
    )
    p.add_argument(
        "--rigid-probe-add-ground-plane",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add Newton ground plane in rigid probe mode (table).",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def _write_kinematic_state(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    target_q: wp.array(dtype=wp.vec3),
    target_qd: wp.array(dtype=wp.vec3),
):
    """Scatter-write controller positions and velocities into state arrays."""
    tid = wp.tid()
    i = indices[tid]
    particle_q[i] = target_q[tid]
    particle_qd[i] = target_qd[tid]


@wp.kernel
def _write_kinematic_state_precomputed(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    target_q_all: wp.array(dtype=wp.vec3),
    target_base: int,
):
    """Scatter-write precomputed controller positions with zero controller velocity."""
    tid = wp.tid()
    i = indices[tid]
    particle_q[i] = target_q_all[target_base + tid]
    particle_qd[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _apply_drag_correction(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    n_object: int,
    dt: float,
    damping: float,
):
    """PhysTwin drag: ``v ← v·exp(-dt·k)`` with position compensation.

    Newton integrates position with the *undamped* velocity, so we subtract
    the overshoot ``v·dt·(1-exp(-dt·k))`` from the position.
    """
    tid = wp.tid()
    if tid >= n_object:
        return
    v = particle_qd[tid]
    scale = wp.exp(-dt * damping)
    particle_q[tid] = particle_q[tid] - v * dt * (1.0 - scale)
    particle_qd[tid] = v * scale


# ═══════════════════════════════════════════════════════════════════════════
# IR helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_ir(path: Path) -> dict[str, np.ndarray]:
    """Load an IR ``.npz`` file into a plain dict."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def ir_scalar(ir: dict, key: str, default: float | None = None) -> float:
    """Extract a scalar value from the IR, with optional default."""
    if key not in ir:
        if default is not None:
            return default
        raise KeyError(f"IR missing required key: {key!r}")
    return float(np.asarray(ir[key]).ravel()[0])


def ir_bool(ir: dict, key: str, default: bool = False) -> bool:
    if key not in ir:
        return default
    return bool(np.asarray(ir[key]).ravel()[0])


def ir_string(ir: dict, key: str, default: str = "") -> str:
    if key not in ir:
        return default
    return str(np.asarray(ir[key]).ravel()[0])


def ir_version(ir: dict) -> int:
    return int(ir_scalar(ir, IR_VERSION_KEY, default=1))


def resolve_device(requested: str) -> str:
    """Return a valid Warp device string, falling back to CPU if unavailable."""
    try:
        wp.get_device(requested)
        return requested
    except Exception:
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Gravity
# ═══════════════════════════════════════════════════════════════════════════

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def resolve_gravity(
    cfg: SimConfig, ir: dict
) -> tuple[float, tuple[float, float, float]]:
    """Compute gravity scalar and 3D vector from config + IR."""
    if cfg.gravity is not None:
        scalar = cfg.gravity
    elif cfg.gravity_from_reverse_z:
        reverse_z = ir_bool(ir, "reverse_z")
        scalar = cfg.gravity_mag * (1.0 if reverse_z else -1.0)
    else:
        scalar = 0.0
    vec = [0.0, 0.0, 0.0]
    vec[AXIS_INDEX[cfg.up_axis]] = scalar
    return scalar, tuple(vec)


# ═══════════════════════════════════════════════════════════════════════════
# Physics validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_config_ranges(cfg: SimConfig) -> None:
    """Fail fast on invalid scale/damping parameters."""
    checks = {
        ("--spring-ke-scale", cfg.spring_ke_scale, ">", 0.0),
        ("--spring-kd-scale", cfg.spring_kd_scale, ">=", 0.0),
        ("--angular-damping", cfg.angular_damping, ">=", 0.0),
        ("--friction-smoothing", cfg.friction_smoothing, ">", 0.0),
        ("--ground-mu-scale", cfg.ground_mu_scale, ">=", 0.0),
        ("--ground-restitution-scale", cfg.ground_restitution_scale, ">=", 0.0),
        ("--particle-contact-kf-scale", cfg.particle_contact_kf_scale, ">=", 0.0),
    }
    for name, value, op, bound in checks:
        if op == ">" and value <= bound:
            raise ValueError(f"{name} must be > {bound}, got {value}")
        if op == ">=" and value < bound:
            raise ValueError(f"{name} must be >= {bound}, got {value}")
    if cfg.particle_contact_ke is not None and cfg.particle_contact_ke <= 0.0:
        raise ValueError(
            f"--particle-contact-ke must be > 0 when provided, got {cfg.particle_contact_ke}"
        )
    if cfg.shape_contact_scale is not None and cfg.shape_contact_scale <= 0.0:
        raise ValueError(
            f"--shape-contact-scale must be > 0 when provided, got {cfg.shape_contact_scale}"
        )
    if cfg.shape_contact_damping_multiplier <= 0.0:
        raise ValueError(
            "--shape-contact-damping-multiplier must be > 0, "
            f"got {cfg.shape_contact_damping_multiplier}"
        )
    if cfg.rigid_probe:
        if cfg.rigid_probe_object_mass <= 0.0:
            raise ValueError(
                f"--rigid-probe-object-mass must be > 0, got {cfg.rigid_probe_object_mass}"
            )
        if cfg.rigid_probe_mass <= 0.0:
            raise ValueError(f"--rigid-probe-mass must be > 0, got {cfg.rigid_probe_mass}")
        if cfg.rigid_probe_inertia_diag <= 0.0:
            raise ValueError(
                f"--rigid-probe-inertia-diag must be > 0, got {cfg.rigid_probe_inertia_diag}"
            )
        if cfg.rigid_probe_shape == "box":
            if (
                cfg.rigid_probe_hx <= 0.0
                or cfg.rigid_probe_hy <= 0.0
                or cfg.rigid_probe_hz <= 0.0
            ):
                raise ValueError(
                    "Rigid probe box half extents must be > 0 "
                    f"(got hx={cfg.rigid_probe_hx}, hy={cfg.rigid_probe_hy}, hz={cfg.rigid_probe_hz})"
                )
        elif cfg.rigid_probe_shape == "bunny":
            if cfg.rigid_probe_bunny_scale <= 0.0:
                raise ValueError(
                    f"--rigid-probe-bunny-scale must be > 0, got {cfg.rigid_probe_bunny_scale}"
                )
            if cfg.rigid_probe_bunny_edge_stride < 1:
                raise ValueError(
                    f"--rigid-probe-bunny-edge-stride must be >= 1, got {cfg.rigid_probe_bunny_edge_stride}"
                )
        else:
            raise ValueError(
                f"--rigid-probe-shape must be one of ['box','bunny'], got {cfg.rigid_probe_shape!r}"
            )


def validate_ir_physics(ir: dict, cfg: SimConfig) -> dict:
    """Validate IR spring semantics.  Returns a diagnostics dict."""
    _validate_config_ranges(cfg)
    strict = cfg.strict_physics_checks
    checks: dict = {}

    # Springs
    edges = np.asarray(ir["spring_edges"])
    n = int(edges.shape[0])
    ke = np.asarray(ir["spring_ke"], dtype=np.float64).ravel()
    if ke.shape[0] != n:
        raise ValueError(f"spring_ke length {ke.shape[0]} != edge count {n}")
    if not np.all(np.isfinite(ke)):
        raise ValueError("Non-finite spring_ke values.")

    # Rest lengths
    rest = None
    if "spring_rest_length" in ir:
        rest = np.asarray(ir["spring_rest_length"], dtype=np.float64).ravel()
        if rest.shape[0] != n:
            raise ValueError(f"spring_rest_length length {rest.shape[0]} != {n}")
        if not np.all(np.isfinite(rest)):
            raise ValueError("Non-finite spring_rest_length.")
        if np.any(rest <= 0.0):
            raise ValueError(f"Non-positive rest length (min={float(np.min(rest))})")
        checks["spring_rest_min"] = float(np.min(rest))
        checks["spring_rest_max"] = float(np.max(rest))
    elif strict:
        raise ValueError("Strict mode requires spring_rest_length.")

    # ke mode
    ke_mode = ir_string(ir, "spring_ke_mode", "unknown")
    checks["spring_ke_mode"] = ke_mode
    if strict and ke_mode != "y_over_rest":
        raise ValueError(
            f"Strict mode requires spring_ke_mode=y_over_rest, got {ke_mode!r}"
        )

    # Cross-check ke ≈ Y / rest
    if "spring_y" in ir and rest is not None:
        y = np.asarray(ir["spring_y"], dtype=np.float64).ravel()
        if y.shape[0] != n:
            raise ValueError(f"spring_y length {y.shape[0]} != {n}")
        expected = y / np.maximum(rest, EPSILON)
        rel_err = np.abs(ke - expected) / np.maximum(np.abs(expected), EPSILON)
        checks["ke_rel_error_max"] = float(np.max(rel_err))
        checks["ke_rel_error_mean"] = float(np.mean(rel_err))
        if strict and float(np.max(rel_err)) > STRICT_KE_REL_TOL:
            raise ValueError(
                f"ke inconsistent with Y/rest (max_rel_err={float(np.max(rel_err)):.2e})"
            )
    elif strict:
        raise ValueError("Strict mode requires spring_y and spring_rest_length.")

    # Controller mass = 0
    if "controller_idx" in ir:
        idx = np.asarray(ir["controller_idx"], dtype=np.int64).ravel()
        mass = np.asarray(ir["mass"], dtype=np.float64).ravel()
        if idx.size > 0:
            if idx.min() < 0 or idx.max() >= mass.shape[0]:
                raise ValueError("controller_idx out of bounds.")
            max_mass = float(np.max(np.abs(mass[idx])))
        else:
            max_mass = 0.0
        checks["controller_mass_abs_max"] = max_mass
        if strict and max_mass > KINEMATIC_MASS_TOL:
            raise ValueError(
                f"Controllers must have mass≈0, got max|mass|={max_mass:.2e}"
            )

    if "self_collision" in ir:
        checks["self_collision"] = ir_bool(ir, "self_collision")

    # Ground-plane parity fields (PhysTwin uses an implicit z=0 plane with
    # collide_elas/collide_fric and reverse_z).
    if "reverse_z" in ir:
        checks["reverse_z"] = ir_bool(ir, "reverse_z")
    elif strict:
        raise ValueError("Strict mode requires reverse_z for ground-plane orientation.")

    if "contact_collide_elas" in ir:
        ground_e = ir_scalar(ir, "contact_collide_elas")
        if not np.isfinite(ground_e):
            raise ValueError(f"Non-finite contact_collide_elas: {ground_e}")
        checks["ground_contact_elas_raw"] = float(ground_e)
    elif strict:
        raise ValueError(
            "Strict mode requires contact_collide_elas (PhysTwin ground restitution)."
        )

    if "contact_collide_fric" in ir:
        ground_mu = ir_scalar(ir, "contact_collide_fric")
        if not np.isfinite(ground_mu):
            raise ValueError(f"Non-finite contact_collide_fric: {ground_mu}")
        checks["ground_contact_fric_raw"] = float(ground_mu)
    elif strict:
        raise ValueError(
            "Strict mode requires contact_collide_fric (PhysTwin ground friction)."
        )

    # PhysTwin particle-particle collision coefficients (exported even when kernel disabled).
    if "contact_collide_object_elas" in ir:
        obj_e = ir_scalar(ir, "contact_collide_object_elas")
        if not np.isfinite(obj_e):
            raise ValueError(f"Non-finite contact_collide_object_elas: {obj_e}")
        checks["object_contact_elas_raw"] = float(obj_e)
    elif strict:
        raise ValueError(
            "Strict mode requires contact_collide_object_elas in IR."
        )

    if "contact_collide_object_fric" in ir:
        obj_mu = ir_scalar(ir, "contact_collide_object_fric")
        if not np.isfinite(obj_mu):
            raise ValueError(f"Non-finite contact_collide_object_fric: {obj_mu}")
        checks["object_contact_fric_raw"] = float(obj_mu)
    elif strict:
        raise ValueError(
            "Strict mode requires contact_collide_object_fric in IR."
        )

    return checks


# ═══════════════════════════════════════════════════════════════════════════
# Collision radius resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_collision_radius(ir: dict, n: int) -> tuple[np.ndarray, int, str]:
    """Return per-particle collision radii from IR.

    We intentionally require IR v2+ and the explicit `collision_radius` field to
    avoid ambiguous legacy heuristics (`radius` could mean topology radius vs
    contact radius).
    """
    ver = ir_version(ir)
    if ver < 2:
        raise ValueError(
            f"IR v2+ required (got ir_version={ver}). Re-export using export_ir.py."
        )
    if IR_COLLISION_RADIUS_KEY not in ir:
        raise KeyError(f"IR missing required key: {IR_COLLISION_RADIUS_KEY!r}")
    r = np.asarray(ir[IR_COLLISION_RADIUS_KEY], dtype=np.float32).reshape(-1)
    if r.shape[0] != n:
        raise ValueError(f"collision_radius length {r.shape[0]} != particle count {n}")
    return r.copy(), ver, "collision_radius"


# ═══════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_particle_contacts(cfg: SimConfig, ir: dict) -> bool:
    if cfg.self_contact_mode is not None:
        return str(cfg.self_contact_mode).lower() != SELF_CONTACT_MODE_OFF
    if cfg.particle_contacts is not None:
        return cfg.particle_contacts
    return ir_bool(ir, "self_collision")


def _resolve_self_contact_mode(cfg: SimConfig, ir: dict) -> str:
    if cfg.self_contact_mode is not None:
        mode = str(cfg.self_contact_mode).lower()
        if mode not in SELF_CONTACT_MODES:
            raise ValueError(f"Unsupported self_contact_mode={cfg.self_contact_mode!r}")
        return mode

    if not _resolve_particle_contacts(cfg, ir):
        return SELF_CONTACT_MODE_OFF
    if not cfg.disable_particle_contact_kernel:
        return SELF_CONTACT_MODE_NATIVE
    return SELF_CONTACT_MODE_OFF


def _use_collision_pipeline(cfg: SimConfig, ir: dict) -> bool:
    """Decide whether to run Newton's shape-collision pipeline this rollout.

    PhysTwin always applies ground-plane collision in its integrator, independent
    of `self_collision`. To preserve parity, we must run the collision pipeline
    whenever a ground plane is present, not only when self-collision is enabled.
    """
    if _resolve_self_contact_mode(cfg, ir) == SELF_CONTACT_MODE_PHYSTWIN:
        # Strict bridge `phystwin` uses its own self-collision + implicit ground
        # plane handling and does not mix in Newton's shape collision pipeline.
        return False
    return bool(cfg.shape_contacts or cfg.add_ground_plane or _resolve_particle_contacts(cfg, ir))


def _add_particles(
    builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, particle_contacts: bool
) -> tuple[np.ndarray, int, str]:
    """Add IR particles to the builder.  Returns (radius, ir_ver, radius_source)."""
    x0 = ir["x0"].astype(np.float32, copy=False)
    v0 = ir["v0"].astype(np.float32, copy=False)
    mass = ir["mass"].astype(np.float32, copy=False)
    radius, ver, src = resolve_collision_radius(ir, x0.shape[0])
    flags = np.full(x0.shape[0], int(newton.ParticleFlags.ACTIVE), dtype=np.int32)

    # Map PhysTwin pairwise distance → Newton per-particle radius
    if "contact_collision_dist" in ir:
        dist = ir_scalar(ir, "contact_collision_dist")
        if particle_contacts:
            radius.fill(max(dist * 0.5, EPSILON))
            src = "contact_collision_dist_half"
        else:
            src = f"{src}_shape_preserved"
        if "controller_idx" in ir:
            ctrl = ir["controller_idx"].astype(np.int64, copy=False)
            if ctrl.size > 0:
                radius[ctrl] = cfg.particle_contact_radius

    if cfg.object_contact_radius is not None:
        if cfg.object_contact_radius <= 0.0:
            raise ValueError(
                f"object_contact_radius must be > 0, got {cfg.object_contact_radius}"
            )
        n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
        radius[:n_obj].fill(cfg.object_contact_radius)
        src = "object_contact_radius_override"

    builder.add_particles(
        pos=[tuple(row.tolist()) for row in x0],
        vel=[tuple(row.tolist()) for row in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
        flags=flags.astype(int).tolist(),
    )
    return radius, ver, src


def _add_springs(
    builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, checks: dict
) -> None:
    """Add IR springs (with scaled ke/kd) to the builder."""
    edges = np.asarray(ir["spring_edges"])
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected spring_edges shape [N, 2], got {edges.shape}")
    n = edges.shape[0]

    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel() * cfg.spring_ke_scale
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel() * cfg.spring_kd_scale
    if ke.shape[0] != n or kd.shape[0] != n:
        raise ValueError(
            f"Spring param count mismatch: ke={ke.shape[0]}, kd={kd.shape[0]}, edges={n}"
        )

    rest = (
        ir["spring_rest_length"].astype(np.float32).ravel()
        if "spring_rest_length" in ir
        else None
    )

    for i in range(n):
        builder.add_spring(
            i=int(edges[i, 0]),
            j=int(edges[i, 1]),
            ke=float(ke[i]),
            kd=float(kd[i]),
            control=0.0,
        )
        if rest is not None:
            # `add_spring()` appends one spring; overwrite its rest length in-place.
            # This keeps spring force semantics aligned with PhysTwin/IR.
            builder.spring_rest_length[-1] = float(rest[i])

    checks["spring_ke_scale"] = cfg.spring_ke_scale
    checks["spring_kd_scale"] = cfg.spring_kd_scale


def _add_ground_plane(
    builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, checks: dict
) -> None:
    """Optionally add a ground plane from IR contact params."""
    if not cfg.add_ground_plane:
        checks["ground_plane_added"] = False
        return

    gcfg = builder.default_shape_cfg.copy()
    _configure_ground_contact_material(
        gcfg, ir, cfg, checks, context="ground_plane"
    )

    reverse_z = ir_bool(ir, "reverse_z")
    if cfg.up_axis == "Z" and reverse_z:
        xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(1, 0, 0, 0))
        builder.add_shape_plane(
            body=-1,
            xform=xform,
            width=0,
            length=0,
            cfg=gcfg,
            label="ground_plane_reverse_z",
        )
    else:
        builder.add_ground_plane(cfg=gcfg)

    checks["ground_plane_added"] = True
    checks["ground_mu"] = float(gcfg.mu)
    checks["ground_restitution"] = float(gcfg.restitution)
    checks["ground_contact_final_ke"] = float(gcfg.ke)
    checks["ground_contact_final_kd"] = float(gcfg.kd)
    checks["ground_contact_final_kf"] = float(gcfg.kf)


def _restitution_to_damping_ratio(restitution: float) -> float:
    """Map coefficient of restitution `e` to equivalent damping ratio `zeta`.

    For an underdamped 2nd-order system:
      e = exp(-zeta*pi/sqrt(1-zeta^2))
      => zeta = -ln(e)/sqrt(pi^2 + ln(e)^2)
    """
    e = float(np.clip(restitution, RESTITUTION_EPS, 1.0 - RESTITUTION_EPS))
    ln_e = float(np.log(e))
    return float(-ln_e / np.sqrt(np.pi * np.pi + ln_e * ln_e))


def _resolve_object_mass_reference(ir: dict) -> float:
    """Return a robust object-particle mass reference for contact mapping."""
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    mass = np.asarray(ir["mass"], dtype=np.float64).ravel()
    if mass.shape[0] < n_obj:
        raise ValueError(f"mass length {mass.shape[0]} < num_object_points {n_obj}")
    obj_mass = mass[:n_obj]
    positive = obj_mass[obj_mass > EPSILON]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _configure_ground_contact_material(
    gcfg,
    ir: dict,
    cfg: SimConfig,
    checks: dict,
    *,
    context: str,
) -> None:
    """Configure native Newton ground contact material from PhysTwin IR.

    Native Newton SemiImplicit uses `ke/kd/kf/mu` for particle-vs-shape contact.
    PhysTwin ground collision uses an impulse-style restitution/friction update.
    Therefore:
    - `strict-native` records collide_elas as unsupported for parity and keeps the
      native Newton ground `ke/kd/kf` defaults.
    - `approximate-native` calibrates native `kd` from collide_elas while keeping
      the native Newton contact model unchanged.
    """
    checks["ground_restitution_mode"] = cfg.ground_restitution_mode
    checks["ground_contact_param_source"] = "native_newton_shape_material"
    checks["ground_contact_context"] = context

    if "contact_collide_fric" in ir:
        fric_raw = ir_scalar(ir, "contact_collide_fric")
        gcfg.mu = float(np.clip(fric_raw * cfg.ground_mu_scale, 0.0, MU_MAX))
        checks["ground_contact_fric_raw"] = float(fric_raw)
    checks["ground_contact_map_mu"] = float(gcfg.mu)

    if "contact_collide_elas" not in ir:
        checks["ground_contact_elas_supported"] = False
        checks["ground_contact_map_ke"] = float(gcfg.ke)
        checks["ground_contact_map_kd"] = float(gcfg.kd)
        checks["ground_contact_map_kf"] = float(gcfg.kf)
        return

    elas_raw = ir_scalar(ir, "contact_collide_elas")
    elas_effective = float(
        np.clip(elas_raw * cfg.ground_restitution_scale, 0.0, RESTITUTION_MAX)
    )
    checks["ground_contact_elas_raw"] = float(elas_raw)
    checks["ground_contact_elas_effective"] = float(elas_effective)

    if cfg.ground_restitution_mode == GROUND_RESTITUTION_MODE_STRICT_NATIVE:
        checks["ground_contact_elas_supported"] = False
        checks["ground_contact_elas_mapping"] = "unsupported_in_native_semi_implicit"
        checks["ground_contact_map_ke"] = float(gcfg.ke)
        checks["ground_contact_map_kd"] = float(gcfg.kd)
        checks["ground_contact_map_kf"] = float(gcfg.kf)
        message = (
            f"{context}: PhysTwin ground collide_elas is present in IR, but native "
            "Newton SemiImplicit has no direct 1-to-1 restitution field for "
            "particle-vs-shape contacts. Ground bounce will therefore use native "
            "Newton ground ke/kd/kf defaults unless "
            "--ground-restitution-mode=approximate-native is selected."
        )
        checks["ground_contact_warning"] = message
        if cfg.strict_physics_checks:
            raise ValueError(message)
        warnings.warn(message)
        return

    # Approximate native calibration: use the native Newton contact model but tune
    # damping so the bounce trend follows the target restitution coefficient.
    m_eff_ref = max(_resolve_object_mass_reference(ir), EPSILON)
    zeta = _restitution_to_damping_ratio(elas_effective)
    ke = max(float(gcfg.ke), EPSILON)
    kd = 2.0 * zeta * np.sqrt(ke * m_eff_ref)
    gcfg.kd = float(max(kd, 0.0))

    checks["ground_contact_elas_supported"] = True
    checks["ground_contact_elas_mapping"] = "approximate_native_damping_calibration"
    checks["ground_contact_map_zeta"] = float(zeta)
    checks["ground_contact_map_m_eff_ref"] = float(m_eff_ref)
    checks["ground_contact_map_ke"] = float(gcfg.ke)
    checks["ground_contact_map_kd"] = float(gcfg.kd)
    checks["ground_contact_map_kf"] = float(gcfg.kf)


def _apply_ps_object_collision_mapping(
    model: newton.Model, ir: dict, cfg: SimConfig, checks: dict
) -> bool:
    """Map PhysTwin collide_object_{elas,fric} to Newton particle contact params.

    This is an approximation bridge:
    - PhysTwin uses impulse-style pairwise collision response.
    - Newton SemiImplicit uses penalty-force particle contacts.
    """
    has_elas = "contact_collide_object_elas" in ir
    has_fric = "contact_collide_object_fric" in ir
    if (not has_elas) and (not has_fric):
        return False
    if has_elas != has_fric:
        raise ValueError(
            "IR must provide both contact_collide_object_elas and "
            "contact_collide_object_fric together for PS object-contact mapping."
        )

    elas_raw = ir_scalar(ir, "contact_collide_object_elas")
    fric_raw = ir_scalar(ir, "contact_collide_object_fric")

    # 1) Friction coefficient maps directly.
    mu = float(np.clip(fric_raw, 0.0, MU_MAX))

    # 2) Restitution maps to damping ratio, then kd via critical damping form:
    #    kd = 2*zeta*sqrt(ke*m_eff_ref)
    zeta = _restitution_to_damping_ratio(elas_raw)
    ke = (
        float(cfg.particle_contact_ke)
        if cfg.particle_contact_ke is not None
        else float(model.particle_ke)
    )
    ke = max(ke, EPSILON)
    m_ref = _resolve_object_mass_reference(ir)
    m_eff_ref = max(0.5 * m_ref, EPSILON)  # equal-mass pair effective mass
    kd = 2.0 * zeta * np.sqrt(ke * m_eff_ref)
    kf = cfg.particle_contact_kf_scale * kd

    model.particle_ke = float(ke)
    model.particle_kd = float(max(kd, 0.0))
    model.particle_kf = float(max(kf, 0.0))
    if cfg.particle_mu_override is None:
        model.particle_mu = mu

    checks["particle_contact_param_source"] = "ps_object_collision_map"
    checks["contact_collide_object_elas_raw"] = float(elas_raw)
    checks["contact_collide_object_fric_raw"] = float(fric_raw)
    checks["particle_contact_map_zeta"] = float(zeta)
    checks["particle_contact_map_m_eff_ref"] = float(m_eff_ref)
    checks["particle_contact_map_ke"] = float(model.particle_ke)
    checks["particle_contact_map_kd"] = float(model.particle_kd)
    checks["particle_contact_map_kf"] = float(model.particle_kf)
    checks["particle_contact_map_mu"] = float(model.particle_mu)
    return True


def _apply_shape_contact_scaling(
    model: newton.Model, cfg: SimConfig, checks: dict
) -> None:
    """Scale the actual particle-shape soft-contact chain used by SemiImplicit.

    Newton's particle-vs-shape kernel does not read ``model.particle_ke/kd/kf``.
    Instead it averages:
    - ``model.soft_contact_ke/kd/kf``
    - ``model.shape_material_ke/kd/kf``

    Therefore low-mass experiments that aim to preserve the original contact
    regime must scale this chain explicitly.
    """
    if cfg.shape_contact_scale is None:
        checks["shape_contact_scale_applied"] = False
        return

    alpha = float(cfg.shape_contact_scale)
    damping_mult = float(cfg.shape_contact_damping_multiplier)

    checks["shape_contact_scale_applied"] = True
    checks["shape_contact_scale"] = alpha
    checks["shape_contact_damping_multiplier"] = damping_mult

    checks["soft_contact_ke_before"] = float(model.soft_contact_ke)
    checks["soft_contact_kd_before"] = float(model.soft_contact_kd)
    checks["soft_contact_kf_before"] = float(model.soft_contact_kf)

    model.soft_contact_ke = float(model.soft_contact_ke * alpha)
    model.soft_contact_kd = float(model.soft_contact_kd * alpha * damping_mult)
    model.soft_contact_kf = float(model.soft_contact_kf * alpha * damping_mult)

    checks["soft_contact_ke_after"] = float(model.soft_contact_ke)
    checks["soft_contact_kd_after"] = float(model.soft_contact_kd)
    checks["soft_contact_kf_after"] = float(model.soft_contact_kf)

    for arr_name in ("shape_material_ke", "shape_material_kd", "shape_material_kf"):
        arr = getattr(model, arr_name)
        vals = arr.numpy().astype(np.float32, copy=False).copy()
        checks[f"{arr_name}_before_mean"] = float(np.mean(vals)) if vals.size else None
        vals *= np.float32(alpha)
        if arr_name != "shape_material_ke":
            vals *= np.float32(damping_mult)
        arr.assign(vals)
        checks[f"{arr_name}_after_mean"] = float(np.mean(vals)) if vals.size else None


def build_model(ir: dict, cfg: SimConfig, device: str) -> ModelResult:
    """Build a complete Newton Model from IR data.

    This is the main extensibility point.  To add new objects to the
    simulation, add ``builder.add_body()`` / ``builder.add_shape_*()``
    calls before ``builder.finalize()``.
    """
    checks = validate_ir_physics(ir, cfg)
    particle_contacts = _resolve_particle_contacts(cfg, ir)
    self_contact_mode = _resolve_self_contact_mode(cfg, ir)
    phystwin_contact_stack = (
        _load_phystwin_contact_stack()
        if self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN
        else None
    )
    if phystwin_contact_stack is not None:
        phystwin_contact_stack["validate_strict_phystwin_mode"](
            cfg,
            ir,
            particle_contacts_enabled=particle_contacts,
        )

    builder = newton.ModelBuilder(
        up_axis=newton.Axis.from_any(cfg.up_axis),
        gravity=0.0,
    )

    radius, ver, radius_src = _add_particles(
        builder,
        ir,
        cfg,
        particle_contacts and (self_contact_mode != SELF_CONTACT_MODE_PHYSTWIN),
    )
    _add_springs(builder, ir, cfg, checks)
    if self_contact_mode != SELF_CONTACT_MODE_PHYSTWIN:
        _add_ground_plane(builder, ir, cfg, checks)

    # ── Extension point ──
    # Add new objects here: builder.add_body(), builder.add_shape_mesh(), etc.
    # They will interact with PhysTwin particles through Newton's collision.

    model = builder.finalize(device=device)

    # Particle contact kernel in SemiImplicit relies on a HashGrid being built each substep.
    # An explicit bridge self-contact mode overrides the legacy particle-contact flags.
    particle_contact_kernel = self_contact_mode == SELF_CONTACT_MODE_NATIVE
    bridge_self_contact_grid = None
    bridge_neighbor_table = None
    bridge_neighbor_count = None
    phystwin_contact_context = None
    if self_contact_mode == SELF_CONTACT_MODE_CUSTOM:
        self_contact_kernels = _load_self_contact_kernels()
        bridge_self_contact_grid = model.particle_grid
        model.particle_grid = None
        edges = np.asarray(ir.get("spring_edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
        (
            bridge_neighbor_table,
            bridge_neighbor_count,
            _,
            exclusion_summary,
        ) = self_contact_kernels["build_filtered_self_contact_tables"](
            edges,
            n_particles=int(np.asarray(ir["x0"]).shape[0]),
            hops=int(cfg.custom_self_contact_hops),
            device=device,
        )
        checks.update(
            {
                "custom_self_contact_hops": int(cfg.custom_self_contact_hops),
                "excluded_neighbor_min": float(exclusion_summary["excluded_neighbor_min"]),
                "excluded_neighbor_mean": float(exclusion_summary["excluded_neighbor_mean"]),
                "excluded_neighbor_median": float(exclusion_summary["excluded_neighbor_median"]),
                "excluded_neighbor_max": float(exclusion_summary["excluded_neighbor_max"]),
                "excluded_pair_count": float(exclusion_summary["excluded_pair_count"]),
            }
        )
    elif self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN:
        model.particle_grid = None
        assert phystwin_contact_stack is not None
        phystwin_contact_context, phystwin_checks = phystwin_contact_stack[
            "build_strict_phystwin_contact_context"
        ](model, ir, cfg, device=device)
        checks.update(phystwin_checks)
    elif not particle_contact_kernel:
        model.particle_grid = None

    # Gravity
    _, gravity_vec = resolve_gravity(cfg, ir)
    model.set_gravity(gravity_vec)

    # Optional mapping from PhysTwin object-collision params to Newton particle-contact params.
    if self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN:
        object_collision_mapped = False
    else:
        object_collision_mapped = _apply_ps_object_collision_mapping(model, ir, cfg, checks)
        _apply_shape_contact_scaling(model, cfg, checks)

    # Particle friction fallback/override.
    if self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN:
        pass
    elif cfg.particle_mu_override is not None:
        model.particle_mu = float(np.clip(cfg.particle_mu_override, 0, MU_MAX))
    elif (not object_collision_mapped) and "contact_collide_fric" in ir:
        model.particle_mu = float(
            np.clip(ir_scalar(ir, "contact_collide_fric"), 0, MU_MAX)
        )

    checks.setdefault("particle_contact_param_source", "newton_default")
    if self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN:
        checks.setdefault("particle_contact_final_ke", None)
        checks.setdefault("particle_contact_final_kd", None)
        checks.setdefault("particle_contact_final_kf", None)
        checks.setdefault("particle_contact_final_mu", None)
    else:
        checks["particle_contact_final_ke"] = float(model.particle_ke)
        checks["particle_contact_final_kd"] = float(model.particle_kd)
        checks["particle_contact_final_kf"] = float(model.particle_kf)
        checks["particle_contact_final_mu"] = float(model.particle_mu)
    checks["particle_contacts_enabled"] = particle_contact_kernel
    checks["self_contact_mode"] = self_contact_mode
    checks["collision_radius_source"] = radius_src

    return ModelResult(
        model=model,
        radius=radius,
        ir_version=ver,
        self_contact_mode=self_contact_mode,
        bridge_self_contact_grid=bridge_self_contact_grid,
        bridge_neighbor_table=bridge_neighbor_table,
        bridge_neighbor_count=bridge_neighbor_count,
        phystwin_contact_context=phystwin_contact_context,
        checks=checks,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Controller interpolation
# ═══════════════════════════════════════════════════════════════════════════


def interpolate_controller(
    traj: np.ndarray, frame: int, substep: int, n_substeps: int, interpolate: bool
) -> np.ndarray:
    """Linearly interpolate controller positions within a frame."""
    if not interpolate:
        return traj[frame]
    if frame == 0:
        return traj[0]
    alpha = (substep + 1) / n_substeps
    return traj[frame - 1] + (traj[frame] - traj[frame - 1]) * alpha


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════


def simulate(
    model: newton.Model,
    ir: dict,
    cfg: SimConfig,
    device: str,
    *,
    model_result: ModelResult | None = None,
) -> SimResult:
    """Run the semi-implicit simulation loop.  Returns per-frame positions."""
    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    # Shape contacts (including the ground plane) require running Newton's
    # collision pipeline each substep.
    collision_pipeline_enabled = _use_collision_pipeline(cfg, ir)
    contacts = model.contacts() if collision_pipeline_enabled else None

    # Controller arrays
    ctrl_idx = ir["controller_idx"].astype(np.int64)
    ctrl_traj = ir["controller_traj"].astype(np.float32)
    n_obj = int(ir["num_object_points"])
    has_controllers = ctrl_idx.size > 0

    if ctrl_idx.ndim != 1:
        raise ValueError(f"controller_idx must be 1D, got {ctrl_idx.shape}")
    if ctrl_traj.ndim != 3 or ctrl_traj.shape[-1] != 3:
        raise ValueError(f"controller_traj must be [F, C, 3], got {ctrl_traj.shape}")
    if ctrl_traj.shape[1] != ctrl_idx.size:
        raise ValueError(
            f"Controller count mismatch: traj={ctrl_traj.shape[1]}, idx={ctrl_idx.size}"
        )

    # GPU buffers for controllers
    ctrl_idx_wp = ctrl_target_wp = ctrl_vel_wp = None
    ctrl_vel_zero = None
    if has_controllers:
        ctrl_idx_wp = wp.array(ctrl_idx.astype(np.int32), dtype=wp.int32, device=device)
        ctrl_target_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
        ctrl_vel_wp = wp.empty(ctrl_idx.size, dtype=wp.vec3, device=device)
        ctrl_vel_zero = np.zeros((ctrl_idx.size, 3), dtype=np.float32)

    # Timing
    sim_dt = cfg.sim_dt if cfg.sim_dt is not None else ir_scalar(ir, "sim_dt")
    substeps = cfg.substeps_per_frame or int(ir_scalar(ir, "sim_substeps"))
    substeps = max(1, substeps)
    n_frames = max(1, min(cfg.num_frames, ctrl_traj.shape[0]))

    # Drag
    drag = 0.0
    if cfg.apply_drag and "drag_damping" in ir:
        drag = ir_scalar(ir, "drag_damping") * cfg.drag_damping_scale

    self_contact_mode = (
        model_result.self_contact_mode if model_result is not None else _resolve_self_contact_mode(cfg, ir)
    )
    use_native_self_contact = self_contact_mode == SELF_CONTACT_MODE_NATIVE
    use_custom_self_contact = self_contact_mode == SELF_CONTACT_MODE_CUSTOM
    use_phystwin_self_contact = self_contact_mode == SELF_CONTACT_MODE_PHYSTWIN
    use_manual_force_path = use_custom_self_contact or use_phystwin_self_contact
    self_contact_kernels = _load_self_contact_kernels() if use_custom_self_contact else {}
    semiimplicit_kernels = _load_semiimplicit_bridge_kernels() if use_custom_self_contact else {}
    phystwin_contact_stack = _load_phystwin_contact_stack() if use_phystwin_self_contact else {}

    particle_grid = (
        model.particle_grid
        if use_native_self_contact
        else (
            model_result.bridge_self_contact_grid
            if (use_custom_self_contact and model_result is not None)
            else None
        )
    )
    if particle_grid is not None:
        with wp.ScopedDevice(model.device):
            particle_grid.reserve(model.particle_count)
        search_radius = float(model.particle_max_radius) * 2.0 + float(model.particle_cohesion)
        # Prevent build errors for degenerate radii.
        search_radius = max(search_radius, float(EPSILON))
    else:
        search_radius = 0.0

    # Collect rollout
    rollout_all: list[np.ndarray] = []
    rollout_obj: list[np.ndarray] = []

    # Frame 0 is the initial state (matches PhysTwin inference indexing).
    q0 = state_in.particle_q.numpy().astype(np.float32)
    rollout_all.append(q0)
    rollout_obj.append(q0[:n_obj])
    frame_range = range(1, n_frames)

    # Main loop
    t0 = time.perf_counter()
    for frame in frame_range:
        for sub in range(substeps):
            # Newton state stores force accumulators; clear them every substep so forces
            # don't unintentionally persist across integration steps.
            state_in.clear_forces()

            if has_controllers:
                # PhysTwin "control points" are position-driven. We replicate that by
                # directly writing controller particle positions (and setting qd=0) so
                # spring forces become the implicit interaction channel.
                target = interpolate_controller(
                    ctrl_traj, frame, sub, substeps, cfg.interpolate_controls
                )
                ctrl_target_wp.assign(target.astype(np.float32, copy=False))
                ctrl_vel_wp.assign(ctrl_vel_zero)
                wp.launch(
                    _write_kinematic_state,
                    dim=ctrl_idx.size,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        ctrl_idx_wp,
                        ctrl_target_wp,
                        ctrl_vel_wp,
                    ],
                    device=device,
                )

            if particle_grid is not None:
                with wp.ScopedDevice(model.device):
                    particle_grid.build(state_in.particle_q, radius=search_radius)

            if collision_pipeline_enabled:
                assert contacts is not None
                model.collide(state_in, contacts)

            if not use_manual_force_path:
                solver.step(state_in, state_out, control, contacts, sim_dt)
            elif use_phystwin_self_contact:
                if model_result is None or model_result.phystwin_contact_context is None:
                    raise RuntimeError(
                        "Strict bridge `phystwin` mode requires a shared PhysTwin contact context."
                    )
                phystwin_contact_stack["step_strict_phystwin_contact_stack"](
                    model,
                    state_in,
                    state_out,
                    control,
                    model_result.phystwin_contact_context,
                    sim_dt=sim_dt,
                    joint_attach_ke=solver.joint_attach_ke,
                    joint_attach_kd=solver.joint_attach_kd,
                )
            else:
                particle_f = state_in.particle_f if state_in.particle_count else None
                semiimplicit_kernels["eval_spring_forces"](model, state_in, particle_f)
                semiimplicit_kernels["eval_triangle_forces"](model, state_in, control, particle_f)
                semiimplicit_kernels["eval_bending_forces"](model, state_in, particle_f)
                semiimplicit_kernels["eval_tetrahedra_forces"](model, state_in, control, particle_f)

                if model_result is None:
                    raise RuntimeError("Custom self-contact path requires bridge-side model metadata.")
                self_contact_kernels["eval_filtered_self_contact_forces"](
                    model,
                    state_in,
                    particle_f,
                    particle_grid,
                    model_result.bridge_neighbor_table,
                    model_result.bridge_neighbor_count,
                )

                if solver.enable_tri_contact:
                    semiimplicit_kernels["eval_triangle_contact_forces"](model, state_in, particle_f)
                solver.integrate_particles(model, state_in, state_out, sim_dt)

            state_in, state_out = state_out, state_in

            if drag > 0.0 and not use_phystwin_self_contact:
                # PhysTwin drag is a post-step velocity damping; Newton doesn't have
                # an equivalent built-in knob, so we apply it explicitly here.
                # Important: this is limited to the first n_obj spring-mass object
                # particles (controllers and any other non-object entities are untouched).
                wp.launch(
                    _apply_drag_correction,
                    dim=n_obj,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        n_obj,
                        sim_dt,
                        drag,
                    ],
                    device=device,
                )

        q = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q)
        rollout_obj.append(q[:n_obj])

    return SimResult(
        particle_q_all=np.stack(rollout_all),
        particle_q_object=np.stack(rollout_obj),
        wall_time_sec=time.perf_counter() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════


def _load_inference(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    import pickle

    with path.open("rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float32)


def _resolve_inference_path(cfg: SimConfig, ir: dict) -> Path | None:
    if cfg.inference_path is not None:
        return cfg.inference_path
    case = ir_string(ir, "case_name")
    if not case:
        return None
    cand = bridge_root() / "inputs" / "cases" / case / "inference.pkl"
    return cand if cand.exists() else None


def save_results(
    cfg: SimConfig, ir: dict, model_result: ModelResult, sim_result: SimResult
) -> dict:
    """Write ``.npz`` + ``.json`` and return the summary dict."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    n_obj = int(ir["num_object_points"])

    # Inference RMSE
    inference_path = _resolve_inference_path(cfg, ir)
    inference = _load_inference(inference_path)
    rmse_per_frame = mean_rmse = max_rmse = None
    compared = 0
    if inference is not None and inference.ndim == 3 and inference.shape[1] == n_obj:
        compared = min(sim_result.particle_q_object.shape[0], inference.shape[0])
        err = sim_result.particle_q_object[:compared] - inference[:compared]
        rmse_per_frame = np.sqrt(np.mean(err**2, axis=(1, 2))).astype(np.float32)
        mean_rmse = float(rmse_per_frame.mean())
        max_rmse = float(rmse_per_frame.max())

    npz_path = cfg.out_dir / f"{cfg.output_prefix}.npz"
    json_path = cfg.out_dir / f"{cfg.output_prefix}.json"

    np.savez_compressed(
        npz_path,
        particle_q_all=sim_result.particle_q_all,
        particle_q_object=sim_result.particle_q_object,
        sim_dt=np.float32(cfg.sim_dt or ir_scalar(ir, "sim_dt")),
        substeps_per_frame=np.int32(
            cfg.substeps_per_frame or int(ir_scalar(ir, "sim_substeps"))
        ),
        frames_run=np.int32(sim_result.particle_q_all.shape[0]),
        rmse_per_frame=(
            rmse_per_frame
            if rmse_per_frame is not None
            else np.zeros(0, dtype=np.float32)
        ),
    )

    gravity_scalar, gravity_vec = resolve_gravity(cfg, ir)
    summary = {
        "ir_path": str(cfg.ir_path),
        "output_npz": str(npz_path),
        "device_requested": cfg.device,
        "device_used": str(model_result.model.device),
        "ir_version": model_result.ir_version,
        "config": {
            "spring_ke_scale": cfg.spring_ke_scale,
            "spring_kd_scale": cfg.spring_kd_scale,
            "angular_damping": cfg.angular_damping,
            "friction_smoothing": cfg.friction_smoothing,
            "enable_tri_contact": cfg.enable_tri_contact,
            "interpolate_controls": cfg.interpolate_controls,
            "strict_physics_checks": cfg.strict_physics_checks,
            "apply_drag": cfg.apply_drag,
            "drag_damping_scale": cfg.drag_damping_scale,
            "shape_contacts": cfg.shape_contacts,
            "add_ground_plane": cfg.add_ground_plane,
            "self_contact_mode": cfg.self_contact_mode,
            "custom_self_contact_hops": cfg.custom_self_contact_hops,
            "particle_contacts_override": cfg.particle_contacts,
            "disable_particle_contact_kernel": cfg.disable_particle_contact_kernel,
            "particle_contact_radius": cfg.particle_contact_radius,
            "object_contact_radius": cfg.object_contact_radius,
            "particle_contact_ke": cfg.particle_contact_ke,
            "particle_contact_kf_scale": cfg.particle_contact_kf_scale,
            "particle_mu_override": cfg.particle_mu_override,
            "ground_mu_scale": cfg.ground_mu_scale,
            "ground_restitution_scale": cfg.ground_restitution_scale,
            "ground_restitution_mode": cfg.ground_restitution_mode,
        },
        "simulation": {
            "frames_run": int(sim_result.particle_q_all.shape[0]),
            "substeps_per_frame": cfg.substeps_per_frame
            or int(ir_scalar(ir, "sim_substeps")),
            "sim_dt": cfg.sim_dt or ir_scalar(ir, "sim_dt"),
            "gravity_scalar": gravity_scalar,
            "gravity_vector": list(gravity_vec),
            "wall_time_sec": sim_result.wall_time_sec,
        },
        "particles": {
            "total": int(sim_result.particle_q_all.shape[1]),
            "object": n_obj,
            "radius_min": float(np.min(model_result.radius)),
            "radius_max": float(np.max(model_result.radius)),
        },
        "contacts": {
            "shape_contacts_requested": bool(cfg.shape_contacts),
            "collision_pipeline_enabled": _use_collision_pipeline(cfg, ir),
            "ground_plane": bool(cfg.add_ground_plane),
            "particle_contacts": model_result.checks.get(
                "particle_contacts_enabled", False
            ),
            "self_contact_mode": model_result.checks.get("self_contact_mode", SELF_CONTACT_MODE_OFF),
        },
        "validation": model_result.checks,
        "baseline": {
            "inference_path": str(inference_path) if inference is not None else None,
            "compared_frames": compared,
            "rmse_mean": mean_rmse,
            "rmse_max": max_rmse,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Optional rigid probe mode (two-way coupling diagnostics)
# ═══════════════════════════════════════════════════════════════════════════


def _to_body_translation(body_q: np.ndarray) -> np.ndarray:
    if body_q.ndim != 2 or body_q.shape[1] < 3:
        raise ValueError(f"Unexpected body_q shape: {body_q.shape}")
    return body_q[:, :3]


def _triangle_edges_from_indices(indices: np.ndarray) -> np.ndarray:
    """Convert triangle indices [T*3] to unique undirected edges [E,2]."""
    tri = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    e01 = tri[:, [0, 1]]
    e12 = tri[:, [1, 2]]
    e20 = tri[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32, copy=False)


def _load_probe_bunny_geometry(
    cfg: SimConfig,
) -> tuple[newton.Mesh, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load canonical bunny mesh for rigid-probe contact + rendering."""
    try:
        import newton.examples as newton_examples
        import newton.usd as newton_usd
        from pxr import Usd
    except Exception as exc:
        raise RuntimeError(
            "Rigid probe shape 'bunny' requires 'newton.examples', 'newton.usd', and 'pxr'."
        ) from exc

    asset_arg = str(cfg.rigid_probe_bunny_asset).strip()
    asset_path = Path(asset_arg)
    if not asset_path.exists():
        asset_path = Path(newton_examples.get_asset(asset_arg))
    asset_path = asset_path.resolve()

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {asset_path}")
    prim = stage.GetPrimAtPath(str(cfg.rigid_probe_bunny_prim))
    if not prim or not prim.IsValid():
        raise RuntimeError(
            f"USD prim not found/invalid: {cfg.rigid_probe_bunny_prim} in {asset_path}"
        )

    mesh = newton_usd.get_mesh(prim)
    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if indices.size % 3 != 0:
        raise ValueError(f"Bunny mesh indices length must be multiple of 3, got {indices.size}")

    edges = _triangle_edges_from_indices(indices)
    stride = max(1, int(cfg.rigid_probe_bunny_edge_stride))
    render_edges = edges[::stride].copy()
    return mesh, vertices, indices, render_edges, asset_path


def _add_probe_ground_plane(
    builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, checks: dict
) -> None:
    """Optional ground/table for rigid-probe mode."""
    if not cfg.rigid_probe_add_ground_plane:
        checks["ground_plane_added"] = False
        return

    gcfg = builder.default_shape_cfg.copy()
    _configure_ground_contact_material(
        gcfg, ir, cfg, checks, context="rigid_probe_ground_plane"
    )

    reverse_z = ir_bool(ir, "reverse_z")
    if cfg.up_axis == "Z" and reverse_z:
        xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(1, 0, 0, 0))
        builder.add_shape_plane(
            body=-1,
            xform=xform,
            width=0,
            length=0,
            cfg=gcfg,
            label="rigid_probe_ground_plane_reverse_z",
        )
    else:
        builder.add_ground_plane(cfg=gcfg)

    checks["ground_plane_added"] = True
    checks["ground_mu"] = float(gcfg.mu)
    checks["ground_restitution"] = float(gcfg.restitution)
    checks["ground_contact_final_ke"] = float(gcfg.ke)
    checks["ground_contact_final_kd"] = float(gcfg.kd)
    checks["ground_contact_final_kf"] = float(gcfg.kf)


def _build_rigid_probe_model(
    ir: dict, cfg: SimConfig, device: str
) -> tuple[newton.Model, int, np.ndarray, dict]:
    """Build probe model: imported spring-mass object + one native rigid body."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any(cfg.up_axis), gravity=0.0)

    x0 = np.asarray(ir["x0"], dtype=np.float32)
    v0 = np.asarray(ir["v0"], dtype=np.float32)
    mass = np.asarray(ir["mass"], dtype=np.float32).copy()
    n_total = x0.shape[0]
    n_obj = int(np.asarray(ir["num_object_points"]).ravel()[0])
    if n_obj <= 0 or n_obj > n_total:
        raise ValueError(f"Invalid num_object_points={n_obj} for total={n_total}")

    # Probe requirement: use a known object-particle mass for momentum diagnostics.
    mass[:n_obj] = float(cfg.rigid_probe_object_mass)

    radius, _, _ = resolve_collision_radius(ir, n_total)
    if "contact_collision_dist" in ir:
        collision_dist = ir_scalar(ir, "contact_collision_dist")
        radius[:n_obj] = max(collision_dist * 0.5, EPSILON)
    if "controller_idx" in ir:
        ctrl_idx = np.asarray(ir["controller_idx"], dtype=np.int64).ravel()
        if ctrl_idx.size:
            radius[ctrl_idx] = float(cfg.particle_contact_radius)

    flags = np.full(n_total, int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
    builder.add_particles(
        pos=[tuple(row.tolist()) for row in x0],
        vel=[tuple(row.tolist()) for row in v0],
        mass=mass.astype(float).tolist(),
        radius=radius.astype(float).tolist(),
        flags=flags.astype(int).tolist(),
    )

    edges = np.asarray(ir["spring_edges"], dtype=np.int32)
    rest = np.asarray(ir["spring_rest_length"], dtype=np.float32).ravel()
    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel() * cfg.spring_ke_scale
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel() * cfg.spring_kd_scale
    if not (edges.shape[0] == rest.shape[0] == ke.shape[0] == kd.shape[0]):
        raise ValueError("Spring arrays have inconsistent lengths.")

    if cfg.rigid_probe_drop_controller_springs:
        keep = (edges[:, 0] < n_obj) & (edges[:, 1] < n_obj)
        edges = edges[keep]
        rest = rest[keep]
        ke = ke[keep]
        kd = kd[keep]

    for idx in range(edges.shape[0]):
        i = int(edges[idx, 0])
        j = int(edges[idx, 1])
        builder.add_spring(i=i, j=j, ke=float(ke[idx]), kd=float(kd[idx]), control=0.0)
        builder.spring_rest_length[-1] = float(rest[idx])

    probe_checks: dict = {}
    _add_probe_ground_plane(builder, ir, cfg, probe_checks)

    obj_center = x0[:n_obj].mean(axis=0)
    body_pos = obj_center + np.asarray(cfg.rigid_probe_offset, dtype=np.float32)
    body_quat = wp.quat_identity()
    if cfg.rigid_probe_shape == "bunny":
        # Keep bunny upright under reverse-z gravity:
        # map mesh local +Y (asset up) -> world -Z (simulation up when reverse_z=True).
        # This is Rx(-90 deg) in xyzw convention.
        body_quat = wp.quat(-0.70710678, 0.0, 0.0, 0.70710678)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*body_pos.tolist()), body_quat),
        mass=float(cfg.rigid_probe_mass),
        inertia=wp.mat33(
            float(cfg.rigid_probe_inertia_diag),
            0.0,
            0.0,
            0.0,
            float(cfg.rigid_probe_inertia_diag),
            0.0,
            0.0,
            0.0,
            float(cfg.rigid_probe_inertia_diag),
        ),
        lock_inertia=True,
        label=f"rigid_probe_{cfg.rigid_probe_shape}",
    )
    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(cfg.rigid_probe_mu)
    rigid_cfg.ke = float(cfg.rigid_probe_ke)
    rigid_cfg.kd = float(cfg.rigid_probe_kd)

    probe_meta: dict = {
        "shape_kind": "box",
        "rigid_hx": float(cfg.rigid_probe_hx),
        "rigid_hy": float(cfg.rigid_probe_hy),
        "rigid_hz": float(cfg.rigid_probe_hz),
    }

    if cfg.rigid_probe_shape == "box":
        builder.add_shape_box(
            body=body,
            hx=float(cfg.rigid_probe_hx),
            hy=float(cfg.rigid_probe_hy),
            hz=float(cfg.rigid_probe_hz),
            cfg=rigid_cfg,
        )
    elif cfg.rigid_probe_shape == "bunny":
        mesh, mesh_vertices_local, mesh_indices, mesh_render_edges, mesh_asset_path = (
            _load_probe_bunny_geometry(cfg)
        )
        scale = float(cfg.rigid_probe_bunny_scale)
        builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            scale=(scale, scale, scale),
            cfg=rigid_cfg,
        )
        scaled_vertices = mesh_vertices_local * scale
        min_v = np.min(scaled_vertices, axis=0)
        max_v = np.max(scaled_vertices, axis=0)
        half = 0.5 * (max_v - min_v)
        probe_meta.update(
            {
                "shape_kind": "bunny_mesh",
                "rigid_hx": float(half[0]),
                "rigid_hy": float(half[1]),
                "rigid_hz": float(half[2]),
                "rigid_mesh_vertices_local": mesh_vertices_local.astype(np.float32, copy=False),
                "rigid_mesh_indices": mesh_indices.astype(np.int32, copy=False),
                "rigid_mesh_render_edges": mesh_render_edges.astype(np.int32, copy=False),
                "rigid_mesh_scale": float(scale),
                "rigid_mesh_asset_path": str(mesh_asset_path),
                "rigid_mesh_prim_path": str(cfg.rigid_probe_bunny_prim),
            }
        )
    else:
        raise ValueError(
            f"Unsupported --rigid-probe-shape={cfg.rigid_probe_shape!r}; expected box or bunny"
        )

    model = builder.finalize(device=device)
    if cfg.rigid_probe_use_scene_gravity:
        gravity_scalar, gravity_vec = resolve_gravity(cfg, ir)
        model.set_gravity(gravity_vec)
    else:
        gravity_scalar, gravity_vec = 0.0, (0.0, 0.0, 0.0)
        model.set_gravity(gravity_vec)

    probe_meta.update(
        {
            "ground_plane_added": bool(probe_checks.get("ground_plane_added", False)),
            "ground_mu": float(probe_checks.get("ground_mu", 0.0)),
            "ground_restitution": float(probe_checks.get("ground_restitution", 0.0)),
            "gravity_scalar": float(gravity_scalar),
            "gravity_vector": tuple(float(v) for v in gravity_vec),
        }
    )
    return model, n_obj, mass, probe_meta


def run_rigid_probe(cfg: SimConfig, ir: dict, device: str) -> dict:
    """Run importer-integrated rigid probe mode and write diagnostics files."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model, n_obj, particle_mass, probe_meta = _build_rigid_probe_model(ir, cfg, device)

    solver = newton.solvers.SolverSemiImplicit(
        model,
        angular_damping=cfg.angular_damping,
        friction_smoothing=cfg.friction_smoothing,
        enable_tri_contact=cfg.enable_tri_contact,
    )
    contacts = model.contacts()
    control = model.control()
    state_in = model.state()
    state_out = model.state()

    ctrl_idx = np.asarray(
        ir.get("controller_idx", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    ).ravel()
    ctrl_idx_wp = None
    ctrl_target_wp = None
    ctrl_vel_wp = None
    if cfg.rigid_probe_freeze_controllers and ctrl_idx.size:
        ctrl_traj = np.asarray(ir["controller_traj"], dtype=np.float32)
        ctrl_target_wp = wp.array(ctrl_traj[0], dtype=wp.vec3, device=device)
        ctrl_idx_wp = wp.array(ctrl_idx, dtype=wp.int32, device=device)
        ctrl_vel_wp = wp.zeros(ctrl_idx.size, dtype=wp.vec3, device=device)

    body_qd = state_in.body_qd.numpy()
    if body_qd.shape[0] < 1:
        raise RuntimeError("Rigid probe requires one rigid body.")
    body_qd[0, 0:3] = np.asarray(cfg.rigid_probe_velocity, dtype=np.float32)
    body_qd[0, 3:6] = 0.0
    state_in.body_qd.assign(body_qd.astype(np.float32))

    sim_dt = cfg.sim_dt if cfg.sim_dt is not None else ir_scalar(ir, "sim_dt")
    substeps = cfg.substeps_per_frame or int(ir_scalar(ir, "sim_substeps"))
    substeps = max(1, substeps)
    n_frames = max(2, int(cfg.num_frames))

    drag = 0.0
    if cfg.apply_drag and "drag_damping" in ir:
        drag = ir_scalar(ir, "drag_damping") * cfg.drag_damping_scale

    body_pos_hist: list[np.ndarray] = []
    body_vel_hist: list[np.ndarray] = []
    obj_com_hist: list[np.ndarray] = []
    p_obj_hist: list[np.ndarray] = []
    p_body_hist: list[np.ndarray] = []
    p_total_hist: list[np.ndarray] = []
    # Scene capture for visualization (particles + rigid body).
    # We capture per-frame snapshots (not per-substep) to keep size manageable.
    obj_q_hist: list[np.ndarray] = []
    body_q_hist: list[np.ndarray] = []

    t0 = time.perf_counter()
    for _frame in range(n_frames):
        q = state_in.particle_q.numpy().astype(np.float32)
        qd = state_in.particle_qd.numpy().astype(np.float32)
        body_q_raw = state_in.body_q.numpy().astype(np.float32)
        bq = _to_body_translation(body_q_raw)[0]
        bqd = state_in.body_qd.numpy().astype(np.float32)[0, 0:3]

        obj_q = q[:n_obj]
        obj_qd = qd[:n_obj]
        obj_mass = particle_mass[:n_obj].astype(np.float32)
        obj_com = (obj_q * obj_mass[:, None]).sum(axis=0) / max(float(obj_mass.sum()), EPSILON)
        p_obj = (obj_qd * obj_mass[:, None]).sum(axis=0)
        p_body = bqd * float(cfg.rigid_probe_mass)
        p_total = p_obj + p_body

        body_pos_hist.append(bq.copy())
        body_vel_hist.append(bqd.copy())
        obj_com_hist.append(obj_com.copy())
        p_obj_hist.append(p_obj.copy())
        p_body_hist.append(p_body.copy())
        p_total_hist.append(p_total.copy())
        obj_q_hist.append(obj_q.copy())
        body_q_hist.append(body_q_raw[0].copy())

        for _sub in range(substeps):
            state_in.clear_forces()

            if ctrl_idx_wp is not None and ctrl_target_wp is not None and ctrl_vel_wp is not None:
                wp.launch(
                    _write_kinematic_state,
                    dim=ctrl_idx.size,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        ctrl_idx_wp,
                        ctrl_target_wp,
                        ctrl_vel_wp,
                    ],
                    device=device,
                )

            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                wp.launch(
                    _apply_drag_correction,
                    dim=n_obj,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        n_obj,
                        sim_dt,
                        drag,
                    ],
                    device=device,
                )

    wall_time = time.perf_counter() - t0

    body_pos = np.asarray(body_pos_hist, dtype=np.float32)
    body_vel = np.asarray(body_vel_hist, dtype=np.float32)
    obj_com = np.asarray(obj_com_hist, dtype=np.float32)
    p_obj = np.asarray(p_obj_hist, dtype=np.float32)
    p_body = np.asarray(p_body_hist, dtype=np.float32)
    p_total = np.asarray(p_total_hist, dtype=np.float32)
    obj_q_frames = np.asarray(obj_q_hist, dtype=np.float32)
    body_q_frames = np.asarray(body_q_hist, dtype=np.float32)

    p0 = p_total[0]
    p1 = p_total[-1]
    p_delta = p1 - p0
    p_delta_norm = float(np.linalg.norm(p_delta))
    p0_norm = float(np.linalg.norm(p0))
    p_rel = float(p_delta_norm / max(p0_norm, EPSILON))
    body_speed0 = float(np.linalg.norm(body_vel[0]))
    body_speed_min = float(np.min(np.linalg.norm(body_vel, axis=1)))
    body_speed_end = float(np.linalg.norm(body_vel[-1]))
    min_body_obj_dist = float(np.min(np.linalg.norm(body_pos - obj_com, axis=1)))

    out_npz = cfg.out_dir / f"{cfg.output_prefix}.npz"
    out_csv = cfg.out_dir / f"{cfg.output_prefix}_timeseries.csv"
    out_json = cfg.out_dir / f"{cfg.output_prefix}.json"
    out_scene_npz = cfg.out_dir / f"{cfg.output_prefix}_scene.npz"

    np.savez_compressed(
        out_npz,
        body_pos=body_pos,
        body_vel=body_vel,
        object_com=obj_com,
        p_obj=p_obj,
        p_body=p_body,
        p_total=p_total,
        sim_dt=np.float32(sim_dt),
        substeps_per_frame=np.int32(substeps),
    )
    scene_payload: dict = {
        "particle_q_object": obj_q_frames,
        "body_q": body_q_frames,
        "num_object_points": np.int32(n_obj),
        "rigid_hx": np.float32(float(probe_meta.get("rigid_hx", cfg.rigid_probe_hx))),
        "rigid_hy": np.float32(float(probe_meta.get("rigid_hy", cfg.rigid_probe_hy))),
        "rigid_hz": np.float32(float(probe_meta.get("rigid_hz", cfg.rigid_probe_hz))),
        "rigid_shape_kind": np.asarray(str(probe_meta.get("shape_kind", "box"))),
        "sim_dt": np.float32(sim_dt),
        "substeps_per_frame": np.int32(substeps),
        "gravity_scalar": np.float32(float(probe_meta.get("gravity_scalar", 0.0))),
        "gravity_vector": np.asarray(probe_meta.get("gravity_vector", (0.0, 0.0, 0.0)), dtype=np.float32),
    }
    if "rigid_mesh_vertices_local" in probe_meta:
        scene_payload["rigid_mesh_vertices_local"] = np.asarray(
            probe_meta["rigid_mesh_vertices_local"], dtype=np.float32
        )
    if "rigid_mesh_indices" in probe_meta:
        scene_payload["rigid_mesh_indices"] = np.asarray(
            probe_meta["rigid_mesh_indices"], dtype=np.int32
        )
    if "rigid_mesh_render_edges" in probe_meta:
        scene_payload["rigid_mesh_render_edges"] = np.asarray(
            probe_meta["rigid_mesh_render_edges"], dtype=np.int32
        )
    if "rigid_mesh_scale" in probe_meta:
        scene_payload["rigid_mesh_scale"] = np.float32(float(probe_meta["rigid_mesh_scale"]))
    np.savez_compressed(out_scene_npz, **scene_payload)

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "body_x",
                "body_y",
                "body_z",
                "obj_x",
                "obj_y",
                "obj_z",
                "body_vx",
                "body_vy",
                "body_vz",
                "p_total_x",
                "p_total_y",
                "p_total_z",
            ]
        )
        for i in range(body_pos.shape[0]):
            writer.writerow(
                [
                    i,
                    *body_pos[i].tolist(),
                    *obj_com[i].tolist(),
                    *body_vel[i].tolist(),
                    *p_total[i].tolist(),
                ]
            )

    summary = {
        "mode": "rigid_probe",
        "ir_path": str(cfg.ir_path),
        "device_requested": cfg.device,
        "device_used": str(model.device),
        "frames": int(n_frames),
        "substeps_per_frame": int(substeps),
        "sim_dt": float(sim_dt),
        "object_mass_set_to": float(cfg.rigid_probe_object_mass),
        "rigid_mass": float(cfg.rigid_probe_mass),
        "rigid_shape": str(probe_meta.get("shape_kind", cfg.rigid_probe_shape)),
        "rigid_probe_offset": [float(v) for v in cfg.rigid_probe_offset],
        "rigid_probe_velocity": [float(v) for v in cfg.rigid_probe_velocity],
        "drop_controller_springs": bool(cfg.rigid_probe_drop_controller_springs),
        "freeze_controllers": bool(cfg.rigid_probe_freeze_controllers),
        "ground_plane_added": bool(probe_meta.get("ground_plane_added", False)),
        "ground_mu": float(probe_meta.get("ground_mu", 0.0)),
        "ground_restitution": float(probe_meta.get("ground_restitution", 0.0)),
        "ground_restitution_mode": str(cfg.ground_restitution_mode),
        "gravity_scalar": float(probe_meta.get("gravity_scalar", 0.0)),
        "gravity_vector": [float(v) for v in probe_meta.get("gravity_vector", (0.0, 0.0, 0.0))],
        "use_scene_gravity": bool(cfg.rigid_probe_use_scene_gravity),
        "probe_add_ground_plane": bool(cfg.rigid_probe_add_ground_plane),
        "initial_total_momentum_norm": p0_norm,
        "final_total_momentum_norm": float(np.linalg.norm(p1)),
        "momentum_delta_norm": p_delta_norm,
        "momentum_delta_relative": p_rel,
        "body_speed_initial": body_speed0,
        "body_speed_min": body_speed_min,
        "body_speed_final": body_speed_end,
        "min_body_object_com_distance": min_body_obj_dist,
        "wall_time_sec": wall_time,
        "outputs": {
            "npz": str(out_npz),
            "csv": str(out_csv),
            "scene_npz": str(out_scene_npz),
        },
    }
    if "rigid_mesh_asset_path" in probe_meta:
        summary["rigid_mesh_asset_path"] = str(probe_meta["rigid_mesh_asset_path"])
    if "rigid_mesh_prim_path" in probe_meta:
        summary["rigid_mesh_prim_path"] = str(probe_meta["rigid_mesh_prim_path"])
    if "rigid_mesh_scale" in probe_meta:
        summary["rigid_mesh_scale"] = float(probe_meta["rigid_mesh_scale"])

    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    cfg = SimConfig.from_args(parse_args())
    ir = load_ir(cfg.ir_path)
    device = resolve_device(cfg.device)
    wp.init()
    _validate_config_ranges(cfg)

    if cfg.rigid_probe:
        summary = run_rigid_probe(cfg, ir, device)
    else:
        model_result = build_model(ir, cfg, device)
        sim_result = simulate(model_result.model, ir, cfg, device, model_result=model_result)
        summary = save_results(cfg, ir, model_result, sim_result)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
