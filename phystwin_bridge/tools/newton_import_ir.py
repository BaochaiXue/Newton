#!/usr/bin/env python3
"""Import a PhysTwin IR into Newton and run a semi-implicit simulation.

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
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

import newton
from path_defaults import default_device

# ═══════════════════════════════════════════════════════════════════════════
# Constants – no magic numbers anywhere else in this file.
# ═══════════════════════════════════════════════════════════════════════════

# IR
IR_VERSION_KEY = "ir_version"
IR_COLLISION_RADIUS_KEY = "collision_radius"

# Numerical tolerances
EPSILON = 1e-12                   # safe-division floor
STRICT_KE_REL_TOL = 1e-4         # max relative error for ke consistency
KINEMATIC_MASS_TOL = 1e-8         # controllers must have |mass| below this

# Physics defaults
DEFAULT_OBJECT_RADIUS = 0.02      # fallback collision radius for legacy IR
LEGACY_RADIUS_RATIO_THRESH = 2.5  # above → interpret radius as topology

# Contact clamping
MU_MAX = 2.0
RESTITUTION_MAX = 1.0

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
    disable_particle_contact_kernel: bool = False

    # Simulation
    num_frames: int = 20
    substeps_per_frame: int | None = None
    sim_dt: float | None = None

    # Gravity
    gravity: float | None = None
    gravity_mag: float = 9.8
    gravity_from_reverse_z: bool = True
    up_axis: str = "Z"

    # Frame sync
    frame_sync: str = "phystwin"

    # Device
    device: str = "cuda:0"

    # Contacts
    shape_contacts: bool = False
    add_ground_plane: bool = True
    particle_contacts: bool | None = None
    particle_contact_radius: float = 1e-5
    object_contact_radius: float | None = None

    # Controls
    interpolate_controls: bool = True

    # Validation
    strict_physics_checks: bool = True

    # Drag
    apply_drag: bool = False
    drag_damping_scale: float = 1.0

    # Friction / restitution
    particle_mu_override: float | None = None
    ground_mu_scale: float = 1.0
    ground_restitution_scale: float = 1.0

    # Baseline
    inference_path: Path | None = None

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
            frame_sync=args.frame_sync,
            device=args.device,
            shape_contacts=args.shape_contacts,
            add_ground_plane=args.add_ground_plane,
            particle_contacts=args.particle_contacts,
            particle_contact_radius=args.particle_contact_radius,
            object_contact_radius=args.object_contact_radius,
            interpolate_controls=args.interpolate_controls,
            strict_physics_checks=args.strict_physics_checks,
            apply_drag=args.apply_drag,
            drag_damping_scale=args.drag_damping_scale,
            particle_mu_override=args.particle_mu_override,
            ground_mu_scale=args.ground_mu_scale,
            ground_restitution_scale=args.ground_restitution_scale,
            inference_path=args.inference,
        )


@dataclass
class ModelResult:
    """Everything produced by ``build_model``."""
    model: newton.Model
    radius: np.ndarray
    ir_version: int
    checks: dict = field(default_factory=dict)


@dataclass
class SimResult:
    """Everything produced by ``simulate``."""
    particle_q_all: np.ndarray       # [frames+1, N, 3]
    particle_q_object: np.ndarray    # [frames+1, n_obj, 3]
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
    p.add_argument("--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disable-particle-contact-kernel", action=argparse.BooleanOptionalAction, default=False)

    # Simulation
    p.add_argument("--num-frames", type=int, default=20)
    p.add_argument("--substeps-per-frame", type=int, default=None)
    p.add_argument("--sim-dt", type=float, default=None)

    # Gravity
    p.add_argument("--gravity", type=float, default=None)
    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--gravity-from-reverse-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")

    # Frame / device
    p.add_argument("--frame-sync", choices=["legacy", "phystwin"], default="phystwin")
    p.add_argument("--device", default=default_device())

    # Contacts
    p.add_argument("--shape-contacts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--add-ground-plane", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--particle-contacts", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--particle-contact-radius", type=float, default=1e-5)
    p.add_argument("--object-contact-radius", type=float, default=None)

    # Controls / validation / drag
    p.add_argument("--interpolate-controls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-physics-checks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--apply-drag", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--drag-damping-scale", type=float, default=1.0)

    # Friction / restitution
    p.add_argument("--particle-mu-override", type=float, default=None)
    p.add_argument("--ground-mu-scale", type=float, default=1.0)
    p.add_argument("--ground-restitution-scale", type=float, default=1.0)

    # Baseline
    p.add_argument("--inference", type=Path, default=None)

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
def _apply_drag_correction(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    n_object: int,
    dt: float,
    damping: float,
):
    """PhysTwin drag: ``v ← v·exp(−dt·k)`` with position compensation.

    Newton integrates position with the *undamped* velocity, so we subtract
    the overshoot ``v·dt·(1−exp(−dt·k))`` from the position.
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
    try:
        wp.get_device(requested)
        return requested
    except Exception:
        return "cpu"

# ═══════════════════════════════════════════════════════════════════════════
# Gravity
# ═══════════════════════════════════════════════════════════════════════════

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

def resolve_gravity(cfg: SimConfig, ir: dict) -> tuple[float, tuple[float, float, float]]:
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
    }
    for name, value, op, bound in checks:
        if op == ">" and value <= bound:
            raise ValueError(f"{name} must be > {bound}, got {value}")
        if op == ">=" and value < bound:
            raise ValueError(f"{name} must be >= {bound}, got {value}")


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
        raise ValueError(f"Strict mode requires spring_ke_mode=y_over_rest, got {ke_mode!r}")

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
            raise ValueError(f"ke inconsistent with Y/rest (max_rel_err={float(np.max(rel_err)):.2e})")
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
            raise ValueError(f"Controllers must have mass≈0, got max|mass|={max_mass:.2e}")

    if "self_collision" in ir:
        checks["self_collision"] = ir_bool(ir, "self_collision")

    return checks

# ═══════════════════════════════════════════════════════════════════════════
# Collision radius resolution
# ═══════════════════════════════════════════════════════════════════════════

def _uniform_radius(ir: dict, n: int, value: float) -> np.ndarray:
    r = np.full((n,), value, dtype=np.float32)
    if "controller_idx" in ir and "collision_controller_radius" in ir:
        idx = ir["controller_idx"].astype(np.int64, copy=False)
        if idx.size > 0:
            r[idx] = ir_scalar(ir, "collision_controller_radius")
    return r


def resolve_collision_radius(ir: dict, n: int) -> tuple[np.ndarray, int, str]:
    """Determine per-particle collision radii from IR.  Returns (radii, version, source)."""
    ver = ir_version(ir)

    if IR_COLLISION_RADIUS_KEY in ir:
        return ir[IR_COLLISION_RADIUS_KEY].astype(np.float32).copy(), ver, "collision_radius"

    if ver >= 2:
        if "radius" in ir:
            warnings.warn("IR v2 missing collision_radius; falling back to radius.", RuntimeWarning)
            return ir["radius"].astype(np.float32).copy(), ver, "v2_radius_fallback"
        raise KeyError("IR v2 requires collision_radius.")

    # Legacy heuristics
    if "radius" in ir and "contact_collision_dist" in ir:
        r = ir["radius"].astype(np.float32).copy()
        dist = ir_scalar(ir, "contact_collision_dist")
        if float(np.max(r) / max(dist, EPSILON)) > LEGACY_RADIUS_RATIO_THRESH:
            warnings.warn("Legacy IR: radius looks like topology; using contact_collision_dist.", RuntimeWarning)
            return _uniform_radius(ir, n, dist), ver, "legacy_contact_collision_dist"
        return r, ver, "legacy_radius"

    if "radius" in ir:
        return ir["radius"].astype(np.float32).copy(), ver, "legacy_radius"

    if "contact_collision_dist" in ir:
        warnings.warn("Legacy IR missing radius; using contact_collision_dist.", RuntimeWarning)
        return _uniform_radius(ir, n, ir_scalar(ir, "contact_collision_dist")), ver, "legacy_contact_collision_dist"

    obj_r = ir_scalar(ir, "collision_object_radius", ir_scalar(ir, "object_radius", DEFAULT_OBJECT_RADIUS))
    ctrl_r = ir_scalar(ir, "collision_controller_radius", ir_scalar(ir, "controller_radius", obj_r))
    r = np.full((n,), obj_r, dtype=np.float32)
    if "controller_idx" in ir:
        idx = ir["controller_idx"].astype(np.int64, copy=False)
        if idx.size > 0:
            r[idx] = ctrl_r
    return r, ver, "legacy_fallback"

# ═══════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_particle_contacts(cfg: SimConfig, ir: dict) -> bool:
    if cfg.particle_contacts is not None:
        return cfg.particle_contacts
    return ir_bool(ir, "self_collision")


def _add_particles(builder: newton.ModelBuilder, ir: dict, cfg: SimConfig,
                   particle_contacts: bool) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Add IR particles to the builder.  Returns (radius, flags, ir_ver, radius_source)."""
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
            raise ValueError(f"object_contact_radius must be > 0, got {cfg.object_contact_radius}")
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
    return radius, flags, ver, src


def _add_springs(builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, checks: dict) -> None:
    """Add IR springs (with scaled ke/kd) to the builder."""
    edges = np.asarray(ir["spring_edges"])
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected spring_edges shape [N, 2], got {edges.shape}")
    n = edges.shape[0]

    ke = np.asarray(ir["spring_ke"], dtype=np.float32).ravel() * cfg.spring_ke_scale
    kd = np.asarray(ir["spring_kd"], dtype=np.float32).ravel() * cfg.spring_kd_scale
    if ke.shape[0] != n or kd.shape[0] != n:
        raise ValueError(f"Spring param count mismatch: ke={ke.shape[0]}, kd={kd.shape[0]}, edges={n}")

    rest = ir["spring_rest_length"].astype(np.float32).ravel() if "spring_rest_length" in ir else None

    for i in range(n):
        builder.add_spring(i=int(edges[i, 0]), j=int(edges[i, 1]),
                           ke=float(ke[i]), kd=float(kd[i]), control=0.0)
        if rest is not None:
            builder.spring_rest_length[-1] = float(rest[i])

    checks["spring_ke_scale"] = cfg.spring_ke_scale
    checks["spring_kd_scale"] = cfg.spring_kd_scale


def _add_ground_plane(builder: newton.ModelBuilder, ir: dict, cfg: SimConfig, checks: dict) -> None:
    """Optionally add a ground plane from IR contact params."""
    if not cfg.add_ground_plane:
        checks["ground_plane_added"] = False
        return

    gcfg = builder.default_shape_cfg.copy()
    if "contact_collide_fric" in ir:
        gcfg.mu = float(np.clip(ir_scalar(ir, "contact_collide_fric") * cfg.ground_mu_scale, 0, MU_MAX))
    if "contact_collide_elas" in ir:
        gcfg.restitution = float(np.clip(
            ir_scalar(ir, "contact_collide_elas") * cfg.ground_restitution_scale, 0, RESTITUTION_MAX))

    reverse_z = ir_bool(ir, "reverse_z")
    if cfg.up_axis == "Z" and reverse_z:
        xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(1, 0, 0, 0))
        builder.add_shape_plane(body=-1, xform=xform, width=0, length=0,
                                cfg=gcfg, label="ground_plane_reverse_z")
    else:
        builder.add_ground_plane(cfg=gcfg)

    checks["ground_plane_added"] = True
    checks["ground_mu"] = float(gcfg.mu)
    checks["ground_restitution"] = float(gcfg.restitution)


def build_model(ir: dict, cfg: SimConfig, device: str) -> ModelResult:
    """Build a complete Newton Model from IR data.

    This is the main extensibility point.  To add new objects to the
    simulation, add ``builder.add_body()`` / ``builder.add_shape_*()``
    calls before ``builder.finalize()``.
    """
    checks = validate_ir_physics(ir, cfg)
    particle_contacts = _resolve_particle_contacts(cfg, ir)

    builder = newton.ModelBuilder(
        up_axis=newton.Axis.from_any(cfg.up_axis),
        gravity=0.0,
    )

    radius, _flags, ver, radius_src = _add_particles(builder, ir, cfg, particle_contacts)
    _add_springs(builder, ir, cfg, checks)
    _add_ground_plane(builder, ir, cfg, checks)

    # ── Extension point ──
    # Add new objects here: builder.add_body(), builder.add_shape_mesh(), etc.
    # They will interact with PhysTwin particles through Newton's collision.

    model = builder.finalize(device=device)

    # Particle contact grid
    if not particle_contacts or cfg.disable_particle_contact_kernel:
        model.particle_grid = None

    # Gravity
    _, gravity_vec = resolve_gravity(cfg, ir)
    model.set_gravity(gravity_vec)

    # Particle friction
    if cfg.particle_mu_override is not None:
        model.particle_mu = float(np.clip(cfg.particle_mu_override, 0, MU_MAX))
    elif "contact_collide_fric" in ir:
        model.particle_mu = float(np.clip(ir_scalar(ir, "contact_collide_fric"), 0, MU_MAX))

    checks["particle_contacts_enabled"] = particle_contacts
    checks["collision_radius_source"] = radius_src

    return ModelResult(model=model, radius=radius, ir_version=ver, checks=checks)

# ═══════════════════════════════════════════════════════════════════════════
# Controller interpolation
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_controller(traj: np.ndarray, frame: int, substep: int,
                           n_substeps: int, interpolate: bool) -> np.ndarray:
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

def simulate(model: newton.Model, ir: dict, cfg: SimConfig, device: str) -> SimResult:
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
    contacts = model.contacts()
    contacts_enabled = cfg.shape_contacts or ir_bool(ir, "self_collision")

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
        raise ValueError(f"Controller count mismatch: traj={ctrl_traj.shape[1]}, idx={ctrl_idx.size}")

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

    # Collect rollout
    rollout_all: list[np.ndarray] = []
    rollout_obj: list[np.ndarray] = []

    if cfg.frame_sync == "phystwin":
        q0 = state_in.particle_q.numpy().astype(np.float32)
        rollout_all.append(q0)
        rollout_obj.append(q0[:n_obj])
        frame_range = range(1, n_frames)
    else:
        frame_range = range(n_frames)

    # Main loop
    t0 = time.perf_counter()
    for frame in frame_range:
        for sub in range(substeps):
            state_in.clear_forces()

            if has_controllers:
                target = interpolate_controller(ctrl_traj, frame, sub, substeps, cfg.interpolate_controls)
                ctrl_target_wp.assign(target.astype(np.float32, copy=False))
                ctrl_vel_wp.assign(ctrl_vel_zero)
                wp.launch(_write_kinematic_state, dim=ctrl_idx.size,
                          inputs=[state_in.particle_q, state_in.particle_qd,
                                  ctrl_idx_wp, ctrl_target_wp, ctrl_vel_wp],
                          device=device)

            if contacts_enabled:
                model.collide(state_in, contacts)
            else:
                contacts.clear()

            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

            if drag > 0.0:
                wp.launch(_apply_drag_correction, dim=n_obj,
                          inputs=[state_in.particle_q, state_in.particle_qd,
                                  n_obj, sim_dt, drag],
                          device=device)

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
    cand = cfg.ir_path.parent.parent.parent / "inputs" / "cases" / case / "inference.pkl"
    return cand if cand.exists() else None


def save_results(cfg: SimConfig, ir: dict, model_result: ModelResult,
                 sim_result: SimResult) -> dict:
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
        rmse_per_frame = np.sqrt(np.mean(err ** 2, axis=(1, 2))).astype(np.float32)
        mean_rmse = float(rmse_per_frame.mean())
        max_rmse = float(rmse_per_frame.max())

    npz_path = cfg.out_dir / f"{cfg.output_prefix}.npz"
    json_path = cfg.out_dir / f"{cfg.output_prefix}.json"

    np.savez_compressed(
        npz_path,
        particle_q_all=sim_result.particle_q_all,
        particle_q_object=sim_result.particle_q_object,
        sim_dt=np.float32(cfg.sim_dt or ir_scalar(ir, "sim_dt")),
        substeps_per_frame=np.int32(cfg.substeps_per_frame or int(ir_scalar(ir, "sim_substeps"))),
        frames_run=np.int32(sim_result.particle_q_all.shape[0]),
        rmse_per_frame=rmse_per_frame if rmse_per_frame is not None else np.zeros(0, dtype=np.float32),
    )

    gravity_scalar, gravity_vec = resolve_gravity(cfg, ir)
    summary = {
        "ir_path": str(cfg.ir_path),
        "output_npz": str(npz_path),
        "device": cfg.device,
        "ir_version": model_result.ir_version,
        "config": {
            "spring_ke_scale": cfg.spring_ke_scale,
            "spring_kd_scale": cfg.spring_kd_scale,
            "angular_damping": cfg.angular_damping,
            "friction_smoothing": cfg.friction_smoothing,
            "enable_tri_contact": cfg.enable_tri_contact,
            "interpolate_controls": cfg.interpolate_controls,
            "frame_sync": cfg.frame_sync,
            "apply_drag": cfg.apply_drag,
            "drag_damping_scale": cfg.drag_damping_scale,
            "ground_mu_scale": cfg.ground_mu_scale,
            "ground_restitution_scale": cfg.ground_restitution_scale,
        },
        "simulation": {
            "frames_run": int(sim_result.particle_q_all.shape[0]),
            "substeps_per_frame": cfg.substeps_per_frame or int(ir_scalar(ir, "sim_substeps")),
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
            "shape_contacts": cfg.shape_contacts,
            "ground_plane": cfg.add_ground_plane,
            "particle_contacts": model_result.checks.get("particle_contacts_enabled", False),
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
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    cfg = SimConfig.from_args(parse_args())
    ir = load_ir(cfg.ir_path)
    device = resolve_device(cfg.device)
    wp.init()

    model_result = build_model(ir, cfg, device)
    sim_result = simulate(model_result.model, ir, cfg, device)
    summary = save_results(cfg, ir, model_result, sim_result)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
