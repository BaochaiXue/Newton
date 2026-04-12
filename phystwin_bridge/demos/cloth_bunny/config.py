#!/usr/bin/env python3
"""Typed configuration helpers for the canonical cloth+bunny package."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from bridge_bootstrap import path_defaults

from .scene import _default_cloth_ir


@dataclass(slots=True)
class ClothBunnySceneConfig:
    ir: Path
    drop_height: float
    cloth_shift_x: float
    cloth_shift_y: float
    rigid_mass: float
    rigid_shape: str
    add_bunny: bool
    add_ground_plane: bool
    shape_contacts: bool


@dataclass(slots=True)
class ClothBunnyRuntimeConfig:
    device: str
    frames: int
    sim_dt: float
    substeps: int
    gravity_mag: float
    apply_drag: bool
    drag_damping_scale: float
    drag_ignore_gravity_axis: bool


@dataclass(slots=True)
class ClothBunnyDiagnosticConfig:
    force_diagnostic: bool
    defer_force_artifacts: bool
    parity_check: bool
    stop_after_diagnostic: bool
    skip_render: bool


@dataclass(slots=True)
class ClothBunnyOutputConfig:
    out_dir: Path
    prefix: str
    render_fps: float
    slowdown: float


@dataclass(slots=True)
class ClothBunnyConfigBundle:
    scene: ClothBunnySceneConfig
    runtime: ClothBunnyRuntimeConfig
    diagnostics: ClothBunnyDiagnosticConfig
    output: ClothBunnyOutputConfig
    legacy_args: argparse.Namespace


def build_config_bundle(args: argparse.Namespace) -> ClothBunnyConfigBundle:
    """Build typed config surfaces from a legacy argparse namespace."""

    return ClothBunnyConfigBundle(
        scene=ClothBunnySceneConfig(
            ir=Path(args.ir).expanduser().resolve(),
            drop_height=float(args.drop_height),
            cloth_shift_x=float(args.cloth_shift_x),
            cloth_shift_y=float(args.cloth_shift_y),
            rigid_mass=float(args.rigid_mass),
            rigid_shape=str(getattr(args, "rigid_shape", "bunny")),
            add_bunny=bool(getattr(args, "add_bunny", True)),
            add_ground_plane=bool(getattr(args, "add_ground_plane", True)),
            shape_contacts=bool(getattr(args, "shape_contacts", True)),
        ),
        runtime=ClothBunnyRuntimeConfig(
            device=str(args.device),
            frames=int(args.frames),
            sim_dt=float(args.sim_dt),
            substeps=int(args.substeps),
            gravity_mag=float(args.gravity_mag),
            apply_drag=bool(args.apply_drag),
            drag_damping_scale=float(args.drag_damping_scale),
            drag_ignore_gravity_axis=bool(args.drag_ignore_gravity_axis),
        ),
        diagnostics=ClothBunnyDiagnosticConfig(
            force_diagnostic=bool(getattr(args, "force_diagnostic", False)),
            defer_force_artifacts=bool(getattr(args, "defer_force_artifacts", False)),
            parity_check=bool(getattr(args, "parity_check", False)),
            stop_after_diagnostic=bool(getattr(args, "stop_after_diagnostic", False)),
            skip_render=bool(getattr(args, "skip_render", False)),
        ),
        output=ClothBunnyOutputConfig(
            out_dir=Path(args.out_dir).expanduser().resolve(),
            prefix=str(args.prefix),
            render_fps=float(args.render_fps),
            slowdown=float(args.slowdown),
        ),
        legacy_args=args,
    )


def create_offline_parser() -> argparse.ArgumentParser:
    """Create the slim canonical CLI for the cloth+bunny offline runner."""

    parser = argparse.ArgumentParser(
        description="Canonical cloth+bunny offline runner with a slim public CLI."
    )
    parser.add_argument("--ir", type=Path, default=_default_cloth_ir())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="cloth_bunny_drop")
    parser.add_argument("--device", default=path_defaults.default_device())
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sim-dt", type=float, default=5.0e-5)
    parser.add_argument("--substeps", type=int, default=667)
    parser.add_argument("--drop-height", type=float, default=0.5)
    parser.add_argument("--cloth-shift-x", type=float, default=0.0)
    parser.add_argument("--cloth-shift-y", type=float, default=0.0)
    parser.add_argument("--rigid-mass", type=float, default=4000.0)
    parser.add_argument("--mass-spring-scale", type=float, default=None)
    parser.add_argument("--object-mass", type=float, default=None)
    parser.add_argument(
        "--force-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-render",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--slowdown", type=float, default=2.0)
    return parser


def parse_offline_args(argv: list[str] | None = None) -> argparse.Namespace:
    return create_offline_parser().parse_args(argv)


def _namespace_from_parser_defaults(parser: argparse.ArgumentParser) -> argparse.Namespace:
    values: dict[str, object] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        default = getattr(action, "default", argparse.SUPPRESS)
        if default is argparse.SUPPRESS:
            continue
        values[dest] = default
    return argparse.Namespace(**values)


def canonical_namespace_to_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Inflate the slim CLI into the legacy namespace consumed by existing logic."""

    from .offline import create_legacy_parser

    legacy = _namespace_from_parser_defaults(create_legacy_parser())
    legacy.ir = Path(args.ir).expanduser().resolve()
    legacy.out_dir = Path(args.out_dir).expanduser().resolve()
    legacy.prefix = str(args.prefix)
    legacy.device = str(args.device)
    legacy.frames = int(args.frames)
    legacy.sim_dt = float(args.sim_dt)
    legacy.substeps = int(args.substeps)
    legacy.drop_height = float(args.drop_height)
    legacy.cloth_shift_x = float(args.cloth_shift_x)
    legacy.cloth_shift_y = float(args.cloth_shift_y)
    legacy.rigid_mass = float(args.rigid_mass)
    legacy.mass_spring_scale = args.mass_spring_scale
    legacy.object_mass = args.object_mass
    legacy.force_diagnostic = bool(args.force_diagnostic)
    legacy.skip_render = bool(args.skip_render)
    legacy.render_fps = float(args.render_fps)
    legacy.slowdown = float(args.slowdown)
    return legacy


__all__ = [
    "ClothBunnyConfigBundle",
    "ClothBunnyDiagnosticConfig",
    "ClothBunnyOutputConfig",
    "ClothBunnyRuntimeConfig",
    "ClothBunnySceneConfig",
    "build_config_bundle",
    "canonical_namespace_to_legacy_args",
    "create_offline_parser",
    "parse_offline_args",
]
