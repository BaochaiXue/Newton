#!/usr/bin/env python3
"""Shared runtime bootstrap for PhysTwin bridge demos.

This module centralizes the demo-side environment setup that used to be copied
across many scripts:

- resolve bridge/core/newton paths relative to the repository layout
- add the required roots to :mod:`sys.path`
- load bridge core helpers from ``tools/core``
- expose stable demo-side handles for ``newton``, ``newton_import_ir``, and
  ``path_defaults``

The intent is to keep path bootstrapping and dynamic module loading in one
place instead of repeating slightly different variants in every demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warp as wp

from bridge_shared import BRIDGE_ROOT, CORE_DIR, load_core_module

WORKSPACE_ROOT = BRIDGE_ROOT.parents[1]
THIS_DIR = Path(__file__).resolve().parent
NEWTON_PY_ROOT = WORKSPACE_ROOT / "Newton" / "newton"


def ensure_bridge_runtime_paths() -> None:
    """Ensure the bridge core path and Newton Python root are importable."""
    for path in (CORE_DIR, NEWTON_PY_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_bridge_runtime_paths()

if not hasattr(wp, "quat_twist"):
    @wp.func
    def _quat_twist_compat(axis: wp.vec3, q: wp.quat) -> wp.quat:
        ax = wp.normalize(axis)
        imag = wp.vec3(q[0], q[1], q[2])
        proj = ax * wp.dot(imag, ax)
        return wp.normalize(wp.quat(proj[0], proj[1], proj[2], q[3]))

    wp.quat_twist = _quat_twist_compat

if not hasattr(wp, "quat_to_euler"):
    @wp.func
    def _quat_to_euler_compat(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = wp.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = wp.asin(wp.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = wp.atan2(siny_cosp, cosy_cosp)
        return wp.vec3(roll, pitch, yaw)

    wp.quat_to_euler = _quat_to_euler_compat

path_defaults = load_core_module(
    "phystwin_bridge_path_defaults",
    CORE_DIR / "path_defaults.py",
)
newton_import_ir = load_core_module(
    "phystwin_bridge_newton_import_ir",
    CORE_DIR / "newton_import_ir.py",
)

import newton  # noqa: E402

__all__ = [
    "BRIDGE_ROOT",
    "CORE_DIR",
    "NEWTON_PY_ROOT",
    "THIS_DIR",
    "WORKSPACE_ROOT",
    "ensure_bridge_runtime_paths",
    "newton",
    "newton_import_ir",
    "path_defaults",
]
