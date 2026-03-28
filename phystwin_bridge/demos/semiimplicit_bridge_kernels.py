#!/usr/bin/env python3
"""Bridge-local compatibility shim for Newton SemiImplicit internal kernels.

Bridge demos occasionally need Newton's low-level SemiImplicit force kernels to
build custom profiling, realtime viewers, or ON/OFF diagnostic paths. Those
symbols currently live under Newton's private ``newton._src`` tree.

To avoid spreading private import paths across many demos, this shim exposes the
few internal kernels that bridge-side experiments rely on. If upstream Newton
moves those symbols, only this file should need updating.
"""
from __future__ import annotations

from bridge_bootstrap import ensure_bridge_runtime_paths

ensure_bridge_runtime_paths()

from newton._src.solvers.semi_implicit.kernels_body import eval_body_joint_forces  # noqa: E402
from newton._src.solvers.semi_implicit.kernels_contact import (  # noqa: E402
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from newton._src.solvers.semi_implicit.kernels_particle import (  # noqa: E402
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)

__all__ = [
    "eval_bending_forces",
    "eval_body_contact_forces",
    "eval_body_joint_forces",
    "eval_particle_body_contact_forces",
    "eval_particle_contact_forces",
    "eval_spring_forces",
    "eval_tetrahedra_forces",
    "eval_triangle_contact_forces",
    "eval_triangle_forces",
]
