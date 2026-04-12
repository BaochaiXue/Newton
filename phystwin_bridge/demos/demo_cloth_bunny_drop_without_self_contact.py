#!/usr/bin/env python3
"""Compatibility shim for the legacy cloth+bunny offline demo path."""
from cloth_bunny.diagnostics import (
    build_full_process_force_sequence_from_rollout as _build_full_process_force_sequence_from_rollout,
    capture_force_snapshot_from_explicit_state as _capture_force_snapshot_from_explicit_state,
    finalize_force_diagnostic_artifacts as _finalize_force_diagnostic_artifacts,
    make_explicit_force_snapshot_context as _make_explicit_force_snapshot_context,
    render_force_artifacts_subprocess as _render_force_artifacts_subprocess,
    resolve_force_dump_dir as _resolve_force_dump_dir,
    write_force_render_bundle as _write_force_render_bundle,
)
from cloth_bunny.offline import *  # noqa: F401,F403
from cloth_bunny.outputs import build_summary, save_collision_force_rollout, save_scene_npz, save_summary_json
from cloth_bunny.runtime import simulate_rollout as simulate
from cloth_bunny.scene import (
    _apply_drag_correction_ignore_axis,
    _apply_particle_contact_scaling,
    _apply_shape_contact_scaling,
    _assign_shape_material_triplet,
    _box_signed_distance,
    _copy_object_only_ir,
    _default_cloth_ir,
    _effective_spring_scales,
    _mass_tag,
    _maybe_autoset_mass_spring_scale,
    _validate_scaling_args,
    build_model,
    load_bunny_mesh,
    load_ir,
    quat_to_rotmat,
)

parse_args = parse_legacy_args


if __name__ == "__main__":
    raise SystemExit(main_legacy())
