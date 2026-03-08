#!/usr/bin/env python3
"""Dedicated killing demo: PhysTwin sloth drops into a Newton water pool with two floating bunnies.

This script is intentionally narrow:

- water only
- one imported PhysTwin soft body (sloth)
- two native Newton bunny rigid bodies
- a grounded 2m x 2m x 2m open pool

It reuses the already-validated native Newton MPM/two-way coupling path and the
existing PhysTwin soft-body import helpers, without modifying Newton core code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import sloth_zebra_mpm_killing_demo as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dedicated killing demo: a PhysTwin sloth soft body falls into a grounded "
            "Newton MPM water pool and hits two floating bunny rigid bodies."
        )
    )
    p.add_argument(
        "--sloth-ir",
        type=Path,
        default=Path(
            "Newton/phystwin_bridge/outputs/double_lift_sloth_normal/"
            "20260305_163731_full_parity/double_lift_sloth_normal_ir.npz"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="sloth_bunny_water_killing_demo")
    p.add_argument("--device", default=base.path_defaults.default_device())

    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--render-only-scene", type=Path, default=None)

    p.add_argument("--gravity-mag", type=float, default=9.8)
    p.add_argument("--ground-extent", type=float, default=8.0)

    p.add_argument("--pool-half-extent", type=float, default=1.0)
    p.add_argument("--pool-height", type=float, default=2.0)
    p.add_argument("--pool-wall-thickness", type=float, default=0.08)
    p.add_argument("--water-fill-height", type=float, default=1.5)

    p.add_argument(
        "--sloth-bottom-z",
        type=float,
        default=3.0,
        help="Absolute bottom height of the imported sloth above the ground plane.",
    )
    p.add_argument("--sloth-target-xy", type=float, nargs=2, default=(0.0, 0.0))
    p.add_argument(
        "--soft-mass-scale",
        type=float,
        default=0.001,
        help="Target per-particle mass scale for the imported PhysTwin sloth.",
    )

    p.add_argument("--bunny-count", type=int, default=2)
    p.add_argument("--bunny-spacing", type=float, default=0.42)
    p.add_argument("--rigid-mass", type=float, default=140.0)
    p.add_argument("--rigid-scale", type=float, default=0.16)
    p.add_argument("--float-submerge-fraction", type=float, default=0.56)
    p.add_argument("--rigid-body-mu", type=float, default=0.32)
    p.add_argument("--rigid-body-ke", type=float, default=1100.0)
    p.add_argument("--rigid-body-kd", type=float, default=130.0)
    p.add_argument("--rigid-body-kf", type=float, default=80.0)
    p.add_argument("--bunny-asset", default="bunny.usd")
    p.add_argument("--bunny-prim", default="/root/bunny")

    p.add_argument("--water-voxel-size", type=float, default=0.09)
    p.add_argument("--water-particles-per-cell", type=int, default=2)
    p.add_argument("--water-points-per-particle", type=int, default=6)
    p.add_argument("--water-density", type=float, default=950.0)
    p.add_argument("--water-young-modulus", type=float, default=3.0e4)
    p.add_argument("--water-poisson-ratio", type=float, default=0.49)
    p.add_argument("--water-damping", type=float, default=8.0)
    p.add_argument("--water-hardening", type=float, default=0.0)
    p.add_argument("--water-friction", type=float, default=0.0)
    p.add_argument("--water-yield-pressure", type=float, default=1.0e10)
    p.add_argument("--water-tensile-yield-ratio", type=float, default=1.0)
    p.add_argument("--water-yield-stress", type=float, default=0.0)
    p.add_argument("--water-air-drag", type=float, default=1.0)
    p.add_argument("--water-grid-type", choices=["fixed", "sparse"], default="fixed")
    p.add_argument("--water-max-iterations", type=int, default=50)
    p.add_argument("--render-mpm-particles", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--angular-damping", type=float, default=0.05)
    p.add_argument("--friction-smoothing", type=float, default=1.0)
    p.add_argument("--enable-tri-contact", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render-soft-springs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--render-edge-stride", type=int, default=14)
    p.add_argument("--particle-radius-vis-scale", type=float, default=2.2)
    p.add_argument("--particle-radius-vis-min", type=float, default=0.012)
    p.add_argument("--soft-mpm-force-scale", type=float, default=1.0)
    p.add_argument("--soft-mpm-max-dv-per-frame", type=float, default=0.25)

    p.add_argument("--show-tray", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tray-cutaway-front", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tray-base-thickness", type=float, default=0.0)
    p.add_argument("--tray-base-gap", type=float, default=0.0)
    p.add_argument("--hidden-floor-z", type=float, default=-0.5)
    p.add_argument("--camera-fov", type=float, default=44.0)
    p.add_argument("--render-fps", type=float, default=30.0)
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--screen-width", type=int, default=1280)
    p.add_argument("--screen-height", type=int, default=720)
    p.add_argument("--viewer-headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overlay-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label-font-size", type=int, default=28)
    p.add_argument("--views", default="wide")
    return p.parse_args()


def _pool_view_specs() -> list[base.ViewSpec]:
    return [
        base.ViewSpec("wide", "Hero Wide", yaw_deg=-34.0, pitch_deg=-12.0, distance=4.9),
        base.ViewSpec("impact", "Impact Close", yaw_deg=-12.0, pitch_deg=-9.0, distance=3.8),
    ]


def _parse_view_names(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("--views must contain at least one view name")
    known = {spec.name for spec in _pool_view_specs()}
    unknown = [item for item in items if item not in known]
    if unknown:
        raise ValueError(f"Unknown view(s): {unknown}. Known: {sorted(known)}")
    return items


def build_model(
    sloth_ir: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> tuple[newton.Model, base.DemoMeta]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.from_any("Z"), gravity=0.0)

    static_cfg = builder.default_shape_cfg.copy()
    static_cfg.mu = 0.32
    static_cfg.ke = 1.5e4
    static_cfg.kd = 6.0e2
    static_cfg.kf = 3.0e2
    static_cfg.is_visible = False

    pool_half = float(args.pool_half_extent)
    pool_height = float(args.pool_height)
    wall_t = float(args.pool_wall_thickness)

    # Infinite ground plane at z=0; this is also the pool floor.
    builder.add_shape_plane(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        width=0.0,
        length=0.0,
        body=-1,
        cfg=static_cfg,
        label="ground_plane",
    )

    wall_z = 0.5 * pool_height
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(pool_half + 0.5 * wall_t, 0.0, wall_z), wp.quat_identity()),
        hx=0.5 * wall_t,
        hy=pool_half + wall_t,
        hz=0.5 * pool_height,
        cfg=static_cfg,
        label="pool_wall_pos_x",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(-(pool_half + 0.5 * wall_t), 0.0, wall_z), wp.quat_identity()),
        hx=0.5 * wall_t,
        hy=pool_half + wall_t,
        hz=0.5 * pool_height,
        cfg=static_cfg,
        label="pool_wall_neg_x",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, pool_half + 0.5 * wall_t, wall_z), wp.quat_identity()),
        hx=pool_half + wall_t,
        hy=0.5 * wall_t,
        hz=0.5 * pool_height,
        cfg=static_cfg,
        label="pool_wall_pos_y",
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, -(pool_half + 0.5 * wall_t), wall_z), wp.quat_identity()),
        hx=pool_half + wall_t,
        hy=0.5 * wall_t,
        hz=0.5 * pool_height,
        cfg=static_cfg,
        label="pool_wall_neg_y",
    )

    object_specs: list[base.SoftSpec] = []
    object_specs.append(
        base._prepare_soft_object(
            builder=builder,
            ir=sloth_ir,
            ir_path=args.sloth_ir,
            name="sloth",
            color_rgb=(0.98, 0.66, 0.18),
            particle_start=0,
            target_xy=tuple(float(v) for v in args.sloth_target_xy),
            target_bottom_z=float(args.sloth_bottom_z),
            render_edge_stride=int(args.render_edge_stride),
            particle_radius_vis_scale=float(args.particle_radius_vis_scale),
            particle_radius_vis_min=float(args.particle_radius_vis_min),
            soft_mass_scale=float(args.soft_mass_scale),
        )
    )

    mesh, mesh_verts_local, mesh_tri_indices, mesh_render_edges, mesh_asset_path = base.load_bunny_mesh(
        args.bunny_asset,
        args.bunny_prim,
    )
    mesh_scale = float(args.rigid_scale)
    rigid_vertices = mesh_verts_local * mesh_scale
    mesh_extent = np.ptp(rigid_vertices, axis=0)

    target_sub = float(max(float(args.float_submerge_fraction) * mesh_extent[2], 0.03))
    target_bottom = float(args.water_fill_height - target_sub)

    rigid_cfg = builder.default_shape_cfg.copy()
    rigid_cfg.mu = float(args.rigid_body_mu)
    rigid_cfg.ke = float(args.rigid_body_ke)
    rigid_cfg.kd = float(args.rigid_body_kd)
    rigid_cfg.kf = float(args.rigid_body_kf)
    rigid_cfg.is_visible = False

    rigid_labels: list[str] = []
    offsets = np.linspace(
        -0.5 * float(args.bunny_spacing),
        0.5 * float(args.bunny_spacing),
        int(args.bunny_count),
        dtype=np.float32,
    )
    yaw_values = [14.0, -18.0]
    for idx, px in enumerate(offsets.tolist()):
        qx, qy, qz, qw = base._bunny_quat_xyzw(yaw_values[idx % len(yaw_values)])
        q = wp.quat(qx, qy, qz, qw)
        rotated = (rigid_vertices @ base.quat_to_rotmat((qx, qy, qz, qw)).T).astype(np.float32)
        z_min = float(rotated[:, 2].min())
        pos_z = float(target_bottom - z_min)
        label = f"water_bunny_{idx}"
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(px), 0.0, pos_z), q),
            mass=float(args.rigid_mass),
            inertia=wp.mat33(
                float(args.rigid_mass * (mesh_extent[1] ** 2 + mesh_extent[2] ** 2) / 12.0),
                0.0,
                0.0,
                0.0,
                float(args.rigid_mass * (mesh_extent[0] ** 2 + mesh_extent[2] ** 2) / 12.0),
                0.0,
                0.0,
                0.0,
                float(args.rigid_mass * (mesh_extent[0] ** 2 + mesh_extent[1] ** 2) / 12.0),
            ),
            lock_inertia=True,
            label=label,
        )
        builder.add_shape_mesh(body=body, mesh=mesh, scale=(mesh_scale, mesh_scale, mesh_scale), cfg=rigid_cfg)
        rigid_labels.append(label)

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -float(args.gravity_mag)))
    model.particle_mu = float(base.newton_import_ir.ir_scalar(sloth_ir, "contact_collide_fric", default=0.5))

    meta = base.DemoMeta(
        rigid_shape="bunny",
        mesh_verts_local=mesh_verts_local.astype(np.float32),
        mesh_tri_indices=mesh_tri_indices.astype(np.int32),
        mesh_render_edges=mesh_render_edges.astype(np.int32),
        mesh_scale=mesh_scale,
        mesh_asset_path=mesh_asset_path,
        object_specs=object_specs,
        rigid_labels=rigid_labels,
        rigid_cluster_center=np.array([0.0, 0.0, float(args.water_fill_height)], dtype=np.float32),
        water_level=float(args.water_fill_height),
        tray_floor_top_z=0.0,
        tray_wall_top_z=float(args.pool_height),
        hidden_floor_z=float(args.hidden_floor_z),
        scene_label="Scene: sloth + water + 2 bunny rigids",
    )
    return model, meta


def build_summary(
    args: argparse.Namespace,
    sim_data: dict[str, Any],
    meta: base.DemoMeta,
    panel_paths: list[Path],
) -> dict[str, Any]:
    return {
        "experiment": "sloth_bunny_water_killing_demo",
        "sloth_ir": str(args.sloth_ir.resolve()),
        "frames": int(sim_data["particle_q_all"].shape[0]),
        "sim_dt": float(sim_data["sim_dt"]),
        "substeps": int(sim_data["substeps"]),
        "frame_dt": float(sim_data["sim_dt"]) * float(sim_data["substeps"]),
        "wall_time_sec": float(sim_data.get("wall_time_sec", 0.0)),
        "gravity_mag": float(args.gravity_mag),
        "ground_extent": float(args.ground_extent),
        "pool_half_extent": float(args.pool_half_extent),
        "pool_height": float(args.pool_height),
        "water_fill_height": float(args.water_fill_height),
        "water_level": float(meta.water_level),
        "wall_thickness": float(args.pool_wall_thickness),
        "sloth_bottom_z": float(args.sloth_bottom_z),
        "soft_mass_scale": float(args.soft_mass_scale),
        "rigid_count": int(args.bunny_count),
        "rigid_mass": float(args.rigid_mass),
        "rigid_scale": float(args.rigid_scale),
        "water_voxel_size": float(args.water_voxel_size),
        "water_particles_per_cell": int(args.water_particles_per_cell),
        "water_points_per_particle": int(args.water_points_per_particle),
        "water_density": float(args.water_density),
        "water_young_modulus": float(args.water_young_modulus),
        "water_poisson_ratio": float(args.water_poisson_ratio),
        "water_damping": float(args.water_damping),
        "water_hardening": float(args.water_hardening),
        "water_friction": float(args.water_friction),
        "water_yield_pressure": float(args.water_yield_pressure),
        "water_tensile_yield_ratio": float(args.water_tensile_yield_ratio),
        "water_yield_stress": float(args.water_yield_stress),
        "water_air_drag": float(args.water_air_drag),
        "water_grid_type": str(args.water_grid_type),
        "panel_videos": [str(path) for path in panel_paths],
        "scene_label": meta.scene_label,
        "soft_objects": [
            {
                "name": spec.name,
                "ir_path": str(spec.ir_path),
                "particle_count": int(spec.particle_count),
                "mass_sum": float(spec.mass_sum),
                "drag_damping": float(spec.drag_damping),
                "bbox_extent": [float(v) for v in spec.bbox_extent],
                "strict_phystwin_export": bool(spec.strict_export),
                "spring_ke_mode": spec.spring_ke_mode,
                "spring_ke_rel_error_max": float(spec.spring_ke_rel_error_max),
                "collider_mu": float(spec.collider_mu),
                "collision_radius_mean": float(spec.collision_radius_mean),
            }
            for spec in meta.object_specs
        ],
    }


class Example:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.args.out_dir = self.args.out_dir.resolve()
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        self.args.show_tray = True
        self.args.tray_cutaway_front = True
        self.args.mpm_material_preset = "water"
        self.args.disable_sloth = False
        self.args.disable_zebra = True
        self.args.zebra_ir = self.args.sloth_ir

        # Reuse the existing MPM water builder; interpret water_level as pool fill height.
        self.args.water_level = float(args.water_fill_height)
        self.args.water_extent = float(args.pool_half_extent)
        self.args.water_depth = float(args.water_fill_height)
        self.args.water_wall_thickness = float(args.pool_wall_thickness)
        self.args.water_wall_height = float(args.pool_height)

        wp.init()
        self.device = base.newton_import_ir.resolve_device(self.args.device)

        self.sloth_ir = base._load_ir(self.args.sloth_ir)

        print("Building dedicated sloth+bunny+water killing model...", flush=True)
        self.model, self.meta = build_model(self.sloth_ir, self.args, self.device)
        print(
            "Soft<->MPM direct coupling disabled; sloth interacts with water through native Newton "
            "soft<->rigid contacts plus native rigid<->MPM two-way coupling.",
            flush=True,
        )

        self.sim_data: dict[str, Any] | None = None
        self.scene_npz: Path | None = None
        self.panel_paths: list[Path] = []
        self.view_names = _parse_view_names(self.args.views)

    def step(self):
        if self.sim_data is not None:
            return
        if self.args.render_only_scene is not None:
            self.scene_npz = self.args.render_only_scene.resolve()
            self.sim_data = base.load_scene_npz(self.scene_npz)
            print(f"Using existing scene NPZ: {self.scene_npz}", flush=True)
        else:
            print("Building native MPM water system...", flush=True)
            water = base.build_water_system(self.model, self.meta, self.args, self.device)
            print("Simulating dedicated killing scene...", flush=True)
            self.sim_data = base.simulate(self.model, self.meta, self.args, self.sloth_ir, water, self.device)
            self.scene_npz = base.save_scene_npz(self.args.out_dir, self.args.prefix, self.sim_data, self.meta)
            print(f"  Scene NPZ: {self.scene_npz}", flush=True)

    def render(self):
        assert self.sim_data is not None
        if self.args.skip_render:
            return
        selected_views = [view for view in _pool_view_specs() if view.name in self.view_names]
        print(f"Rendering {len(selected_views)} dedicated view(s)...", flush=True)
        for view in selected_views:
            panel_path = self.args.out_dir / f"{base._output_stem(self.args.prefix)}_{view.name}.mp4"
            base.render_view_mp4(
                model=self.model,
                sim_data=self.sim_data,
                meta=self.meta,
                args=self.args,
                device=self.device,
                view_spec=view,
                out_mp4=panel_path,
            )
            self.panel_paths.append(panel_path)
            print(f"  Saved: {panel_path}", flush=True)

    def finalize(self) -> int:
        assert self.sim_data is not None
        summary = build_summary(self.args, self.sim_data, self.meta, self.panel_paths)
        summary_path = self.args.out_dir / f"{base._output_stem(self.args.prefix)}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"  Summary: {summary_path}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    def run(self) -> int:
        self.step()
        self.render()
        return self.finalize()


def main() -> int:
    return Example(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
