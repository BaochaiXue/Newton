> status: local_only_secondary
> canonical_replacement: `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`
> owner_surface: `robot_rope_franka_tabletop_push_hero`
> last_reviewed: `2026-04-01`
> review_interval: `30d`
> update_rule: `Update only when the local BEST_RUN mirror target changes. Do not use this file as committed authority.`
> notes: Local convenience mirror only. Do not use this file as the committed source of truth for current/promoted run meaning.

# BEST_RUN Mirror (Local-Only)

Local-only convenience mirror for the currently promoted tabletop-push hero
bundle.

Committed authority for current/promoted run meaning lives in:

- `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`

Current committed promoted run id:

- `20260401_093102_fixeddt_c10_contactfix_cam`

Local source candidate:

- `../candidates/20260401_093102_fixeddt_c10_contactfix_cam`

Why this local mirror points here:

- native Newton Franka remains visible in all three views
- rope is visibly resting on a native Newton tabletop before the push
- contact is readable in the hero view without pedestal occlusion
- validation view still shows the honest full geometry
- rope moves after visible robot contact, not before
- visible contact starts earlier than in the older `c08` cut, reducing the impression that the rope moves before the finger arrives
- strict validator + truthful manual review both pass

Key summary numbers:

- `duration_s = 6.2`
- `first_contact_time_s = 2.2011`
- `contact_duration_s = 2.9348`
- `contact_started = true`
- `gripper_center_path_length_m = 0.3289980292`
- `min_clearance_min_m = -0.0103589874`
- `preroll_settle_pass = true`

Implementation note:

- the accepted tabletop hero keeps the robot native to Newton and the rope on
  the PhysTwin -> Newton bridge path, but uses a tabletop-only native
  joint-space waypoint controller because the bridge-layer tabletop IK path did
  not reliably hit the contact line under fixed `sim_dt = 5e-5`,
  `substeps = 667`
