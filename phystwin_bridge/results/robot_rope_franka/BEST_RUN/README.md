> status: local-only
> canonical_replacement: `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`
> owner_surface: `robot_rope_franka_tabletop_push_hero`
> last_reviewed: `2026-04-01`
> review_interval: `30d`
> update_rule: `Update only if the local BEST_RUN mirror is refreshed. Do not use this page as committed authority.`
> notes: Local convenience mirror for the committed promoted run. The registry JSON is the authority surface.

# Local BEST_RUN Mirror

Committed promoted run id:

- `20260401_081639_fixeddt_c08_gatepass`

Committed authority:

- `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`

Local source candidate:

- `../candidates/20260401_081639_fixeddt_c08_gatepass`

Why this local mirror exists:

- native Newton Franka remains visible in all three views
- rope is visibly resting on a native Newton tabletop before the push
- contact is readable in the hero view without pedestal occlusion
- validation view still shows the honest full geometry
- rope moves after visible robot contact, not before
- strict validator + truthful manual review both pass

Local summary numbers copied from the promoted bundle:

- `duration_s = 6.2`
- `first_contact_time_s = 2.56795`
- `contact_duration_s = 2.6013`
- `contact_started = true`
- `gripper_center_path_length_m = 0.3496762514`
- `min_clearance_min_m = -0.0113213938`
- `preroll_settle_pass = true`

Implementation note mirrored for reruns:

- the accepted tabletop hero keeps the robot native to Newton and the rope on
  the PhysTwin -> Newton bridge path, but uses a tabletop-only native
  joint-space waypoint controller because the bridge-layer tabletop IK path did
  not reliably hit the contact line under fixed `sim_dt = 5e-5`,
  `substeps = 667`
