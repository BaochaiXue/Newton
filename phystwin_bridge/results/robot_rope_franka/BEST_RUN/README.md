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

- `20260401_203416_remotefix_truthcam_c12`

Local source candidate:

- `../candidates/20260401_203416_remotefix_truthcam_c12`

Why this local mirror points here:

- the rope is now rendered at a thickness consistent with its physical contact radius
- the visible Franka finger itself is clearly the contactor in the hero view
- debug contact reporting is grounded in actual finger-box contact rather than `finger_span`
- validation view preserves the honest full geometry while still showing readable contact
- strict validator + truthful manual review + fail-closed full-video review all pass

Key summary numbers:

- `duration_s = 6.2`
- `first_contact_time_s = 1.6675`
- `actual_finger_box_first_contact_time_s = 1.6675`
- `contact_duration_s = 2.96815`
- `contact_peak_proxy = right_tip_box`
- `min_clearance_min_m = -0.0128063839`
- `rope_surface_clearance_to_table_min_m = -0.0048188120`
- `preroll_settle_pass = true`

Implementation note:

- the accepted tabletop hero keeps the robot native to Newton and the rope on
  the PhysTwin -> Newton bridge path, while still using the tabletop-only
  native joint-space waypoint controller under fixed `sim_dt = 5e-5`,
  `substeps = 667`
- the remote-interaction fix is a truth fix, not a hidden-helper trick:
  - honest rope render thickness
  - actual finger-box contact reporting
  - tighter hero framing around the real contact patch
