# BEST RUN

Promoted run: `20260401_081639_fixeddt_c08_gatepass`

Source candidate:

- `/home/xinjie/Newton_Connection/Newton/phystwin_bridge/results/robot_rope_franka/candidates/20260401_081639_fixeddt_c08_gatepass`

Why this run won:

- native Newton Franka remains visible in all three views
- rope is visibly resting on a native Newton tabletop before the push
- contact is readable in the hero view without pedestal occlusion
- validation view still shows the honest full geometry
- rope moves after visible robot contact, not before
- strict validator + truthful manual review both pass

Key summary numbers:

- `duration_s = 6.2`
- `first_contact_time_s = 3.23495`
- `contact_duration_s = 1.8676`
- `contact_started = true`
- `gripper_center_path_length_m = 0.3496762514`
- `min_clearance_min_m = -0.0090658087`
- `preroll_settle_pass = true`

Implementation note:

- the accepted tabletop hero keeps the robot native to Newton and the rope on
  the PhysTwin -> Newton bridge path, but uses a tabletop-only native
  joint-space waypoint controller because the bridge-layer tabletop IK path did
  not reliably hit the contact line under fixed `sim_dt = 5e-5`,
  `substeps = 667`
