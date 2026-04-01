> status: local-only
> canonical_replacement: `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`
> owner_surface: `robot_rope_franka_tabletop_push_hero`
> last_reviewed: `2026-04-01`
> review_interval: `30d`
> update_rule: `Update only for local navigation or bundle-layout changes. Do not use this page as committed authority.`
> notes: Local bundle-root overview only. Committed promoted-run meaning lives in results_meta.

# Robot Rope Franka Hero Results

Local result root for the native Newton Franka + native Newton table +
PhysTwin rope hero-demo workflow.

Do not use this page as the committed source of truth for current/promoted run
meaning. Use:

- `results_meta/tasks/robot_rope_franka_tabletop_push_hero.json`
- `results_meta/INDEX.md`
- `results_meta/LATEST.md`

## Expected Layout

- `BEST_RUN/`
  - local convenience slot for an accepted run once one exists
- `candidates/<timestamp>_<short_tag>/`
  - each candidate run folder with:
    - `manifest.json`
    - `run_command.txt`
    - `metrics.json`
    - `validation.md`
    - `hero_presentation.mp4`
    - `hero_debug.mp4`
    - `validation_camera.mp4`
    - `contact_sheet.png`
    - `keyframes/`

## Validation

Use the strict hero validator:

```bash
python scripts/validate_robot_rope_franka_hero.py <candidate_run_dir> \
  --manual-review-json <candidate_run_dir>/manual_review.json
```

The validator writes:

- `ffprobe.json`
- `metrics.json`
- `validation.md`
- `contact_sheet.png`
- `keyframes/`
- `manifest.json` when absent
- `run_command.txt` when absent

Wrapper for the full three-video local candidate flow:

```bash
scripts/run_robot_rope_franka_tabletop_hero.sh --tag <short_tag> [extra demo args...]
```

## Current Local Convenience Pointers

Current committed promoted run id:

- `20260401_081639_fixeddt_c08_gatepass`

Local candidate directory:

- `candidates/20260401_081639_fixeddt_c08_gatepass/`

Local convenience mirror:

- `BEST_RUN/`

Claim boundary mirrored from the committed registry:

- native Newton Franka
- native Newton tabletop support
- PhysTwin rope resting on the tabletop
- readable slow lateral push
- readable robot-caused rope deformation / sliding
- strict validator + truthful manual review both pass

Implementation note kept for reruns:

- the accepted run uses a tabletop-only native joint-space waypoint controller
  inside `demo_robot_rope_franka.py` because the earlier tabletop IK path did
  not reliably reach the contact line under fixed `sim_dt = 5e-5`,
  `substeps = 667`
