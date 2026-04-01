# Robot Rope Franka Hero Results

Local result root for the native Newton Franka + native Newton table +
PhysTwin rope hero-demo workflow.

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
scripts/run_robot_rope_franka_hero.sh --slug <short_tag> [extra demo args...]
```

## Status

Accepted hero run:

- `candidates/20260401_081639_fixeddt_c08_gatepass/`

Promoted mirror:

- `BEST_RUN/`

Accepted claim:

- native Newton Franka
- native Newton tabletop support
- PhysTwin rope resting on the tabletop
- readable slow lateral push
- readable robot-caused rope deformation / sliding
- strict validator + truthful manual review both pass

Implementation note:

- the accepted run uses a tabletop-only native joint-space waypoint controller
  inside `demo_robot_rope_franka.py` because the earlier tabletop IK path did
  not reliably reach the contact line under fixed `sim_dt = 5e-5`,
  `substeps = 667`
