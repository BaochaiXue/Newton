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

This bundle is still a workflow scaffold. No accepted hero run has been
selected yet.

Current blocker:

- stable tabletop-smoke runs can now preroll-settle the rope, but the robot
  still does not make credible contact with the rope under the stable timestep
