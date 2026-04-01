# Robot Rope Franka Hero Results

Canonical result root for the native Newton Franka + native Newton table +
PhysTwin rope hero demo.

## Expected Layout

- `BEST_RUN/`
  - authoritative accepted run and its copied/pinned artifacts
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

## Status

This bundle is a workflow scaffold. No accepted hero run has been selected yet.
