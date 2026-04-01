# phystwin_bridge Status

## Current Focus

- Hero-demo validation pipeline for the native Newton Franka + native Newton
  table + PhysTwin rope workflow.
- New strict validator:
  - `scripts/validate_robot_rope_franka_hero.py`

## Canonical Result Root

- `Newton/phystwin_bridge/results/robot_rope_franka/`

## Expected Candidate Layout

- `candidates/<timestamp>_<short_tag>/`
  - `manifest.json`
  - `run_command.txt`
  - `metrics.json`
  - `validation.md`
  - `ffprobe.json`
  - `contact_sheet.png`
  - `keyframes/`

## Current State

- Validation workflow scaffolded.
- No accepted hero run has been selected yet.
