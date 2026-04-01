# Native Franka Tabletop Push Hero Results

This folder is the phystwin_bridge-local canonical result root for the meeting
hero demo:

- native Newton Franka
- native Newton tabletop
- PhysTwin-loaded rope
- slow, readable robot push

## Layout

- `BEST_RUN/`
  - exact promoted assets only
- `candidates/<timestamp>_<short_tag>/`
  - every serious candidate with metrics and validation

## Required Candidate Files

- `manifest.json`
- `run_command.txt`
- `metrics.json`
- `validation.md`
- `hero_presentation.mp4`
- `hero_debug.mp4`
- `validation_camera.mp4`
- `contact_sheet.png`
- `keyframes/`

## Promotion Rule

Do not copy a candidate into `BEST_RUN/` unless it passes the visual and
artifact hard gates in the task page and local status notes.
