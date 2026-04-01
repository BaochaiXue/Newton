# Demos Subtree Rules

This subtree contains recurring demo-family entry points.

## Rule

Prefer bridge-layer task presets, validators, and wrappers over one-off demo
forks.

## When Editing Demos

- keep `Newton/newton/` untouched
- preserve claim boundaries in the task docs
- if a demo changes accepted outputs or command semantics, update:
  - the relevant task page
  - the matching `tasks/status/*.md`
  - `results_meta/tasks/*.json` if an authoritative run is affected

## Video Tasks

For meeting-facing media, automatic QC is not enough.
Pair demo changes with skeptical review bundle preparation and a separate video
audit.
