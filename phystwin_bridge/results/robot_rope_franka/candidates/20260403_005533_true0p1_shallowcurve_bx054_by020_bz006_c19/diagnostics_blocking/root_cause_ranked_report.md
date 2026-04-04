# Root Cause Ranked Report

## Ranked Hypotheses

1. `H1/H2`: the current tabletop path is effectively kinematic because it writes joint state directly before the solver and overwrites post-solve state back to the target.
2. `H5`: the old promoted tabletop task was only a readable baseline and never claimed physical table-blocking.
3. `H4`: stiff target following may amplify the visible tunneling, but it is secondary to the overwrite semantics.
4. `H3`: robot-table collisions appear present in the scene; no bridge/demo-level evidence currently shows they are filtered out entirely.

## Answers

- Is the current tabletop path physically actuated or effectively kinematic?
  - Effectively kinematic in the promoted `joint_trajectory` path.
- Is table penetration caused primarily by state overwrite / control semantics?
  - Yes; penetration coexists with almost-zero target-vs-actual error.
- Are actual robot-table collisions present but being numerically overpowered?
  - Likely yes at least at the geometry level, but any solver reaction is being erased by the overwrite path.
- Is there any hidden helper?
  - No evidence of one in the bridge/demo layer.
- Can this be fixed at bridge/demo level without touching `Newton/newton/`?
  - Potentially yes, if the tabletop path is moved onto existing SemiImplicit drive/control surfaces instead of direct joint-state overwrite. This still needs implementation proof.
