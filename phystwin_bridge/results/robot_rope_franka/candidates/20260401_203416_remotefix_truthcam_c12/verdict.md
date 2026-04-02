# Verdict — 20260401_203416_remotefix_truthcam_c12

- status: `PASS`
- claim boundary:
  - native Newton Franka + native Newton tabletop + PhysTwin rope tabletop push baseline
  - visible finger / claw contact is now conservatively readable
  - not full two-way coupling

## Why Pass

- No hidden helper was found.
- The rope is rendered at a thickness consistent with the physical rope collision radius, removing the earlier stand-off impression.
- Full-video review now supports: visible finger contact first, then rope deformation / motion.
- Debug contact reporting now uses actual finger-box contact rather than `finger_span` as the primary proof surface.
