# 20260401_203416_remotefix_truthcam_c12

Truth-fixed tabletop hero candidate promoted from the remote-interaction
root-cause investigation.

## Why This Candidate Exists

- The earlier promoted tabletop hero rendered the rope much thinner than the
  physical rope collision thickness, which created a stand-off / remote-contact
  impression.
- This candidate fixes that visual-truth mismatch, switches tabletop contact
  reporting to actual finger-box contact, and tightens the hero framing around
  the real contact patch.

## Key Verdict

- strict validator: `PASS`
- truthful manual review: `PASS`
- full-video multimodal review: `PASS`
- hidden helper: `NO`
- claim boundary:
  - native Newton Franka + native Newton tabletop + PhysTwin rope tabletop push
  - readable fingertip-driven contact baseline
  - not full two-way coupling

## Main Artifacts

- `hero_presentation.mp4`
- `hero_debug.mp4`
- `validation_camera.mp4`
- `summary.json`
- `metrics.json`
- `validation.md`
- `diagnostics/`
