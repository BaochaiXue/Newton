# Hidden Helper Audit

- hidden_helper_exists: `false`
- hidden_helper_affects_visible_clip: `false`

## Notes
- Physics path includes only rope particles/springs, native Franka URDF colliders, and the native tabletop box.
- Tabletop rope particles remain active; there is no visible-clip support patch or hidden pusher active in the tabletop task.
- The hero pedestal hide flag only affects viewer decorations, not solver geometry.
