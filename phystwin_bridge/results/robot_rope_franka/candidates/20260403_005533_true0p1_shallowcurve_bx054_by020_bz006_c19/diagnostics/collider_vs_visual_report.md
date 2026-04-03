# Collider Vs Visual Report

- The solver uses native Franka URDF collision shapes plus the native tabletop box.
- The visible finger mesh is not identical to the solver collider; the fingers use multiple box colliders.
- Left finger relevant box colliders: `4`
- Right finger relevant box colliders: `4`

See `collider_inventory.json` for the exact shape inventory.
