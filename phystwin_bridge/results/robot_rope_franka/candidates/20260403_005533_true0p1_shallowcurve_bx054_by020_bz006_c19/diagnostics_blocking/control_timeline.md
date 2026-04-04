# Control Timeline

Per tabletop substep in the current `joint_trajectory` path:

1. Compute desired joint target from the hard-coded phase waypoint line.
2. Write desired joint position/velocity directly into `state_in`.
3. Run collision detection.
4. Run the semi-implicit solver with `control=None`.
5. Overwrite `state_out` joint position/velocity back to the desired target.
6. Recompute FK from the overwritten target state and swap buffers.

Because step 5 happens every substep, contact cannot accumulate as persistent articulation lag.
