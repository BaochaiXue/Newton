# Suspected Kinematic Override

- During table-contact frames, `ee_target_to_actual_error_during_block_mean_m = 2.5122469651250867e-06`
- During table-contact frames, `ee_target_to_actual_error_during_block_max_m = 4.6442164602922276e-06`
- Worst sampled finger-box penetration is `robot_table_penetration_min_m = -0.01999959722161293`

These two facts together are the key kinematic-override signature:

- the hand penetrates the table materially
- but actual end-effector motion remains almost identical to the target

A physically blocked controller would allow target-vs-actual error to grow under table contact instead.
