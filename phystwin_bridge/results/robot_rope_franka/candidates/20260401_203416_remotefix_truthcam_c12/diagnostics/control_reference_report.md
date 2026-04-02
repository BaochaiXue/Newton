# Control Reference Report

- tabletop_control_mode: `joint_trajectory`
- ee body label: `fr3/fr3_link7`
- target semantics: `joint_trajectory_fk_gripper_center`
- mean target->left finger center offset: `0.040000 m`
- mean target->right finger center offset: `0.040000 m`
- mean target->gripper center offset: `0.000003 m`

The promoted tabletop path is joint-space, so the reported target is the FK gripper-center path rather than an explicit fingertip or finger-pad target.
