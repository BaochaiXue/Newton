# Control Update Order Report

- Tabletop `joint_trajectory` mode is selected at [demo_robot_rope_franka.py:None](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:None)
- The desired joint target is written directly into `state_in.joint_q` at [demo_robot_rope_franka.py:2088](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2088)
- The desired joint velocity is written directly into `state_in.joint_qd` at [demo_robot_rope_franka.py:2089](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2089)
- The semi-implicit solver is then stepped with `control=None` at [demo_robot_rope_franka.py:2107](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2107)
- Immediately after the solver, the code writes `state_out.joint_q` back to the desired target at [demo_robot_rope_franka.py:2109](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2109)
- It also writes `state_out.joint_qd` back to the desired target velocity at [demo_robot_rope_franka.py:2110](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2110)
- Forward kinematics is recomputed from that overwritten state at [demo_robot_rope_franka.py:2111](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2111)

This update order means table contact has no durable path to create tracking error in the saved articulation state.
