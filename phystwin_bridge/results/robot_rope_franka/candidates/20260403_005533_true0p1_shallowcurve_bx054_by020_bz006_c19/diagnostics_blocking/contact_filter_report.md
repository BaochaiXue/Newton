# Contact Filter Report

- `SimConfig.shape_contacts=True` at [demo_robot_rope_franka.py:1512](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:1512)
- Native Franka URDF is added at [demo_robot_rope_franka.py:1543](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:1543)
- Native tabletop world box is added at [demo_robot_rope_franka.py:1778](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:1778)
- Broad-phase / narrow-phase contact generation is invoked through `model.collide(...)` at [demo_robot_rope_franka.py:2106](/home/xinjie/Newton_Connection/Newton/phystwin_bridge/demos/demo_robot_rope_franka.py:2106)
- No tabletop-specific collision filter was found in the bridge/demo layer that explicitly disables robot-table contact.
