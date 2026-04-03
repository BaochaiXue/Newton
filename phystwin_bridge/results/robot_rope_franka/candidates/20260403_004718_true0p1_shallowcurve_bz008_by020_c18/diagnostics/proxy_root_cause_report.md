# Proxy Root Cause Report

- summary contact proxy counts: `{'right_box_0': 5, 'right_box_2': 21, 'right_tip_box': 29}`
- summary contact peak proxy: `right_tip_box`
- proxy finger_span first contact frame: `83`
- actual any-finger-box first contact frame: `98`
- actual left-tip first contact frame: `None`
- rope lateral motion frame (3.0 mm COM-xy): `98`
- rope deformation frame (5.0 mm mid-rope): `100`

If rope motion/deformation begins before or at roughly the same time as visible near-touch, but the accepted proof is still `finger_span`, then the diagnostic semantics are looser than the fingertip-contact claim.
