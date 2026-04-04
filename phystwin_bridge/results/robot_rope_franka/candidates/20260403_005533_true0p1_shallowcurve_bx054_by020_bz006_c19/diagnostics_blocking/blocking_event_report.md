# Blocking Event Report

- first robot-table contact frame: `50`
- first robot-table contact time: `1.667500` s
- robot-table contact duration: `3.7685500000000007` s
- worst finger-box penetration: `-0.01999959722161293` m
- mean target-vs-actual EE error during contact: `2.5122469651250867e-06` m
- max target-vs-actual EE error during contact: `4.6442164602922276e-06` m
- mean actual normal speed into table during contact: `0.01011337898671627` m/s

Interpretation:

- A blocked physical controller should show a growing target-vs-actual error once table contact occurs.
- The current run instead shows deep penetration with nearly zero target-vs-actual error.
- That is consistent with effective kinematic overwrite, not contact-limited actuation.
