# Contact Filter Report

- native tabletop support is added as a static box shape (`body=-1`) in the tabletop task.
- Franka colliders come from the URDF collision shapes, including multiple finger/hand boxes.
- this diagnostic does not find any hidden helper collider in the visible clip.

Filtering still needs to be validated against actual runtime contact response, not only scene construction.
