# Rope Contact Thickness Report

- rope collision radius: `0.002600 m`
- rope collision diameter: `0.005200 m`
- rope render-only radius for this run: `0.002600 m`
- render/physical radius ratio: `1.000`
- diagnostic proxy radius: `0.033000 m`
- diagnostic proxy diameter: `0.066000 m`

The rope is physically much thicker than the current render if the render-only cap is small. In that case the solver can begin true fingertip contact while the visible rope still looks too thin.
