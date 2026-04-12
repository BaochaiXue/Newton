#!/usr/bin/env python3
"""Public runtime surface for cloth+bunny rollouts."""
from __future__ import annotations

from typing import Any


def simulate_rollout(
    model,
    ir_obj: dict[str, Any],
    meta: dict[str, Any],
    args,
    n_obj: int,
    device: str,
) -> dict[str, Any]:
    """Run the canonical cloth+bunny simulation runtime."""

    from .offline import simulate

    return simulate(model, ir_obj, meta, args, n_obj, device)


__all__ = ["simulate_rollout"]
