#!/usr/bin/env python3
"""Canonical rope-family bridge package.

The first cleanup pass keeps legacy top-level demo scripts as the behavioral
owners. Package modules here provide stable import/entrypoint names while the
large rope demos are migrated one slice at a time.
"""

__all__ = [
    "bunny_drop",
    "common",
    "control_viewer",
    "sloth_ground_contact",
    "two_ropes_ground_contact",
]
