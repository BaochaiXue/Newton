#!/usr/bin/env python3
"""Compatibility wrapper for legacy imports.

Bridge demos should import shared helpers from :mod:`bridge_shared` instead of
this legacy ``demo_*`` module name.
"""
from bridge_shared import *  # noqa: F401,F403
