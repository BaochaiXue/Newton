#!/usr/bin/env python3
"""Compatibility shim for the legacy cloth+bunny realtime viewer path."""
from cloth_bunny.example import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
