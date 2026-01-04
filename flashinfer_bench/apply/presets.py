"""Builtin config registry presets."""

from functools import cache

from .config import ApplyConfigRegistry


@cache
def get_default_registry() -> ApplyConfigRegistry:
    """Get the default config registry (cached)."""
    return ApplyConfigRegistry()
