"""
Shim for backward compatibility during refactoring: re-export apply API from runtime.apply.
"""

from __future__ import annotations

from flashinfer_bench.runtime.apply import (
    ApplyRuntime,
    apply,
)

__all__ = [
    "apply",
    "ApplyRuntime",
]
