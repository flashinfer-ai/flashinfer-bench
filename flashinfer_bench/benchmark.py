"""
Shim for backward compatibility during refactoring: re-export benchmark API from benchmark.benchmark.
"""

from __future__ import annotations

from flashinfer_bench.benchmark.benchmark import (
    Benchmark,
    build_solution,
)

__all__ = [
    "Benchmark",
    "build_solution",
]
