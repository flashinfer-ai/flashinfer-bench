"""
Shim for backward compatibility during refactoring: re-export tracer API from runtime.tracer.
"""

from __future__ import annotations

from flashinfer_bench.runtime.tracer.tracer import (
    TraceEntry,
    Tracer,
    TracingRule,
    enable_tracing,
    end_tracing,
    get_tracer,
)

__all__ = [
    "TraceEntry",
    "Tracer",
    "TracingRule",
    "enable_tracing",
    "get_tracer",
    "end_tracing",
]
