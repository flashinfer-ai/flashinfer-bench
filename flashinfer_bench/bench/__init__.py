"""Benchmark execution engine."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .benchmark import Benchmark
    from .config import BenchmarkConfig
    from .error_taxonomy import EvaluationTaxonomy, classify_evaluation, classify_trace

_LAZY_IMPORTS = {
    "Benchmark": ("flashinfer_bench.bench.benchmark", "Benchmark"),
    "BenchmarkConfig": ("flashinfer_bench.bench.config", "BenchmarkConfig"),
    "EvaluationTaxonomy": ("flashinfer_bench.bench.error_taxonomy", "EvaluationTaxonomy"),
    "classify_evaluation": ("flashinfer_bench.bench.error_taxonomy", "classify_evaluation"),
    "classify_trace": ("flashinfer_bench.bench.error_taxonomy", "classify_trace"),
}

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "EvaluationTaxonomy",
    "classify_evaluation",
    "classify_trace",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'flashinfer_bench.bench' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
