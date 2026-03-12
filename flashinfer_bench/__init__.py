from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

try:
    from ._version import __version__, __version_tuple__
except Exception:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

if TYPE_CHECKING:
    from flashinfer_bench.agents import FFI_PROMPT, FFI_PROMPT_SIMPLE
    from flashinfer_bench.apply import (
        ApplyConfig,
        ApplyConfigRegistry,
        apply,
        disable_apply,
        enable_apply,
    )
    from flashinfer_bench.bench import (
        Benchmark,
        BenchmarkConfig,
        EvaluationTaxonomy,
        classify_evaluation,
        classify_trace,
    )
    from flashinfer_bench.data import (
        AxisConst,
        AxisVar,
        BuildSpec,
        Correctness,
        Definition,
        Environment,
        Evaluation,
        EvaluationStatus,
        Performance,
        RandomInput,
        SafetensorsInput,
        Solution,
        SourceFile,
        SupportedBindings,
        SupportedLanguages,
        TensorSpec,
        Trace,
        TraceSet,
        Workload,
    )
    from flashinfer_bench.tracing import (
        TracingConfig,
        TracingConfigRegistry,
        disable_tracing,
        enable_tracing,
    )

_LAZY_IMPORTS = {
    "Benchmark": ("flashinfer_bench.bench", "Benchmark"),
    "BenchmarkConfig": ("flashinfer_bench.bench", "BenchmarkConfig"),
    "EvaluationTaxonomy": ("flashinfer_bench.bench", "EvaluationTaxonomy"),
    "classify_evaluation": ("flashinfer_bench.bench", "classify_evaluation"),
    "classify_trace": ("flashinfer_bench.bench", "classify_trace"),
    "apply": ("flashinfer_bench.apply", "apply"),
    "enable_apply": ("flashinfer_bench.apply", "enable_apply"),
    "disable_apply": ("flashinfer_bench.apply", "disable_apply"),
    "ApplyConfig": ("flashinfer_bench.apply", "ApplyConfig"),
    "ApplyConfigRegistry": ("flashinfer_bench.apply", "ApplyConfigRegistry"),
    "enable_tracing": ("flashinfer_bench.tracing", "enable_tracing"),
    "disable_tracing": ("flashinfer_bench.tracing", "disable_tracing"),
    "TracingConfig": ("flashinfer_bench.tracing", "TracingConfig"),
    "TracingConfigRegistry": ("flashinfer_bench.tracing", "TracingConfigRegistry"),
    "Definition": ("flashinfer_bench.data", "Definition"),
    "Solution": ("flashinfer_bench.data", "Solution"),
    "Trace": ("flashinfer_bench.data", "Trace"),
    "TraceSet": ("flashinfer_bench.data", "TraceSet"),
    "AxisConst": ("flashinfer_bench.data", "AxisConst"),
    "AxisVar": ("flashinfer_bench.data", "AxisVar"),
    "TensorSpec": ("flashinfer_bench.data", "TensorSpec"),
    "SourceFile": ("flashinfer_bench.data", "SourceFile"),
    "BuildSpec": ("flashinfer_bench.data", "BuildSpec"),
    "SupportedBindings": ("flashinfer_bench.data", "SupportedBindings"),
    "SupportedLanguages": ("flashinfer_bench.data", "SupportedLanguages"),
    "RandomInput": ("flashinfer_bench.data", "RandomInput"),
    "SafetensorsInput": ("flashinfer_bench.data", "SafetensorsInput"),
    "Workload": ("flashinfer_bench.data", "Workload"),
    "Correctness": ("flashinfer_bench.data", "Correctness"),
    "Performance": ("flashinfer_bench.data", "Performance"),
    "Environment": ("flashinfer_bench.data", "Environment"),
    "Evaluation": ("flashinfer_bench.data", "Evaluation"),
    "EvaluationStatus": ("flashinfer_bench.data", "EvaluationStatus"),
    "FFI_PROMPT_SIMPLE": ("flashinfer_bench.agents", "FFI_PROMPT_SIMPLE"),
    "FFI_PROMPT": ("flashinfer_bench.agents", "FFI_PROMPT"),
}

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "EvaluationTaxonomy",
    "classify_evaluation",
    "classify_trace",
    "apply",
    "enable_apply",
    "disable_apply",
    "ApplyConfig",
    "ApplyConfigRegistry",
    "enable_tracing",
    "disable_tracing",
    "TracingConfig",
    "TracingConfigRegistry",
    "Definition",
    "Solution",
    "Trace",
    "TraceSet",
    "AxisConst",
    "AxisVar",
    "TensorSpec",
    "SourceFile",
    "BuildSpec",
    "SupportedBindings",
    "SupportedLanguages",
    "RandomInput",
    "SafetensorsInput",
    "Workload",
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "FFI_PROMPT_SIMPLE",
    "FFI_PROMPT",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'flashinfer_bench' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
