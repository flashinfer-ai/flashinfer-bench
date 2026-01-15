from flashinfer_bench.agents import FFI_PROMPT, FFI_PROMPT_SIMPLE
from flashinfer_bench.apply import apply, disable_apply, enable_apply
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
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
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.logging import configure_logging, get_logger
from flashinfer_bench.tracing import (
    TracingConfig,
    disable_tracing,
    enable_tracing,
    get_tracing_runtime,
)

try:
    from ._version import __version__, __version_tuple__
except Exception:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

__all__ = [
    # Main classes
    "Benchmark",
    "BenchmarkConfig",
    # Apply API
    "apply",
    "enable_apply",
    "disable_apply",
    # Tracing API
    "enable_tracing",
    "get_tracing_runtime",
    "disable_tracing",
    "TracingConfig",
    "Definition",
    "Solution",
    "Trace",
    "TraceSet",
    # Definition types
    "AxisConst",
    "AxisVar",
    "TensorSpec",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "SupportedLanguages",
    # Trace types
    "RandomInput",
    "SafetensorsInput",
    "Workload",
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "configure_logging",
    "get_logger",
    # FFI Prompts
    "FFI_PROMPT_SIMPLE",
    "FFI_PROMPT",
]
