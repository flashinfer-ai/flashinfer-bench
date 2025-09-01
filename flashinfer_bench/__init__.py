"""FlashInfer Bench - A benchmarking framework for GPU kernel implementations."""

from flashinfer_bench.benchmark import Benchmark
from flashinfer_bench.benchmark_config import BenchmarkConfig
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

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "Benchmark",
    "BenchmarkConfig",
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
]
