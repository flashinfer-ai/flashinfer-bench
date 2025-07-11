"""FlashInfer Bench - A benchmarking framework for GPU kernel implementations."""

from flashinfer_bench.benchmark import Benchmark
from flashinfer_bench.benchmark_config import BenchmarkConfig
from flashinfer_bench.definition import Definition
from flashinfer_bench.solution import Solution
from flashinfer_bench.trace import Trace
from flashinfer_bench.trace_set import TraceSet

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "Definition", 
    "Solution",
    "Trace",
    "TraceSet",
]