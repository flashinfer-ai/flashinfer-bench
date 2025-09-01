from __future__ import annotations

from flashinfer_bench.benchmark.config import BenchmarkConfig
from flashinfer_bench.compile.builder import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import Evaluation, Workload


class Runner:
    """Placeholder: run trials/timing/compare/aggregate."""

    def run(
        self,
        defn: Definition,
        impl: Runnable,
        ref: Runnable,
        workload: Workload,
        cfg: BenchmarkConfig,
    ) -> Evaluation:
        raise NotImplementedError
