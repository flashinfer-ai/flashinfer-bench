from __future__ import annotations

from collections import defaultdict
from typing import List

from flashinfer_bench.compile import get_registry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet
from flashinfer_bench.logging import get_logger

from .config import BenchmarkConfig
from .runner import MultiProcessRunner


class Benchmark:
    """Benchmark execution engine for FlashInfer-Bench kernel solutions.

    It runs the solutions against the workloads, and stores the results back to the trace set.
    This class manages the GPU resources and will allocate multiple processes to run the solutions
    in parallel.
    """

    def __init__(self, trace_set: TraceSet, config: BenchmarkConfig = None) -> None:
        """Initialize the Benchmark with a TraceSet and configuration.

        Parameters
        ----------
        trace_set : TraceSet
            The dataset containing definitions, solutions, and workloads to benchmark.
        config : BenchmarkConfig, optional
            Configuration parameters for benchmark execution, by default BenchmarkConfig().

        Raises
        ------
        ValueError
            If log_level is not one of the valid logging levels.
        """
        # Dataset and configuration
        self._trace_set = trace_set
        self._config = config if config is not None else BenchmarkConfig()

        # Setup logger
        self._logger = get_logger("Benchmark")

        # Setup runner
        self._runner = MultiProcessRunner(self._logger, self._config.log_dir)

        # Setup registry
        self._registry = get_registry()

    def get_trace_set(self) -> TraceSet:
        """Get the TraceSet associated with this benchmark.

        Returns
        -------
        TraceSet
            The TraceSet containing definitions, solutions, and workloads.
        """
        return self._trace_set

    def run_all(self, dump_traces: bool = True) -> TraceSet:
        """Run benchmark for all solutions in the trace set.

        Parameters
        ----------
        dump_traces : bool, optional
            If True, store traces to the trace set and in the disk.

        Returns
        -------
        TraceSet
            A new TraceSet containing the original data plus the execution traces
            from this benchmark run. The traces are organized by definition name.
        """
        result_traces: List[Trace] = []

        for def_name, defn in self._trace_set.definitions.items():
            sols = self._trace_set.solutions.get(def_name, [])
            if not sols:
                self._logger.warning(f"No solutions found for def={def_name}, skipping definition")
                continue

            self._logger.info(f"Processing definition: {def_name} with {len(sols)} solutions")

            workloads = self._trace_set.workloads.get(def_name, [])

            for wl_trace in workloads:
                wl = wl_trace.workload

                try:
                    results = self._runner.run_workload(
                        defn, wl, sols, self._config, self._trace_set.root
                    )
                except RuntimeError as e:
                    self._logger.error(f"Failed to run workload {wl.uuid}: {e}")
                    continue

                for sol_name, ev in results.items():
                    trace = Trace(
                        definition=def_name, workload=wl, solution=sol_name, evaluation=ev
                    )

                    result_traces.append(trace)

                    if ev.status == EvaluationStatus.PASSED:
                        self._logger.info(
                            f"Solution '{sol_name}' for workload {wl.uuid}: PASSED with "
                            f"{ev.performance.speedup_factor:.2f}x speedup"
                        )
                    else:
                        self._logger.warning(
                            f"Solution '{sol_name}' for workload {wl.uuid}: {ev.status.value}"
                        )

        traces_by_def = defaultdict(list)
        for trace in result_traces:
            traces_by_def[trace.definition].append(trace)

        # Create a new TraceSet with the results
        result_traceset = TraceSet(
            root=self._trace_set.root,
            definitions=self._trace_set.definitions.copy(),
            solutions=self._trace_set.solutions.copy(),
            workloads=self._trace_set.workloads.copy(),
            traces=dict(traces_by_def),
        )

        if dump_traces:
            self._trace_set.add_traces(result_traces)

        return result_traceset
