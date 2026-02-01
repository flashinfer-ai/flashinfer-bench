"""Main benchmark orchestration class."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Set, Tuple

from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet

from .config import BenchmarkConfig
from .runner import IsolatedRunner, PersistentRunner

logger = logging.getLogger(__name__)


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

        # Setup registry
        self._registry = BuilderRegistry.get_instance()

        # Create runner
        if self._config.use_isolated_runner:
            self._runner = IsolatedRunner(self._config.log_dir)
        else:
            self._runner = PersistentRunner(self._config.log_dir)

    def get_trace_set(self) -> TraceSet:
        """Get the TraceSet associated with this benchmark.

        Returns
        -------
        TraceSet
            The TraceSet containing definitions, solutions, and workloads.
        """
        return self._trace_set

    def run_all(self, dump_traces: bool = True, resume: bool = False) -> TraceSet:
        """Run benchmark for all solutions in the trace set.

        Parameters
        ----------
        dump_traces : bool, optional
            If True, store traces to the trace set and in the disk.
        resume : bool, optional
            If True, skip solutions that have already been evaluated for each workload.

        Returns
        -------
        TraceSet
            A new TraceSet containing the original data plus the execution traces
            from this benchmark run. The traces are organized by definition name.
        """
        result_traces: List[Trace] = []

        definitions_to_run = self._trace_set.definitions.items()
        if self._config.definitions is not None:
            definitions_to_run = [
                (name, definition)
                for name, definition in definitions_to_run
                if name in self._config.definitions
            ]
            provided_defs = set(self._config.definitions)
            existing_defs = set(self._trace_set.definitions.keys())
            missing_defs = provided_defs - existing_defs
            if missing_defs:
                logger.warning(f"Definitions not found in trace set: {sorted(missing_defs)}")

        for def_name, definition in definitions_to_run:
            sols = self._trace_set.solutions.get(def_name, [])
            if not sols:
                logger.warning(f"No solutions found for def={def_name}, skipping definition")
                continue

            if self._config.solutions is not None:
                sols = [s for s in sols if s.name in self._config.solutions]
                if not sols:
                    logger.info(f"No matching solutions for def={def_name} after filtering")
                    continue

            logger.info(f"Processing definition: {def_name} with {len(sols)} solutions")

            existing_traces: Set[Tuple[str, str]] = set()  # (workload_uuid, solution_name)
            if resume:
                existing_def_traces = self._trace_set.traces.get(def_name, [])
                for trace in existing_def_traces:
                    if trace.solution and trace.evaluation:
                        existing_traces.add((trace.workload.uuid, trace.solution))
                if existing_traces:
                    logger.info(f"Found {len(existing_traces)} existing traces for def={def_name}")

            workloads = self._trace_set.workloads.get(def_name, [])
            def_traces: List[Trace] = []

            for wl_trace in workloads:
                workload = wl_trace.workload

                sols_to_run = sols
                if resume:
                    sols_to_run = [
                        s for s in sols if (workload.uuid, s.name) not in existing_traces
                    ]

                if not sols_to_run:
                    logger.info(f"All solutions already evaluated for workload {workload.uuid}")
                    continue

                try:
                    results = self._runner.run_workload(
                        definition, workload, sols_to_run, self._config, self._trace_set.root
                    )
                except RuntimeError as e:
                    logger.error(f"Failed to run workload {workload.uuid}: {e}")
                    continue

                for sol_name, ev in results.items():
                    trace = Trace(
                        definition=def_name, workload=workload, solution=sol_name, evaluation=ev
                    )

                    result_traces.append(trace)
                    def_traces.append(trace)

                    if ev.status == EvaluationStatus.PASSED:
                        logger.info(
                            f"Solution '{sol_name}' for workload {workload.uuid}: PASSED with "
                            f"{ev.performance.speedup_factor:.2f}x speedup"
                        )
                    else:
                        logger.warning(
                            f"Solution '{sol_name}' for workload {workload.uuid}: {ev.status.value}"
                        )

            if dump_traces and def_traces:
                self._trace_set.add_traces(def_traces)
                logger.info(f"Saved {len(def_traces)} traces for definition {def_name}")

        traces_by_def = defaultdict(list)
        for trace in result_traces:
            traces_by_def[trace.definition].append(trace)

        if self._config.solutions is not None:
            provided_sols = set(self._config.solutions)
            existing_sols = set()
            for sols_list in self._trace_set.solutions.values():
                existing_sols.update(s.name for s in sols_list)
            missing_sols = provided_sols - existing_sols
            if missing_sols:
                logger.warning(f"Solutions not found in trace set: {sorted(missing_sols)}")

        # Create a new TraceSet with the results
        result_trace_set = TraceSet(
            root=self._trace_set.root,
            definitions=self._trace_set.definitions.copy(),
            solutions=self._trace_set.solutions.copy(),
            workloads=self._trace_set.workloads.copy(),
            traces=dict(traces_by_def),
        )

        return result_trace_set

    def close(self) -> None:
        """Release all resources held by the benchmark runner."""
        self._runner.close()
