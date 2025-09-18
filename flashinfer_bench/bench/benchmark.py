from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

from flashinfer_bench.compile import get_registry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet

from .config import BenchmarkConfig
from .runner import MultiProcessRunner


class Benchmark:
    def __init__(
        self,
        trace_set: TraceSet,
        config: BenchmarkConfig = BenchmarkConfig(),
        log_level: str = "INFO",
    ) -> None:
        # Dataset and configuration
        self._trace_set = trace_set
        self._config = config

        # Setup logger
        self._logger = self._setup_logger(log_level)

        # Setup runner
        self._runner = MultiProcessRunner(self._logger, config.log_dir)

        # Setup registry
        self._registry = get_registry()

        # Setup traces to dump to database
        self._traces_to_dump: List[Trace] = []

        # The traces will be backed up before flush(). Checks if the traces have been backed up.
        self._is_traces_backed_up = False

    def _setup_logger(self, log_level: str) -> logging.Logger:
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {log_level}")

        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}@{id(self):x}")
        logger.setLevel(getattr(logging, log_level.upper()))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_trace_set(self) -> TraceSet:
        return self._trace_set

    def run_all(self, dump_traces: bool = True) -> TraceSet:
        """
        Run benchmark and process solutions for all definitions and workloads.

        Args:
            dump_traces: If True, dump traces to disk.

        Returns:
            List of traces.
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

        if dump_traces:
            self._traces_to_dump.extend(result_traces)

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
        return result_traceset

    def flush(self) -> None:
        if not self._is_traces_backed_up:
            self._trace_set.backup_traces()
            self._is_traces_backed_up = True

        self._trace_set.add_traces(self._traces_to_dump)
        self._traces_to_dump.clear()
