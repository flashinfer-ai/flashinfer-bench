from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner import BaselineHandle, Runner
from flashinfer_bench.bench.runners.mp_runner import MultiProcessRunner
from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.registry import get_registry
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.json_codec import append_jsonl_lines
from flashinfer_bench.data.solution import Solution
from flashinfer_bench.data.trace import (
    Evaluation,
    EvaluationStatus,
    Trace,
    Workload,
)
from flashinfer_bench.data.traceset import TraceSet
from flashinfer_bench.utils import env_snapshot, list_cuda_devices, torch_dtype_from_def


class Benchmark:
    def __init__(self, trace_set: TraceSet, log_level: str = "INFO") -> None:
        self.trace_set = trace_set
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                        datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._staging_traces: List[Trace] = []
        self._did_archive = False

        self._runners = [MultiProcessRunner(d) for d in list_cuda_devices()]
        self._curr_runner_idx = 0
        self._registry = get_registry()

        if len(self._runners) == 0:
            raise RuntimeError("No CUDA devices available")
        
        self.logger.info(f"Initialized benchmark with {len(self._runners)} CUDA devices")

    def _pick_runners(self, K: int) -> list[Runner]:
        # K = min(len(self.runners), len(solutions))
        if K <= 0 or not self._runners:
            return []
        D = len(self._runners)
        start = self._curr_runner_idx
        sel = [self._runners[(start + i) % D] for i in range(min(K, D))]
        self._curr_runner_idx = (start + K) % D
        return sel

    def run(self, config: BenchmarkConfig = BenchmarkConfig()) -> None:
        for def_name, defn in self.trace_set.definitions.items():
            sols = self.trace_set.solutions.get(def_name, [])
            if not sols:
                self.logger.warning(f"No solutions found for def={def_name}, skipping definition")
                continue

            self.logger.info(f"Processing definition: {def_name} with {len(sols)} solutions")
            
            workloads = self.trace_set.workload.get(def_name, [])
            
            for wl_trace in workloads:
                wl = wl_trace.workload

                K = min(len(self._runners), len(sols))
                selected = self._pick_runners(K)
                if not selected:
                    raise RuntimeError("No healthy runners available")

                # Build baselines on each runner
                baselines: dict[Runner, BaselineHandle] = {}
                failed_runners: list[Runner] = []
                with ThreadPoolExecutor(max_workers=K) as pool:
                    baseline_futs = {
                        pool.submit(
                            r.run_ref,
                            defn,
                            wl,
                            config,
                            self.trace_set.root,
                        ): r
                        for r in selected
                    }
                    for fut, r in baseline_futs.items():
                        try:
                            h = fut.result()
                        except Exception as e:
                            failed_runners.append(r)
                            self.logger.error(
                                f"Runner {r.device} failed while running reference for def={def_name} wl={wl.uuid}: {e}, skipping workload"
                            )
                            continue
                        baselines[r] = h

                # If a runner fails to run reference, we should consider it dead
                if failed_runners:
                    self._runners = [r for r in self._runners if r not in set(failed_runners)]
                    if not self._runners:
                        raise RuntimeError("No healthy runners available")
                    self._curr_runner_idx %= len(self._runners)

                selected = [r for r in selected if r in baselines]
                if not selected:
                    raise RuntimeError("No healthy runners available")

                # Evaluate solutions round-robin across runners
                with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                    sol_futs: Dict[str, any] = {}
                    for i, sol in enumerate(sols):
                        r = selected[i % len(selected)]
                        sol_futs[sol.name] = pool.submit(r.run_solution, sol, baselines[r], config)

                    results: Dict[str, Evaluation] = {
                        name: fut.result() for name, fut in sol_futs.items()
                    }

                for sol_name, ev in results.items():
                    self._staging_traces.append(Trace(def_name, wl, sol_name, ev))
                    
                    if ev.status == EvaluationStatus.PASSED:
                        self.logger.info(f"Solution '{sol_name}' for workload {wl.uuid}: PASSED with {ev.performance.speedup_factor:.2f}x speedup")
                    else:
                        self.logger.warning(f"Solution '{sol_name}' for workload {wl.uuid}: {ev.status.value}")

                for r in selected:
                    r.release(baselines[r])

    def evaluate(self, config: BenchmarkConfig = BenchmarkConfig()) -> TraceSet:
        """
        Evaluate solutions and return a TraceSet with results immediately. 
        Used for small TraceSets that need immediate feedback.
        """
        collected_traces: List[Trace] = []
        
        for def_name, defn in self.trace_set.definitions.items():
            sols = self.trace_set.solutions.get(def_name, [])
            if not sols:
                self.logger.warning(f"No solutions found for def={def_name}, skipping definition")
                continue

            self.logger.info(f"Processing definition: {def_name} with {len(sols)} solutions")
            
            workloads = self.trace_set.workload.get(def_name, [])
            
            for wl_trace in workloads:
                wl = wl_trace.workload

                K = min(len(self._runners), len(sols))
                selected = self._pick_runners(K)
                if not selected:
                    raise RuntimeError("No healthy runners available")

                # Build baselines on each runner
                baselines: dict[Runner, BaselineHandle] = {}
                failed_runners: list[Runner] = []
                with ThreadPoolExecutor(max_workers=K) as pool:
                    baseline_futs = {
                        pool.submit(
                            r.run_ref,
                            defn,
                            wl,
                            config,
                            self.trace_set.root,
                        ): r
                        for r in selected
                    }
                    for fut, r in baseline_futs.items():
                        try:
                            h = fut.result()
                        except Exception as e:
                            failed_runners.append(r)
                            self.logger.error(
                                f"Runner {r.device} failed while running reference for def={def_name} wl={wl.uuid}: {e}, skipping workload"
                            )
                            continue
                        baselines[r] = h

                # If a runner fails to run reference, we should consider it dead
                if failed_runners:
                    self._runners = [r for r in self._runners if r not in set(failed_runners)]
                    if not self._runners:
                        raise RuntimeError("No healthy runners available")
                    self._curr_runner_idx %= len(self._runners)

                selected = [r for r in selected if r in baselines]
                if not selected:
                    raise RuntimeError("No healthy runners available")

                # Evaluate solutions round-robin across runners
                with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                    sol_futs: Dict[str, any] = {}
                    for i, sol in enumerate(sols):
                        r = selected[i % len(selected)]
                        sol_futs[sol.name] = pool.submit(r.run_solution, sol, baselines[r], config)

                    results: Dict[str, Evaluation] = {
                        name: fut.result() for name, fut in sol_futs.items()
                    }

                for sol_name, ev in results.items():
                    collected_traces.append(Trace(def_name, wl, sol_name, ev))
                    
                    if ev.status == EvaluationStatus.PASSED:
                        self.logger.info(f"Solution '{sol_name}' for workload {wl.uuid}: PASSED with {ev.performance.speedup_factor:.2f}x speedup")
                    else:
                        self.logger.warning(f"Solution '{sol_name}' for workload {wl.uuid}: {ev.status.value}")
                        
                for r in selected:
                    r.release(baselines[r])
        
        traces_by_def = defaultdict(list)
        for trace in collected_traces:
            traces_by_def[trace.definition].append(trace)
        
        # Create a new TraceSet with the results
        result_traceset = TraceSet(
            root=self.trace_set.root,
            definitions=self.trace_set.definitions.copy(),
            solutions=self.trace_set.solutions.copy(),
            workload=self.trace_set.workload.copy(),
            traces=dict(traces_by_def)
        )
        
        return result_traceset

    def _ensure_archive(self) -> None:
        if self._did_archive:
            return

        traces_dir = self.trace_set.root / "traces"
        backup = self.trace_set.root / f"traces_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(traces_dir), str(backup))
        traces_dir.mkdir(parents=True, exist_ok=True)

        wk = backup / "workloads"
        if wk.exists():
            (traces_dir / "workloads").parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wk), str(traces_dir / "workloads"))
        self._did_archive = True

    def flush(self) -> None:
        if not self._staging_traces:
            return
        self._ensure_archive()

        buckets = defaultdict(list)
        for tr in self._staging_traces:
            defn = self.trace_set.definitions[tr.definition]
            path = self.trace_set.root / "traces" / defn.type / f"{defn.name}.jsonl"
            buckets[path].append(tr)

        self._staging_traces.clear()

        for path, traces in buckets.items():
            append_jsonl_lines(path, traces)
