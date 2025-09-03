from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner import BaselineHandle, Runner
from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.registry import get_registry
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.json_codec import save_jsonl_file
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
    def __init__(self, trace_set: TraceSet) -> None:
        self.trace_set = trace_set

        self._staging_traces: List[Trace] = []
        self._did_archive = False

        self._runners = [Runner(d) for d in list_cuda_devices()]
        self._curr_device_idx = 0
        self._registry = get_registry()

        if len(self._runners) == 0:
            raise RuntimeError("No CUDA devices available")

    def _pick_runners(self, K: int) -> list[Runner]:
        # K = min(len(self.runners), len(solutions))
        if K <= 0:
            return []
        D = len(self._runners)
        start = self._curr_device_idx
        sel = [self._runners[(start + i) % D] for i in range(K)]
        self._curr_device_idx = (start + K) % D
        return sel

    def _prefetch_safetensors(self, defn: Definition, wl: Workload) -> Dict[str, torch.Tensor]:
        try:
            import safetensors.torch as st
        except Exception:
            raise RuntimeError("safetensors is not available in the current environment")

        expected = defn.get_input_shapes(wl.axes)
        host_tensors: Dict[str, torch.Tensor] = {}
        for name, desc in wl.inputs.items():
            if desc.type != "safetensors":
                continue

            tensors = st.load_file(desc.path)
            if desc.tensor_key not in tensors:
                raise ValueError(f"Missing key '{desc.tensor_key}' in '{desc.path}'")
            t = tensors[desc.tensor_key]
            # shape check
            if list(t.shape) != expected[name]:
                raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
            # dtype check
            expect_dtype = torch_dtype_from_def(defn.inputs[name].dtype)
            if t.dtype != expect_dtype:
                raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

            try:
                t = t.contiguous().pin_memory()
            except Exception:
                t = t.contiguous()
            host_tensors[name] = t
        return host_tensors

    def run(self, config: BenchmarkConfig = BenchmarkConfig()) -> None:
        for def_name, defn in self.trace_set.definitions.items():
            sols = self.trace_set.solutions.get(def_name, [])
            if not sols:
                print(f"No solutions found for definition: {def_name}, skipping")
                continue

            runnable_ref = self._registry.build_reference(defn)

            for wl_trace in self.trace_set.workload.get(def_name, []):
                wl = wl_trace.workload

                # Prefetch safetensors and validate structure
                try:
                    host_tensors = self._prefetch_safetensors(defn, wl)
                except Exception as e:
                    print(
                        f"Error loading safetensors for definition: {def_name} / workload: {wl.uuid}: {e}, skipping"
                    )
                    continue

                K = min(len(self._runners), len(sols))
                selected_runners = self._pick_runners(K)

                # Build baselines on each runner
                with ThreadPoolExecutor(max_workers=K) as pool:
                    baseline_futs = {
                        pool.submit(
                            r.run_reference, defn, wl, config, runnable_ref, host_tensors
                        ): r
                        for r in selected_runners
                    }
                    baselines: dict[Runner, BaselineHandle] = {}
                    failed_runners: list[Runner] = []
                    for fut, r in baseline_futs.items():
                        try:
                            h = fut.result()
                        except Exception:
                            failed_runners.append(r)
                            print(
                                f"Error running reference for definition: {def_name} / workload: {wl.uuid}: {e}"
                            )
                            continue
                        baselines[r] = h

                # TODO(shanli): runner recovery, better failure handling
                if failed_runners:
                    # remove failed runners
                    self._runners = [r for r in self._runners if r not in set(failed_runners)]
                    if self._runners:
                        self._curr_device_idx %= len(self._runners)
                    else:
                        raise RuntimeError("No healthy runners available")

                # Evaluate solutions round-robin across runners
                with ThreadPoolExecutor(max_workers=K) as pool:
                    sol_futs = []
                    for i, sol in enumerate(sols):
                        r = selected_runners[i % K]
                        sol_futs.append(
                            pool.submit(self._submit_one_sol, defn, sol, r, baselines[r], config)
                        )
                    results = [f.result() for f in sol_futs]

                for sol_name, ev in results:
                    self._staging_traces.append(Trace(def_name, wl, sol_name, ev))

                for r in selected_runners:
                    r.release(baselines[r])

            try:
                runnable_ref.close()
            except Exception:
                pass

    def _submit_one_sol(
        self,
        defn: Definition,
        sol: Solution,
        runner: Runner,
        baseline_handle: BaselineHandle,
        cfg: BenchmarkConfig,
    ) -> Tuple[str, Evaluation]:
        try:
            runnable_sol = self._registry.build(defn, sol)
        except BuildError:
            ev = Evaluation(
                status=EvaluationStatus.COMPILE_ERROR,
                log_file="build.log",  # TODO(shanli): replace with a real log file
                environment=env_snapshot(runner.device),
                timestamp=datetime.now().isoformat(),
            )
            return sol.name, ev
        try:
            ev = runner.run_solution(runnable_sol, baseline_handle, cfg)
        finally:
            try:
                runnable_sol.close()
            except:
                pass
        return sol.name, ev

    def _ensure_archive(self) -> None:
        if self._did_archive:
            return
        backup = self.trace_set.root / f"traces_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup.mkdir(parents=True, exist_ok=True)
        for item in self.trace_set.root.iterdir():
            if item.name == "workloads":
                continue
            shutil.move(str(item), str(backup / item.name))
        self._did_archive = True

    def flush(self) -> None:
        if not self._staging_traces:
            return
        self._ensure_archive()

        while self._staging_traces:
            trace = self._staging_traces.pop(0)
            defn = self.trace_set.definitions.get(trace.definition)

            trace_file = self.trace_set.root / defn.type / f"{defn.name}.jsonl"

            save_jsonl_file([trace], trace_file)
