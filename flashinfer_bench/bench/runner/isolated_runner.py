from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import multiprocessing as mp

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile import Runnable, get_registry
from flashinfer_bench.data import Definition, Evaluation, EvaluationStatus, Solution, Workload
from flashinfer_bench.logging import get_logger
from flashinfer_bench.utils import (
    env_snapshot,
    list_cuda_devices,
    redirect_stdio_to_file,
    torch_dtype_from_def,
)

from .evaluator import SolutionEvaluator
from .runner import BaselineHandle, DeviceBaseline, Runner, RunnerError, RunnerFatalError
from .runner_utils import (
    compute_frequency_distribution,
    gen_inputs,
    is_sampling_operation,
    load_safetensors,
    make_eval,
    normalize_outputs,
)

LOGGER = get_logger("MPRunner")


class SubprocessWorker:
    """Each instance binds to a CUDA device; the baseline resides in the main process; each Solution starts an independent Worker process for strong isolation."""

    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
        """Per device subprocess worker

        Parameters
        ----------
        device : str
            Device string (e.g. "cuda:0").
        log_dir : str, optional
            Directory for log files, by default "/tmp/flashinfer_bench".
        """
        self._device = device
        self._log_dir = log_dir
        self._baselines: Dict[BaselineHandle, DeviceBaseline] = {}
        self._registry = get_registry()

    def run_ref(
        self,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        traceset_root: Optional[Path] = None,
    ) -> BaselineHandle:
        torch.cuda.set_device(int(self._device.split(":")[1]))
        dev = torch.device(self._device)

        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}
        runnable_ref = self._registry.build_reference(defn)
        st_cpu = (
            load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        is_sampling = is_sampling_operation(defn)
        inputs_all: List[Dict[str, Any]] = []
        ref_out_all: List[Dict[str, torch.Tensor]] = []

        if is_sampling:
            inp = gen_inputs(defn, workload, device=self._device, stensors=st_cpu)
            inputs_all.append(inp)

            freq_dist = compute_frequency_distribution(
                runnable_ref, [inp], self._device, defn, num_trials=50000
            )
            ref_out = {"frequency_distribution": freq_dist}
            ref_out_all.append(ref_out)
        else:
            for _ in range(cfg.num_trials):
                inp = gen_inputs(defn, workload, device=self._device, stensors=st_cpu)
                inputs_all.append(inp)

                with torch.no_grad():
                    out = runnable_ref(**inp)
                torch.cuda.synchronize(device=dev)
                ref_out = normalize_outputs(
                    out,
                    device=dev,
                    output_names=list(defn.outputs.keys()),
                    output_dtypes=output_dtypes,
                )
                ref_out_all.append(ref_out)

        ref_lat_all: List[float] = []
        for inp in inputs_all:
            ms = time_runnable(runnable_ref, inp, cfg.warmup_runs, cfg.iterations, self._device)
            ref_lat_all.append(ms)

        ref_mean_latency_ms = sum(ref_lat_all) / float(len(ref_lat_all))

        handle = BaselineHandle(uuid.uuid4().hex)

        self._baselines[handle] = DeviceBaseline(
            handle=handle,
            defn=defn,
            device=self._device,
            inputs_dev=inputs_all,
            ref_outputs_dev=ref_out_all,
            ref_mean_latency_ms=ref_mean_latency_ms,
        )
        return handle

    def run_solution(
        self, sol: Solution, baseline: BaselineHandle, cfg: BenchmarkConfig
    ) -> Evaluation:
        """Run solution in an isolated subprocess.

        Parameters
        ----------
        sol : Solution
            Solution to evaluate.
        baseline : BaselineHandle
            Handle to baseline for comparison.
        cfg : BenchmarkConfig
            Benchmark configuration.

        Returns
        -------
        Evaluation
            Evaluation results with status, correctness, and performance metrics.
        """
        if baseline not in self._baselines:
            raise RunnerError(f"Baseline handle not found: {baseline}")
        bl = self._baselines[baseline]

        log_path = os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log")
        # New process for each solution run
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_solution_worker_main,
            args=(child_conn, self._device, bl.defn, sol, cfg, log_path),
            daemon=True,
        )
        proc.start()

        evaluation: Optional[Evaluation] = None
        try:
            msg = parent_conn.recv()
            if msg.get("cmd") != "READY":
                raise RunnerFatalError(f"Worker failed to start, got: {msg}")
            parent_conn.send({"ok": True})

            while True:
                msg = parent_conn.recv()
                cmd = msg.get("cmd")

                if cmd == "LOAN":
                    # Zero-effect copy via IPC handle
                    parent_conn.send(
                        {
                            "ok": True,
                            "inputs": bl.inputs_dev,
                            "ref_outputs": bl.ref_outputs_dev,
                            "ref_mean_latency_ms": bl.ref_mean_latency_ms,
                        }
                    )

                elif cmd == "EVAL":
                    evaluation = msg["evaluation"]
                    break

                elif cmd == "ERROR":
                    error_msg = msg.get("msg", "Unknown error")
                    evaluation = make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_path=log_path,
                        extra_msg=error_msg,
                    )
                    break

                else:
                    LOGGER.warning("Unknown worker command: %s", cmd)
                    continue

        except EOFError as e:
            LOGGER.error("Worker crashed (EOF) running %s: %s", sol.name, e)
        except Exception:
            LOGGER.error("Unknown error running %s", sol.name, exc_info=True)
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
            try:
                proc.join(timeout=2)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

        if evaluation is None:
            evaluation = make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_path=log_path,
                extra_msg="Worker process failed unexpectedly",
            )

        return evaluation

    def release(self, baseline: BaselineHandle) -> None:
        self._baselines.pop(baseline, None)

    def close(self) -> None:
        self._baselines.clear()


def _solution_worker_main(
    conn: mp.connection.Connection,
    device: str,
    defn: Definition,
    sol: Solution,
    cfg: BenchmarkConfig,
    log_path: str,
) -> None:
    """Worker process: strong isolation for single Solution.

    Borrow/return trial data via Pipe and send Evaluation back to parent process.

    Parameters
    ----------
    conn : mp.connection.Connection
        Multiprocessing connection for communication with parent process.
    device : str
        Device string (e.g. "cuda:0").
    defn : Definition
        Operation definition.
    sol : Solution
        Solution to evaluate.
    cfg : BenchmarkConfig
        Benchmark configuration.
    log_path : str
        Path to log file.
    """
    redirect_stdio_to_file(log_path)
    PROCESS_LOGGER = get_logger(f"IsolatedWorker_{device}")
    try:
        PROCESS_LOGGER.info(f"Running solution {sol.name} of definition {defn.name}")

        torch.cuda.set_device(int(device.split(":")[1]))
        registry = get_registry()

        # Handshake
        conn.send({"cmd": "READY"})
        init = conn.recv()
        if not init.get("ok", False):
            conn.send({"cmd": "ERROR", "msg": "Init not ok"})
            return

        # Build impl
        try:
            runnable_sol: Runnable = registry.build(defn, sol)
        except Exception as e:
            import traceback
            PROCESS_LOGGER.error(f"Build error: {type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
            ev = make_eval(
                status=EvaluationStatus.COMPILE_ERROR,
                device=device,
                log_path=log_path,
            )
            conn.send({"cmd": "EVAL", "evaluation": ev})
            return
        PROCESS_LOGGER.info(f"Successfully built solution runnable")


        conn.send({"cmd": "LOAN"})
        loan = conn.recv()

        inputs_bl = loan["inputs"]
        ref_outputs_bl = loan["ref_outputs"]
        ref_mean_latency_ms = loan["ref_mean_latency_ms"]

        inputs: List[Dict[str, Any]] = [
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            for inp in inputs_bl
        ]

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable_sol,
            inputs=inputs,
            ref_outputs=ref_outputs_bl,
            ref_mean_latency_ms=ref_mean_latency_ms,
            cfg=cfg,
            device=device,
            log_path=log_path,
            defn=defn,
        )

        PROCESS_LOGGER.info(f"Evaluation completed. Status: {evaluation.status}")
        conn.send({"cmd": "EVAL", "evaluation": evaluation})

    except Exception as e:
        try:
            conn.send({"cmd": "ERROR", "msg": str(e)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class IsolatedRunner(Runner):
    def __init__(self, logger: logging.Logger, log_dir: str = "/tmp/flashinfer_bench") -> None:
        """Initialize the isolated runner with per device workers.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output.
        log_dir : str, optional
            Directory for log files, by default "/tmp/flashinfer_bench".
        """
        self._logger = logger
        # Track retry attempts for each device
        self._device_retry_counts: Dict[str, int] = {}
        self._worker_max_retries = 3

        # Initialize workers for all available CUDA devices
        self._available_devices = list_cuda_devices()
        self._workers = [SubprocessWorker(d, log_dir) for d in self._available_devices]
        self._curr_worker_idx = 0

        if len(self._workers) == 0:
            raise RuntimeError("No CUDA devices available")

        self._logger.info(
            f"Initialized benchmark multi-process on {len(self._available_devices)} CUDA devices "
            f"and {len(self._workers)} workers"
        )

    def _pick_workers(self, K: int) -> list[SubprocessWorker]:
        """Pick K workers in round-robin fashion.

        Parameters
        ----------
        K : int
            Number of workers to pick.

        Returns
        -------
        list[SubprocessWorker]
            List of selected workers.
        """
        if K <= 0 or not self._workers:
            return []
        D = len(self._workers)
        start = self._curr_worker_idx
        sel = [self._workers[(start + i) % D] for i in range(min(K, D))]
        self._curr_worker_idx = (start + K) % D
        return sel

    def _relaunch_worker(self, device: str) -> SubprocessWorker:
        """Relaunch a worker for the given device.

        Parameters
        ----------
        device : str
            Device string (e.g. "cuda:0").

        Returns
        -------
        SubprocessWorker
            New worker instance for the device.
        """
        self._logger.info(f"Relaunching worker for device {device}")
        return SubprocessWorker(device, self._log_dir)

    def _handle_failed_workers(self, failed_workers: List[SubprocessWorker]) -> None:
        """Handle failed workers by attempting to relaunch them or removing them.

        Parameters
        ----------
        failed_workers : List[SubprocessWorker]
            List of workers that have failed.
        """
        workers_to_remove = []
        workers_to_add = []

        for failed_worker in failed_workers:
            device = failed_worker._device
            retry_count = self._device_retry_counts.get(device, 0)

            if retry_count < self._worker_max_retries:
                self._device_retry_counts[device] = retry_count + 1
                try:
                    new_worker = self._relaunch_worker(device)
                    workers_to_add.append(new_worker)
                    self._logger.info(f"Successfully relaunched worker for device {device} ")
                except Exception:
                    self._logger.error(f"Failed to relaunch worker for device {device} ")
                    if retry_count + 1 >= self._worker_max_retries:
                        workers_to_remove.append(failed_worker)
                        self._logger.warning(
                            f"Removing device {device} after {self._worker_max_retries} failed attempts"
                        )
            else:
                workers_to_remove.append(failed_worker)
                self._logger.warning(
                    f"Removing device {device} after {self._worker_max_retries} failed attempts"
                )
        if workers_to_remove:
            self._workers = [r for r in self._workers if r not in workers_to_remove]

        self._workers.extend(workers_to_add)

        if self._workers:
            self._curr_worker_idx %= len(self._workers)

    def _has_healthy_workers(self) -> bool:
        """Check if there are any healthy workers available.

        Returns
        -------
        bool
            True if there are healthy workers, False otherwise.
        """
        return bool(self._workers)

    def run_workload(
        self,
        defn: Definition,
        wl: Workload,
        solutions: List[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> Dict[str, Evaluation]:
        """Run a workload with the given solutions and return evaluation results.

        Parameters
        ----------
        defn : Definition
            Operation definition.
        wl : Workload
            Workload specification.
        solutions : List[Solution]
            List of solutions to evaluate.
        config : BenchmarkConfig
            Benchmark configuration.
        root : Path
            Root path for the trace set.

        Returns
        -------
        Dict[str, Evaluation]
            Dictionary mapping solution names to their evaluations.
        """
        if not solutions:
            return {}

        K = min(len(self._workers), len(solutions))
        selected = self._pick_workers(K)
        if not selected:
            raise RuntimeError("No healthy workers available")

        # Build baselines on each worker
        baselines: dict[SubprocessWorker, BaselineHandle] = {}
        failed_workers: list[SubprocessWorker] = []

        with ThreadPoolExecutor(max_workers=K) as pool:
            baseline_futs = {pool.submit(r.run_ref, defn, wl, config, root): r for r in selected}
            for fut, r in baseline_futs.items():
                try:
                    h = fut.result()
                    baselines[r] = h
                except Exception as e:
                    failed_workers.append(r)
                    self._logger.error(
                        f"Runner {r._device} failed while running reference for "
                        f"def={defn.name} wl={wl.uuid}: {e}"
                    )

        # Handle failed workers
        if failed_workers:
            self._handle_failed_workers(failed_workers)
            if not self._has_healthy_workers():
                raise RuntimeError("No healthy workers available")

        # Filter out workers that failed to build baselines
        selected = [r for r in selected if r in baselines]
        if not selected:
            raise RuntimeError("No healthy workers available after baseline setup")

        try:
            # Evaluate solutions round-robin across workers
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                sol_futs: Dict[str, any] = {}
                for i, sol in enumerate(solutions):
                    r = selected[i % len(selected)]
                    sol_futs[sol.name] = pool.submit(r.run_solution, sol, baselines[r], config)

                results: Dict[str, Evaluation] = {
                    name: fut.result() for name, fut in sol_futs.items()
                }
        finally:
            # Always release baselines, even if solution execution fails
            for r in selected:
                if r in baselines:
                    r.release(baselines[r])

        return results
