"""Isolated runner that spawns a new process for each solution."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import multiprocessing as mp

import flashinfer_bench.utils as fib_utils
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import BuilderRegistry, Runnable
from flashinfer_bench.data import Definition, Evaluation, EvaluationStatus, Solution, Workload
from flashinfer_bench.logging import get_logger
from flashinfer_bench.utils import redirect_stdio_to_file

from .runner import BaselineHandle, DeviceBaseline, Runner, RunnerError, RunnerFatalError

logger = logging.getLogger(__name__)


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

    def run_ref(
        self,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        trace_set_root: Optional[Path] = None,
    ) -> BaselineHandle:
        evaluator_cls = resolve_evaluator(definition)
        baseline = evaluator_cls.build_baseline(
            definition=definition,
            workload=workload,
            cfg=cfg,
            device=self._device,
            trace_set_root=trace_set_root,
        )
        self._baselines[baseline.handle] = baseline
        return baseline.handle

    def run_solution(
        self, solution: Solution, baseline: BaselineHandle, cfg: BenchmarkConfig
    ) -> Evaluation:
        """Run solution in an isolated subprocess.

        Parameters
        ----------
        solution : Solution
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

        log_path = os.path.join(self._log_dir, f"{solution.name}_{time.time()}.log")
        # New process for each solution run
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_solution_worker_main,
            args=(child_conn, self._device, bl.definition, solution, cfg, log_path),
            daemon=True,
        )
        proc.start()

        evaluation: Optional[Evaluation] = None
        start_time = time.time()
        try:
            if parent_conn.poll(timeout=30.0):  # 30 seconds for startup
                msg = parent_conn.recv()
                if msg.get("cmd") != "READY":
                    raise RunnerFatalError(f"Worker failed to start, got: {msg}")
                parent_conn.send({"ok": True})
            else:
                evaluation = make_eval(
                    status=EvaluationStatus.TIMEOUT,
                    device=self._device,
                    log_path=log_path,
                    extra_msg="Worker failed to start within 30 seconds",
                )
                return evaluation

            while True:
                # Check if we've exceeded total timeout
                elapsed = time.time() - start_time
                remaining_timeout = max(1.0, cfg.timeout_seconds - elapsed)

                if elapsed >= cfg.timeout_seconds:
                    evaluation = make_eval(
                        status=EvaluationStatus.TIMEOUT,
                        device=self._device,
                        log_path=log_path,
                        extra_msg=f"Evaluation timeout after {cfg.timeout_seconds} seconds for solution {solution.name}",
                    )
                    break

                # Wait for message with remaining timeout
                if parent_conn.poll(timeout=remaining_timeout):
                    msg = parent_conn.recv()
                    cmd = msg.get("cmd")
                else:
                    # Timeout
                    evaluation = make_eval(
                        status=EvaluationStatus.TIMEOUT,
                        device=self._device,
                        log_path=log_path,
                        extra_msg=f"Evaluation timeout after {cfg.timeout_seconds} seconds for solution {solution.name}",
                    )
                    break

                if cmd == "LOAN":
                    # Zero-effect copy via IPC handle
                    parent_conn.send(
                        {
                            "ok": True,
                            "inputs": bl.inputs,
                            "ref_outputs": bl.outputs,
                            "ref_mean_latency_ms": bl.mean_latency_ms,
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
                    logger.warning("Unknown worker command: %s", cmd)
                    continue

        except EOFError as e:
            logger.error("Worker crashed (EOF) running %s: %s", solution.name, e)
        except Exception:
            logger.error("Unknown error running %s", solution.name, exc_info=True)
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
    definition: Definition,
    solution: Solution,
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
    definition : Definition
        Operation definition.
    solution : Solution
        Solution to evaluate.
    cfg : BenchmarkConfig
        Benchmark configuration.
    log_path : str
        Path to log file.
    """
    original_stdout_fd, original_stderr_fd = redirect_stdio_to_file(log_path)
    try:
        torch.cuda.set_device(int(device.split(":")[1]))
        registry = BuilderRegistry.get_instance()

        # Handshake
        conn.send({"cmd": "READY"})
        init = conn.recv()
        if not init.get("ok", False):
            conn.send({"cmd": "ERROR", "msg": "Init not ok"})
            return

        # Build impl
        try:
            runnable_sol: Runnable = registry.build(definition, solution)
        except Exception as e:
            import traceback

            print(
                f"Build error: {type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            ev = make_eval(status=EvaluationStatus.COMPILE_ERROR, device=device, log_path=log_path)
            conn.send({"cmd": "EVAL", "evaluation": ev})
            return

        conn.send({"cmd": "LOAN"})
        loan = conn.recv()

        inputs_bl = loan["inputs"]
        ref_outputs_bl = loan["ref_outputs"]
        ref_mean_latency_ms = loan["ref_mean_latency_ms"]

        inputs: List[List[Any]] = [
            [v.clone() if isinstance(v, torch.Tensor) else v for v in inp] for inp in inputs_bl
        ]

        evaluator_cls = resolve_evaluator(definition)
        evaluation = evaluator_cls.evaluate(
            definition=definition,
            sol_runnable=runnable_sol,
            inputs=inputs,
            ref_outputs=ref_outputs_bl,
            ref_mean_latency_ms=ref_mean_latency_ms,
            cfg=cfg,
            log_path=log_path,
            device=device,
        )

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
    def __init__(self, log_dir: str) -> None:
        """Initialize the isolated runner with per device workers.

        Parameters
        ----------
        log_dir : str, optional
            Directory for log files, by default "/tmp/flashinfer_bench".
        """
        # Track retry attempts for each device
        self._device_retry_counts: Dict[str, int] = {}
        self._worker_max_retries = 3

        # Initialize workers for all available CUDA devices
        self._available_devices = fib_utils.list_cuda_devices()
        self._workers = [SubprocessWorker(d, log_dir) for d in self._available_devices]
        self._curr_worker_idx = 0
        self._log_dir = log_dir

        if len(self._workers) == 0:
            raise RuntimeError("No CUDA devices available")

        logger.info(
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
        logger.info(f"Relaunching worker for device {device}")
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
                    logger.info(f"Successfully relaunched worker for device {device} ")
                except Exception:
                    logger.error(f"Failed to relaunch worker for device {device} ")
                    if retry_count + 1 >= self._worker_max_retries:
                        workers_to_remove.append(failed_worker)
                        logger.warning(
                            f"Removing device {device} after {self._worker_max_retries} failed attempts"
                        )
            else:
                workers_to_remove.append(failed_worker)
                logger.warning(
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
        definition: Definition,
        workload: Workload,
        solutions: List[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> Dict[str, Evaluation]:
        """Run a workload with the given solutions and return evaluation results.

        Parameters
        ----------
        definition : Definition
            Operation definition.
        workload : Workload
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
            baseline_futs = {
                pool.submit(r.run_ref, definition, workload, config, root): r for r in selected
            }
            for fut, r in baseline_futs.items():
                try:
                    h = fut.result()
                    baselines[r] = h
                except Exception as e:
                    failed_workers.append(r)
                    logger.error(
                        f"Runner {r._device} failed while running reference for "
                        f"def={definition.name} workload={workload.uuid}: {e}"
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
                sol_futs: Dict[str, Any] = {}
                for i, solution in enumerate(solutions):
                    r = selected[i % len(selected)]
                    sol_futs[solution.name] = pool.submit(
                        r.run_solution, solution, baselines[r], config
                    )

                results: Dict[str, Evaluation] = {
                    name: fut.result() for name, fut in sol_futs.items()
                }
        finally:
            # Always release baselines, even if solution execution fails
            for r in selected:
                if r in baselines:
                    r.release(baselines[r])

        return results

    def close(self) -> None:
        """Release all resources and terminate worker processes."""
        for worker in self._workers:
            worker.close()
        self._workers.clear()
