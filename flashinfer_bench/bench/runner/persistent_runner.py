from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import multiprocessing as mp

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile import BuildError, Runnable, get_registry
from flashinfer_bench.data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    Solution,
    Workload,
)
from flashinfer_bench.logging import get_logger
from flashinfer_bench.utils import list_cuda_devices, torch_dtype_from_def

from .runner import BaselineHandle, DeviceBaseline, Runner, RunnerError, RunnerFatalError
from .runner_utils import (
    compute_frequency_distribution,
    gen_inputs,
    is_sampling_operation,
    load_safetensors,
    make_eval,
    normalize_outputs,
)
from .evaluator import SolutionEvaluator

LOGGER = get_logger("PersistentRunner")


class WorkerCommand(Enum):
    RUN_SOLUTION = "run_solution"
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    READY = "ready"
    EVALUATION = "evaluation"
    ERROR = "error"
    HEALTHY = "healthy"
    CORRUPTED = "corrupted"


@dataclass
class SolutionFailureRecord:
    """Track failures for a solution."""

    solution_name: str
    failure_count: int
    last_error: str
    last_status: EvaluationStatus
    last_failure_time: float


class PersistentSubprocessWorker:
    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
        """Per device persistent subprocess worker
        
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

        # Solution failure tracking
        self._failure_records: Dict[str, SolutionFailureRecord] = {}
        self._max_failures = 3  # if a solution fails for more than 3 times, it will be skipped

        self._worker_proc: Optional[mp.Process] = None
        self._parent_conn: Optional[mp.connection.Connection] = None

        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker_proc is not None and self._worker_proc.is_alive():
            self._shutdown_worker()

        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe(duplex=True)

        self._worker_proc = ctx.Process(
            target=_persistent_worker_main,
            args=(child_conn, self._device, self._log_dir),
            daemon=True,
        )
        self._worker_proc.start()

        try:
            msg = self._parent_conn.recv()
            if msg.get("cmd") == WorkerResponse.READY.value:
                LOGGER.info(f"Persistent worker started for device {self._device}")
            else:
                raise RunnerFatalError(f"Worker failed to start: {msg}")
        except Exception as e:
            raise RunnerFatalError(f"Failed to start worker: {e}")

    def _shutdown_worker(self) -> None:
        if self._parent_conn is not None:
            try:
                self._parent_conn.send({"cmd": WorkerCommand.SHUTDOWN.value})
                self._parent_conn.close()
            except Exception:
                pass
            self._parent_conn = None

        if self._worker_proc is not None:
            try:
                self._worker_proc.join(timeout=5)
            except Exception:
                pass
            if self._worker_proc.is_alive():
                try:
                    self._worker_proc.terminate()
                    self._worker_proc.join(timeout=2)
                except Exception:
                    pass
            self._worker_proc = None

        # Clear GPU memory after worker shutdown
        try:
            torch.cuda.set_device(int(self._device.split(":")[1]))
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=self._device)
        except Exception:
            pass

    def is_healthy(self) -> bool:
        if (
            self._parent_conn is None
            or self._worker_proc is None
            or not self._worker_proc.is_alive()
        ):
            return False

        # Check if connection is closed
        if self._parent_conn.closed:
            LOGGER.warning(f"Connection is closed for device {self._device}")
            return False

        try:
            self._parent_conn.send({"cmd": WorkerCommand.HEALTH_CHECK.value})

            if self._parent_conn.poll(timeout=5.0):
                try:
                    msg = self._parent_conn.recv()

                    if msg.get("cmd") == WorkerResponse.HEALTHY.value:
                        return True
                    elif msg.get("cmd") == WorkerResponse.CORRUPTED.value:
                        LOGGER.warning(f"GPU context corrupted on device {self._device}")
                        return False
                    else:
                        LOGGER.warning(
                            f"Unexpected health check response on device {self._device}: {msg}"
                        )
                        return False

                except (EOFError, ConnectionResetError, OSError) as e:
                    LOGGER.warning(
                        f"Connection error during health check on device {self._device}: {e}"
                    )
                    return False
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "ran out of input" in error_str
                        or "pickle" in error_str
                        or "unpickling" in error_str
                    ):
                        LOGGER.warning(
                            f"Connection closed or corrupted during health check on device {self._device}: {e}"
                        )
                    else:
                        LOGGER.warning(
                            f"Failed to decode health check response on device {self._device}: {e}"
                        )
                    return False
            else:
                LOGGER.warning(f"Health check timeout on device {self._device}")
                return False

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            LOGGER.warning(f"Connection broken during health check on device {self._device}: {e}")
            return False
        except Exception as e:
            LOGGER.warning(f"Health check failed on device {self._device}: {e}")
            return False

    def restart(self) -> bool:
        """Restart the worker process.
        
        Returns
        -------
        bool
            True if restart was successful, False otherwise.
        """
        try:
            LOGGER.info(f"Restarting worker for device {self._device}")

            self._baselines.clear()
            self._shutdown_worker()
            self._start_worker()

            LOGGER.info(f"Successfully restarted worker for device {self._device}")
            return True

        except Exception as e:
            LOGGER.error(f"Failed to restart worker for device {self._device}: {e}")
            return False

    def _should_skip_solution(self, solution_name: str) -> Optional[SolutionFailureRecord]:
        if solution_name in self._failure_records:
            record = self._failure_records[solution_name]
            if record.failure_count >= self._max_failures:
                return record
        return None

    def _record_failure(self, solution_name: str, error: str, status: EvaluationStatus) -> None:
        if solution_name in self._failure_records:
            record = self._failure_records[solution_name]
            if status == EvaluationStatus.COMPILE_ERROR:
                record.failure_count = self._max_failures
            else:
                record.failure_count += 1
            record.last_error = error
            record.last_status = status
            record.last_failure_time = time.time()
        else:
            failure_count = self._max_failures if status == EvaluationStatus.COMPILE_ERROR else 1
            self._failure_records[solution_name] = SolutionFailureRecord(
                solution_name=solution_name,
                failure_count=failure_count,
                last_error=error,
                last_status=status,
                last_failure_time=time.time(),
            )

    def _clear_failure_record(self, solution_name: str) -> None:
        self._failure_records.pop(solution_name, None)

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
        """Run solution using cached compilation."""
        if baseline not in self._baselines:
            raise RunnerError(f"Baseline handle not found: {baseline}")
        bl = self._baselines[baseline]

        solution_name = sol.name
        failure_record = self._should_skip_solution(solution_name)
        if failure_record is not None:
            LOGGER.info(
                f"Skipping solution {sol.name} due to {failure_record.failure_count} consecutive failures"
            )
            return make_eval(
                status=failure_record.last_status,
                device=self._device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=f"Solution skipped after {failure_record.failure_count} failures. Last error: {failure_record.last_error}",
            )

        eval_msg = {
            "cmd": WorkerCommand.RUN_SOLUTION.value,
            "definition": bl.defn,
            "solution": sol,
            "inputs": bl.inputs_dev,
            "ref_outputs": bl.ref_outputs_dev,
            "ref_mean_latency_ms": bl.ref_mean_latency_ms,
            "config": cfg,
            "solution_name": sol.name,
        }

        if self._parent_conn is None or self._parent_conn.closed:
            error_msg = "Connection is closed or invalid"
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=error_msg,
            )

        try:
            self._parent_conn.send(eval_msg)

            if self._parent_conn.poll(timeout=300.0):
                try:
                    response = self._parent_conn.recv()

                    if response.get("cmd") == WorkerResponse.EVALUATION.value:
                        evaluation = response["evaluation"]
                        if evaluation.status == EvaluationStatus.PASSED:
                            self._clear_failure_record(sol.name)
                        elif evaluation.status in (
                            EvaluationStatus.RUNTIME_ERROR,
                            EvaluationStatus.INCORRECT_SHAPE,
                            EvaluationStatus.INCORRECT_DTYPE,
                            EvaluationStatus.COMPILE_ERROR,
                        ):
                            self._record_failure(
                                sol.name, evaluation.error or "Evaluation failed", evaluation.status
                            )
                        return evaluation
                    elif response.get("cmd") == WorkerResponse.ERROR.value:
                        error_msg = response.get("error", "Unknown evaluation error")
                        self._record_failure(sol.name, error_msg, EvaluationStatus.RUNTIME_ERROR)
                        return make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                            error=error_msg,
                        )
                    else:
                        error_msg = f"Unexpected evaluation response: {response}"
                        self._record_failure(sol.name, error_msg, EvaluationStatus.RUNTIME_ERROR)
                        return make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                            error=error_msg,
                        )

                except (EOFError, ConnectionResetError, OSError) as e:
                    error_msg = f"Connection error during evaluation: {e}"
                    return make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                        error=error_msg,
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "ran out of input" in error_str
                        or "pickle" in error_str
                        or "unpickling" in error_str
                    ):
                        error_msg = f"Connection closed or corrupted during evaluation: {e}"
                    else:
                        error_msg = f"Failed to decode evaluation response: {e}"
                    return make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                        error=error_msg,
                    )
            else:
                error_msg = f"Evaluation timeout after 300 seconds for solution {sol.name}"
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=self._device,
                    log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                    error=error_msg,
                )

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            error_msg = f"Connection broken during evaluation: {e}"
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Failed to communicate with worker: {e}"
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=error_msg,
            )

    def release(self, baseline: BaselineHandle) -> None:
        self._baselines.pop(baseline, None)

    def close(self) -> None:
        self._shutdown_worker()
        self._baselines.clear()
        self._failure_records.clear()


class PersistentRunner(Runner):
    def __init__(self, logger: logging.Logger, log_dir: str = "/tmp/flashinfer_bench") -> None:
        """Initialize the persistent runner with multiple workers.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output.
        log_dir : str, optional
            Directory for log files, by default "/tmp/flashinfer_bench".
        """
        self._logger = logger
        self._log_dir = log_dir

        # Track retry attempts for each device
        self._device_retry_counts: Dict[str, int] = {}
        self._worker_max_retries = 3

        self._available_devices = list_cuda_devices()
        self._workers = [PersistentSubprocessWorker(d, log_dir) for d in self._available_devices]
        self._curr_worker_idx = 0

        if len(self._workers) == 0:
            raise RuntimeError("No CUDA devices available")

        self._logger.info(
            f"Initialized benchmark persistent runner on {len(self._available_devices)} CUDA devices "
            f"and {len(self._workers)} workers"
        )

    def _pick_workers(self, K: int) -> list[PersistentSubprocessWorker]:
        """Pick K workers in round-robin fashion."""
        if K <= 0 or not self._workers:
            return []
        D = len(self._workers)
        start = self._curr_worker_idx
        sel = [self._workers[(start + i) % D] for i in range(min(K, D))]
        self._curr_worker_idx = (start + K) % D
        return sel

    def _handle_failed_workers(
        self, failed_workers: List[PersistentSubprocessWorker], increment_retries: bool = True
    ) -> None:
        """Handle failed workers by attempting to restart them or removing them.

        Parameters
        ----------
        failed_workers : List[PersistentSubprocessWorker]
            List of workers that have failed.
        increment_retries : bool, optional
            Whether to increment retry count (True for health failures, False for solution failures), by default True.
        """
        workers_to_remove = []

        for failed_worker in failed_workers:
            device = failed_worker._device
            retry_count = self._device_retry_counts.get(device, 0)

            if retry_count < self._worker_max_retries:
                if increment_retries:
                    self._device_retry_counts[device] = retry_count + 1
                    new_retry_count = retry_count + 1
                else:
                    new_retry_count = retry_count

                if failed_worker.restart():
                    self._logger.info(
                        f"Successfully restarted persistent worker for device {device}"
                    )
                else:
                    self._logger.error(f"Failed to restart persistent worker for device {device}")
                    if new_retry_count >= self._worker_max_retries:
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
            for worker in workers_to_remove:
                try:
                    worker.close()
                except Exception:
                    pass
            self._workers = [r for r in self._workers if r not in workers_to_remove]

        if self._workers:
            self._curr_worker_idx %= len(self._workers)

    def _has_healthy_workers(self) -> bool:
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
            raise RuntimeError("No healthy persistent workers available")

        # Build baselines on each worker
        baselines: dict[PersistentSubprocessWorker, BaselineHandle] = {}
        failed_workers: list[PersistentSubprocessWorker] = []

        with ThreadPoolExecutor(max_workers=K) as pool:
            baseline_futs = {pool.submit(r.run_ref, defn, wl, config, root): r for r in selected}
            for fut, r in baseline_futs.items():
                try:
                    h = fut.result()
                    baselines[r] = h
                except Exception as e:
                    failed_workers.append(r)
                    self._logger.error(
                        f"Persistent worker {r._device} failed while running reference for "
                        f"def={defn.name} wl={wl.uuid}: {e}"
                    )

        if failed_workers:
            self._handle_failed_workers(failed_workers, increment_retries=True)
            if not self._has_healthy_workers():
                raise RuntimeError("No healthy persistent workers available")

        # Filter out workers that failed to build baselines
        selected = [r for r in selected if r in baselines]
        if not selected:
            raise RuntimeError("No healthy persistent workers available after baseline setup")

        def run_solution_with_health_check(
            worker: PersistentSubprocessWorker, solution: Solution, baseline_handle: BaselineHandle
        ) -> Evaluation:
            try:
                if not worker.is_healthy():
                    LOGGER.warning(
                        f"Worker on device {worker._device} is unhealthy, attempting restart"
                    )
                    if worker.restart():
                        try:
                            new_baseline = worker.run_ref(defn, wl, config, root)
                            worker.release(baseline_handle)
                            baseline_handle = new_baseline
                            LOGGER.info(f"Rebuilt baseline for worker on device {worker._device}")
                        except Exception as e:
                            LOGGER.error(
                                f"Failed to rebuild baseline after restart for device {worker._device}: {e}"
                            )
                            return make_eval(
                                status=EvaluationStatus.RUNTIME_ERROR,
                                device=worker._device,
                                log_file=os.path.join(
                                    self._log_dir, f"{solution.name}_{time.time()}.log"
                                ),
                                error=f"Failed to rebuild baseline after restart: {e}",
                            )
                    else:
                        LOGGER.error(f"Failed to restart worker on device {worker._device}")
                        return make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=worker._device,
                            log_file=os.path.join(
                                self._log_dir, f"{solution.name}_{time.time()}.log"
                            ),
                            error="Worker restart failed",
                        )

                # Run the solution
                return worker.run_solution(solution, baseline_handle, config)

            except Exception as e:
                LOGGER.error(f"Unexpected error in solution execution for {solution.name}: {e}")
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=worker._device,
                    log_file=os.path.join(self._log_dir, f"{solution.name}_{time.time()}.log"),
                    error=f"Unexpected error: {e}",
                )

        try:
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                sol_futs: Dict[str, any] = {}

                for i, sol in enumerate(solutions):
                    worker = selected[i % len(selected)]
                    baseline_handle = baselines[worker]

                    sol_futs[sol.name] = pool.submit(
                        run_solution_with_health_check, worker, sol, baseline_handle
                    )

                results: Dict[str, Evaluation] = {
                    name: fut.result() for name, fut in sol_futs.items()
                }
        finally:
            # Clean up baselines
            for r in selected:
                if r in baselines:
                    try:
                        r.release(baselines[r])
                    except Exception as e:
                        LOGGER.warning(f"Failed to release baseline for device {r._device}: {e}")

        return results


def _persistent_worker_main(conn: mp.connection.Connection, device: str, log_dir: str) -> None:
    """Long-lived worker process that handles solution evaluations.
    
    Caches compiled solutions to avoid recompilation (handled in builder registry).
    
    Parameters
    ----------
    conn : mp.connection.Connection
        Multiprocessing connection for communication with parent process.
    device : str
        Device string (e.g. "cuda:0").
    log_dir : str
        Directory for log files.
    """
    try:
        torch.cuda.set_device(int(device.split(":")[1]))
        registry = get_registry()

        conn.send({"cmd": WorkerResponse.READY.value})

        while True:
            try:
                msg = conn.recv()
                cmd = msg.get("cmd")

                if cmd == WorkerCommand.SHUTDOWN.value:
                    break

                elif cmd == WorkerCommand.HEALTH_CHECK.value:
                    try:
                        # GPU health check
                        test_tensor = torch.zeros(1, device=device)
                        test_tensor += 1
                        torch.cuda.synchronize(device=device)
                        del test_tensor
                        conn.send({"cmd": WorkerResponse.HEALTHY.value})
                    except Exception:
                        conn.send({"cmd": WorkerResponse.CORRUPTED.value})
                        break

                elif cmd == WorkerCommand.RUN_SOLUTION.value:
                    defn = msg["definition"]
                    sol = msg["solution"]
                    inputs_bl = msg["inputs"]
                    ref_outputs_bl = msg["ref_outputs"]
                    ref_mean_latency_ms = msg["ref_mean_latency_ms"]
                    cfg = msg["config"]
                    solution_name = msg["solution_name"]

                    log_path = os.path.join(log_dir, f"{solution_name}_{time.time()}.log")

                    try:
                        # Use registry to build/get cached solution
                        runnable_sol = registry.build(defn, sol)

                        evaluation = _evaluate_solution_worker(
                            runnable_sol=runnable_sol,
                            inputs_bl=inputs_bl,
                            ref_outputs_bl=ref_outputs_bl,
                            ref_mean_latency_ms=ref_mean_latency_ms,
                            cfg=cfg,
                            device=device,
                            log_path=log_path,
                            defn=defn,
                        )

                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )

                    except BuildError as e:
                        import traceback

                        error_msg = f"BuildError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

                        evaluation = make_eval(
                            status=EvaluationStatus.COMPILE_ERROR,
                            device=device,
                            log_file=log_path,
                            error=error_msg,
                        )
                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )
                    except Exception as e:
                        import traceback

                        error_msg = (
                            f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        )

                        evaluation = make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=device,
                            log_file=log_path,
                            error=error_msg,
                        )
                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )
                else:
                    conn.send(
                        {"cmd": WorkerResponse.ERROR.value, "error": f"Unknown command: {cmd}"}
                    )

            except EOFError:
                # parent closed connection
                break
            except Exception as e:
                try:
                    conn.send({"cmd": WorkerResponse.ERROR.value, "error": str(e)})
                except Exception:
                    break

    except Exception as e:
        try:
            conn.send({"cmd": WorkerResponse.ERROR.value, "error": f"Worker startup failed: {e}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _evaluate_solution_worker(
    runnable_sol: Runnable,
    inputs_bl: List[Dict[str, Any]],
    ref_outputs_bl: List[Dict[str, torch.Tensor]],
    ref_mean_latency_ms: float,
    cfg: BenchmarkConfig,
    device: str,
    log_path: str,
    defn: Definition,
) -> Evaluation:
    inputs: List[Dict[str, Any]] = [
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
        for inp in inputs_bl
    ]

    return SolutionEvaluator.evaluate(
        runnable_sol=runnable_sol,
        inputs=inputs,
        ref_outputs=ref_outputs_bl,
        ref_mean_latency_ms=ref_mean_latency_ms,
        cfg=cfg,
        device=device,
        log_path=log_path,
        defn=defn
    )
