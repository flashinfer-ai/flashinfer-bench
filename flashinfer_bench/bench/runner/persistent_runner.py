from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import multiprocessing as mp

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile import Runnable, get_registry
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
from flashinfer_bench.utils import (
    env_snapshot,
    list_cuda_devices,
    redirect_stdio_to_file,
    torch_dtype_from_def,
)

from .runner import BaselineHandle, DeviceBaseline, Runner, RunnerError, RunnerFatalError

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


def _rand_tensor(shape: List[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)

    # low-precision floats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)

    # booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


def _normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        # Python scalar -> 0-D tensor for comparison
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError("Single Tensor returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def _compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, cfg: BenchmarkConfig
) -> Tuple[float, float, bool]:
    """Return (max_abs_err, max_rel_err, exceeds_tol) for elementwise comparison."""

    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    diff = (x - y).abs()
    if diff.numel() == 0:
        return 0.0, 0.0, False

    tol = cfg.atol + cfg.rtol * y.abs()
    ratio = diff / tol.clamp_min(torch.finfo(torch.float32).tiny)

    max_abs = float(diff.max().item())
    max_rel = float(ratio.max().item())

    return max_abs, max_rel, bool(max_rel > 1.0)


def _load_safetensors(
    defn: Definition, wl: Workload, traceset_root: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    try:
        import safetensors.torch as st
    except Exception:
        raise RuntimeError("safetensors is not available in the current environment")

    expected = defn.get_input_shapes(wl.axes)
    stensors: Dict[str, torch.Tensor] = {}
    for name, input_spec in wl.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = input_spec.path
        if traceset_root is not None and not Path(path).is_absolute():
            path = str(traceset_root / path)

        tensors = st.load_file(path)
        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]
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
        stensors[name] = t
    return stensors


def _gen_inputs(
    defn: Definition, wl: Workload, device: str, stensors: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    shapes = defn.get_input_shapes(wl.axes)
    dev = torch.device(device)
    out: Dict[str, Any] = {}

    for name, spec in defn.inputs.items():
        dtype = torch_dtype_from_def(spec.dtype)

        if name in wl.inputs and wl.inputs[name].type == "safetensors":
            if stensors is None or name not in stensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = stensors[name]
            out[name] = t_cpu.to(device=dev, non_blocking=True)
        elif name in wl.inputs and wl.inputs[name].type == "scalar":
            out[name] = wl.inputs[name].value
        else:  # random
            shape = shapes[name]
            out[name] = _rand_tensor(shape, dtype, dev)
    return out


def _make_eval(
    status: EvaluationStatus,
    device: str,
    log_file: str,
    correctness: Optional[Correctness] = None,
    performance: Optional[Performance] = None,
    error: Optional[str] = None,
) -> Evaluation:
    return Evaluation(
        status=status,
        log_file=log_file,
        environment=env_snapshot(device),
        timestamp=datetime.now().isoformat(),
        correctness=correctness,
        performance=performance,
        error=error,
    )


class PersistentSubprocessWorker:
    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
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
        """
        Returns:
            bool: True if restart was successful, False otherwise
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
            record.failure_count += 1
            record.last_error = error
            record.last_status = status
            record.last_failure_time = time.time()
        else:
            self._failure_records[solution_name] = SolutionFailureRecord(
                solution_name=solution_name,
                failure_count=1,
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
            _load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs_all: List[Dict[str, Any]] = []
        ref_out_all: List[Dict[str, torch.Tensor]] = []
        for _ in range(cfg.num_trials):
            inp = _gen_inputs(defn, workload, device=self._device, stensors=st_cpu)
            inputs_all.append(inp)

            with torch.no_grad():
                out = runnable_ref(**inp)
            torch.cuda.synchronize(device=dev)
            ref_out = _normalize_outputs(
                out, device=dev, output_names=list(defn.outputs.keys()), output_dtypes=output_dtypes
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
            return _make_eval(
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
            return _make_eval(
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
                        ):
                            self._record_failure(
                                sol.name, evaluation.error or "Evaluation failed", evaluation.status
                            )
                        return evaluation
                    elif response.get("cmd") == WorkerResponse.ERROR.value:
                        error_msg = response.get("error", "Unknown evaluation error")
                        self._record_failure(sol.name, error_msg, EvaluationStatus.RUNTIME_ERROR)
                        return _make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                            error=error_msg,
                        )
                    else:
                        error_msg = f"Unexpected evaluation response: {response}"
                        self._record_failure(sol.name, error_msg, EvaluationStatus.RUNTIME_ERROR)
                        return _make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                            error=error_msg,
                        )

                except (EOFError, ConnectionResetError, OSError) as e:
                    error_msg = f"Connection error during evaluation: {e}"
                    return _make_eval(
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
                    return _make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                        error=error_msg,
                    )
            else:
                error_msg = f"Evaluation timeout after 300 seconds for solution {sol.name}"
                return _make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=self._device,
                    log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                    error=error_msg,
                )

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            error_msg = f"Connection broken during evaluation: {e}"
            return _make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Failed to communicate with worker: {e}"
            return _make_eval(
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

        Args:
            failed_workers: List of workers that have failed
            increment_retries: Whether to increment retry count (True for health failures, False for solution failures)
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
        """
        Run a workload with the given solutions and return evaluation results.
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
                            return _make_eval(
                                status=EvaluationStatus.RUNTIME_ERROR,
                                device=worker._device,
                                log_file=os.path.join(
                                    self._log_dir, f"{solution.name}_{time.time()}.log"
                                ),
                                error=f"Failed to rebuild baseline after restart: {e}",
                            )
                    else:
                        LOGGER.error(f"Failed to restart worker on device {worker._device}")
                        return _make_eval(
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
                return _make_eval(
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
    """
    Long-lived worker process that handles solution evaluations.
    Caches compiled solutions to avoid recompilation (handled in builder registry).
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
                        )

                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )

                    except Exception as e:
                        import traceback

                        error_msg = (
                            f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        )
                        evaluation = _make_eval(
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
) -> Evaluation:
    inputs: List[Dict[str, Any]] = [
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
        for inp in inputs_bl
    ]

    if not ref_outputs_bl:
        raise RuntimeError("No reference outputs provided")

    output_names = list(ref_outputs_bl[0].keys())
    output_dtypes = {k: v.dtype for k, v in ref_outputs_bl[0].items()}

    max_abs = 0.0
    max_rel = 0.0
    numerical_incorrect = False

    # Check correctness
    for t, inp in enumerate(inputs):
        try:
            with torch.no_grad():
                out = runnable_sol(**inp)
            torch.cuda.synchronize(device=device)
        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return _make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=device,
                log_file=log_path,
                error=error_msg,
            )

        out_t = _normalize_outputs(
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )
        ref_t = ref_outputs_bl[t]

        for k in ref_t.keys():
            if k not in out_t:
                return _make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_file=log_path
                )
            if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                return _make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_file=log_path
                )
            if out_t[k].dtype != ref_t[k].dtype:
                return _make_eval(
                    status=EvaluationStatus.INCORRECT_DTYPE, log_file=log_path, device=device
                )

            # Check for non-finite values
            non_finite_err_val = None
            if torch.isinf(out_t[k]).any().item():
                non_finite_err_val = float("inf")
            elif torch.isnan(out_t[k]).any().item():
                non_finite_err_val = float("nan")
            if non_finite_err_val is not None:
                correctness = Correctness(
                    max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
                )
                return _make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    log_file=log_path,
                    device=device,
                    correctness=correctness,
                )

            abs_err, rel_err, exceeds_tol = _compute_error_stats(out_t[k], ref_t[k], cfg)

            if exceeds_tol:
                numerical_incorrect = True

            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

    correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)
    if numerical_incorrect:
        return _make_eval(
            status=EvaluationStatus.INCORRECT_NUMERICAL,
            log_file=log_path,
            correctness=correctness,
            device=device,
        )

    # Measure performance
    soln_lats: List[float] = []
    for inp in inputs:
        lat_ms = time_runnable(runnable_sol, inp, cfg.warmup_runs, cfg.iterations, device)
        soln_lats.append(lat_ms)

    if not soln_lats:
        raise RuntimeError("Failed to collect solution latencies")

    soln_mean_latency_ms = sum(soln_lats) / float(len(soln_lats))
    performance = Performance(
        latency_ms=soln_mean_latency_ms,
        reference_latency_ms=ref_mean_latency_ms,
        speedup_factor=(ref_mean_latency_ms / soln_mean_latency_ms),
    )

    return _make_eval(
        status=EvaluationStatus.PASSED,
        device=device,
        log_file=log_path,
        correctness=correctness,
        performance=performance,
    )
