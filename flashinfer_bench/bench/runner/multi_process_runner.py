from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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

LOGGER = get_logger("MPRunner")


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

    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)
    if abs_error.numel() == 0:
        return 0.0, 0.0, False

    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())
    exceeds_tol = not torch.allclose(x, y, atol=cfg.atol, rtol=cfg.rtol)

    return max_abs, max_rel, exceeds_tol


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


class SubprocessWorker:
    """Each instance binds to a CUDA device; the baseline resides in the main process; each Solution starts an independent Worker process for strong isolation."""

    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
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
                    evaluation = _make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_file=log_path,
                        error=error_msg,
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
            evaluation = _make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_file=log_path,
                error="Worker process failed unexpectedly",
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
    """Worker process: strong isolation for single Solution. Borrow/return trial data via Pipe and send Evaluation back to parent process."""
    try:
        redirect_stdio_to_file(log_path)
        torch.cuda.set_device(int(device.split(":")[1]))
        registry = get_registry()

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}

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

            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            ev = Evaluation(
                status=EvaluationStatus.COMPILE_ERROR,
                log_file=log_path,
                environment=env_snapshot(device),
                timestamp=datetime.now().isoformat(),
                error=error_msg,
            )
            conn.send({"cmd": "EVAL", "evaluation": ev})
            return

        conn.send({"cmd": "LOAN"})
        loan = conn.recv()

        inputs_bl = loan["inputs"]
        ref_outputs_bl = loan["ref_outputs"]
        ref_mean_latency_ms = loan["ref_mean_latency_ms"]

        inputs: List[Dict[str, Any]] = [
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            for inp in inputs_bl
        ]

        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        for t, inp in enumerate(inputs):
            try:
                with torch.no_grad():
                    out = runnable_sol(**inp)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                import traceback

                error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                ev = _make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=device,
                    log_file=log_path,
                    error=error_msg,
                )
                conn.send({"cmd": "EVAL", "evaluation": ev})
                return

            out_t = _normalize_outputs(
                out,
                device=torch.device(device),
                output_names=output_names,
                output_dtypes=output_dtypes,
            )
            ref_t = ref_outputs_bl[t]
            for k in ref_t.keys():
                if k not in out_t:
                    ev = _make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_file=log_path
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return
                if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                    ev = _make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, log_file=log_path, device=device
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return
                if out_t[k].dtype != ref_t[k].dtype:
                    ev = _make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, log_file=log_path, device=device
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return

                non_finite_err_val = None
                if torch.isinf(out_t[k]).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(out_t[k]).any().item():
                    non_finite_err_val = float("nan")
                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
                    )
                    ev = _make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        log_file=log_path,
                        device=device,
                        correctness=correctness,
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return

                abs_err, rel_err, exceeds_tol = _compute_error_stats(out_t[k], ref_t[k], cfg)

                if exceeds_tol:
                    numerical_incorrect = True

                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)
        if numerical_incorrect:
            ev = _make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                log_file=log_path,
                correctness=correctness,
                device=device,
            )
            conn.send({"cmd": "EVAL", "evaluation": ev})
            return

        # Passed numerical checks; now measure implementation performance
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
        ev = _make_eval(
            status=EvaluationStatus.PASSED,
            device=device,
            log_file=log_path,
            correctness=correctness,
            performance=performance,
        )
        conn.send({"cmd": "EVAL", "evaluation": ev})
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


class MultiProcessRunner(Runner):
    def __init__(self, logger: logging.Logger, log_dir: str = "/tmp/flashinfer_bench") -> None:
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
        """Pick K workers in round-robin fashion."""
        # K = min(len(self._workers), len(solutions))
        if K <= 0 or not self._workers:
            return []
        D = len(self._workers)
        start = self._curr_worker_idx
        sel = [self._workers[(start + i) % D] for i in range(min(K, D))]
        self._curr_worker_idx = (start + K) % D
        return sel

    def _relaunch_worker(self, device: str) -> SubprocessWorker:
        """Relaunch a worker for the given device."""
        self._logger.info(f"Relaunching worker for device {device}")
        return SubprocessWorker(device, self._log_dir)

    def _handle_failed_workers(self, failed_workers: List[SubprocessWorker]) -> None:
        """Handle failed workers by attempting to relaunch them or removing them."""
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
        """Check if there are any healthy workers available."""
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

        Args:
            defn: Definition object
            wl: Workload object
            solutions: List of solutions to evaluate
            config: Benchmark configuration
            root: Root path for the trace set

        Returns:
            Dictionary mapping solution names to their evaluations
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
