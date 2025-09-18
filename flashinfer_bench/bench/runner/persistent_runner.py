from __future__ import annotations

import os
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from torch import multiprocessing as mp

from flashinfer_bench.compile.registry import get_registry
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution
from flashinfer_bench.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import env_snapshot, redirect_stdio_to_file, torch_dtype_from_def

from ..config import BenchmarkConfig
from ..runner import (
    BaselineHandle,
    DeviceBaseline,
    Runner,
    RunnerError,
    RunnerFatalError,
)
from ..timing import time_runnable
from .runner_utils import gen_inputs, load_safetensors, make_eval, normalize_outputs, rand_tensor


class WorkerCommand(Enum):
    COMPILE_SOLUTION = "compile_solution"
    RUN_SOLUTION = "run_solution"
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    READY = "ready"
    COMPILED = "compiled"
    EVALUATION = "evaluation"
    ERROR = "error"
    HEALTHY = "healthy"
    CORRUPTED = "corrupted"


@dataclass
class SolutionCacheKey:
    definition_hash: str
    solution_hash: str
    
    @classmethod
    def from_defn_sol(cls, defn: Definition, sol: Solution) -> "SolutionCacheKey":
        defn_str = f"{defn.name}:{defn.type}:{str(sorted(defn.inputs.items()))}:{str(sorted(defn.outputs.items()))}"
        
        # solution hash formed by name, spec, and source content
        sources_hash = "|".join(f"{src.path}:{src.content}" for src in sol.sources)
        sol_str = f"{sol.name}:{sol.spec.entry_point}:{sol.spec.language.value}:{sources_hash}"
        
        return cls(
            definition_hash=hashlib.sha256(defn_str.encode()).hexdigest()[:16],
            solution_hash=hashlib.sha256(sol_str.encode()).hexdigest()[:16]
        )
    
    def __hash__(self) -> int:
        return hash((self.definition_hash, self.solution_hash))


@dataclass
class CachedSolution:
    """Cached compiled solution."""
    key: SolutionCacheKey
    runnable: Runnable
    compiled_at: float
    definition: Definition
    solution: Solution


@dataclass
class SolutionFailureRecord:
    """Track failures for a solution."""
    key: SolutionCacheKey
    failure_count: int
    last_error: str
    last_status: EvaluationStatus
    last_failure_time: float


class PersistentRunner(Runner):
    """
    Persistent runner that caches compiled solutions and uses long-lived worker processes.
    Only restarts workers when GPU context corruption is detected.
    """

    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
        super().__init__(device, log_dir)
        self._baselines: Dict[BaselineHandle, DeviceBaseline] = {}
        self._registry = get_registry()
        
        # Solution cache
        self._solution_cache: Dict[SolutionCacheKey, CachedSolution] = {}
        
        # Failure tracking
        self._failure_records: Dict[SolutionCacheKey, SolutionFailureRecord] = {}
        self._max_failures = 3 #TODO(Alex): make this configurable
        
        self._worker_proc: Optional[mp.Process] = None
        self._parent_conn: Optional[mp.connection.Connection] = None
        self._worker_healthy = False
        self._worker_restart_count = 0
        self._max_worker_restarts = 3 #TODO(Alex): make this configurable
        
        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker_proc is not None and self._worker_proc.is_alive():
            self._shutdown_worker()
        
        self._worker_restart_count += 1

        if self._worker_restart_count > self._max_worker_restarts:
            raise RunnerFatalError(f"Worker on device {self.device} failed to start after {self._max_worker_restarts} attempts")
        
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe(duplex=True)
        
        self._worker_proc = ctx.Process(
            target=_persistent_worker_main,
            args=(child_conn, self.device, self._log_dir),
            daemon=True,
        )
        self._worker_proc.start()
        
        try:
            msg = self._parent_conn.recv()
            if msg.get("cmd") == WorkerResponse.READY.value:
                self._worker_healthy = True
                print(f"Persistent worker started for device {self.device}")
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
                self._worker_proc.join(timeout=2)
            except Exception:
                pass
            if self._worker_proc.is_alive():
                try:
                    self._worker_proc.terminate()
                except Exception:
                    pass
            self._worker_proc = None
        
        self._worker_healthy = False

    def _check_worker_health(self) -> bool:
        if not self._worker_healthy or self._parent_conn is None:
            return False
        
        try:
            self._parent_conn.send({"cmd": WorkerCommand.HEALTH_CHECK.value})
            msg = self._parent_conn.recv()
            
            if msg.get("cmd") == WorkerResponse.HEALTHY.value:
                return True
            elif msg.get("cmd") == WorkerResponse.CORRUPTED.value:
                print(f"GPU context corrupted on device {self.device}, restarting worker")
                self._start_worker()
                return self._worker_healthy
            else:
                print(f"Unexpected health check response: {msg}")
                return False
                
        except Exception as e:
            print(f"Health check failed: {e}, restarting worker")
            self._start_worker()
            return self._worker_healthy

    def _should_skip_solution(self, cache_key: SolutionCacheKey) -> Optional[SolutionFailureRecord]:
        if cache_key in self._failure_records:
            record = self._failure_records[cache_key]
            if record.failure_count >= self._max_failures:
                return record
        return None

    def _record_failure(self, cache_key: SolutionCacheKey, error: str, status: EvaluationStatus) -> None:
        if cache_key in self._failure_records:
            record = self._failure_records[cache_key]
            record.failure_count += 1
            record.last_error = error
            record.last_status = status
            record.last_failure_time = time.time()
        else:
            self._failure_records[cache_key] = SolutionFailureRecord(
                key=cache_key,
                failure_count=1,
                last_error=error,
                last_status=status,
                last_failure_time=time.time()
            )

    def _clear_failure_record(self, cache_key: SolutionCacheKey) -> None:
        """Clear failure record when solution succeeds."""
        self._failure_records.pop(cache_key, None)

    def _get_or_compile_solution(self, defn: Definition, sol: Solution) -> str:
        cache_key = SolutionCacheKey.from_defn_sol(defn, sol)
        
        failure_record = self._should_skip_solution(cache_key)
        if failure_record is not None:
            print(f"Skipping solution {sol.name} due to {failure_record.failure_count} consecutive failures")
            raise RunnerError(f"Solution skipped after {failure_record.failure_count} failures. Last error: {failure_record.last_error}")
        
        if cache_key in self._solution_cache:
            cached = self._solution_cache[cache_key]
            print(f"Using cached solution: {sol.name}")
            return f"{cache_key.definition_hash}:{cache_key.solution_hash}"
        
        if not self._check_worker_health():
            raise RunnerError("Worker is not healthy")
        
        compile_msg = {
            "cmd": WorkerCommand.COMPILE_SOLUTION.value,
            "definition": defn,
            "solution": sol,
            "cache_key": f"{cache_key.definition_hash}:{cache_key.solution_hash}"
        }
        
        try:
            self._parent_conn.send(compile_msg)
            response = self._parent_conn.recv()
            
            if response.get("cmd") == WorkerResponse.COMPILED.value:
                cached_sol = CachedSolution(
                    key=cache_key,
                    runnable=None,  # Not needed in parent process
                    compiled_at=time.time(),
                    definition=defn,
                    solution=sol
                )
                self._solution_cache[cache_key] = cached_sol
                self._clear_failure_record(cache_key)
                print(f"Compiled and cached solution: {sol.name}")
                return f"{cache_key.definition_hash}:{cache_key.solution_hash}"
            
            elif response.get("cmd") == WorkerResponse.ERROR.value:
                error_msg = response.get("error", "Unknown compilation error")
                self._record_failure(cache_key, error_msg, EvaluationStatus.COMPILE_ERROR)
                raise RunnerError(f"Compilation failed: {error_msg}")
            
            else:
                error_msg = f"Unexpected compilation response: {response}"
                self._record_failure(cache_key, error_msg, EvaluationStatus.COMPILE_ERROR)
                raise RunnerError(error_msg)
                
        except Exception as e:
            self._worker_healthy = False
            error_msg = f"Failed to compile solution: {e}"
            self._record_failure(cache_key, error_msg, EvaluationStatus.COMPILE_ERROR)
            raise RunnerError(error_msg)

    def run_ref(
        self,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        traceset_root: Optional[Path] = None,
    ) -> BaselineHandle:
        torch.cuda.set_device(int(self.device.split(":")[1]))
        dev = torch.device(self.device)

        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}
        runnable_ref = self._registry.build_reference(defn)
        st_cpu = (
            load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs_all: List[Dict[str, Any]] = []
        ref_out_all: List[Dict[str, torch.Tensor]] = []
        for _ in range(cfg.num_trials):
            inp = gen_inputs(defn, workload, device=self.device, stensors=st_cpu)
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
            ms = time_runnable(runnable_ref, inp, cfg.warmup_runs, cfg.iterations, self.device)
            ref_lat_all.append(ms)

        ref_mean_latency_ms = sum(ref_lat_all) / float(len(ref_lat_all))

        handle = BaselineHandle(uuid.uuid4().hex)

        self._baselines[handle] = DeviceBaseline(
            handle=handle,
            defn=defn,
            device=self.device,
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
        
        cache_key = SolutionCacheKey.from_defn_sol(bl.defn, sol)
        failure_record = self._should_skip_solution(cache_key)
        if failure_record is not None:
            print(f"Skipping solution {sol.name} due to {failure_record.failure_count} consecutive failures")
            return make_eval(
                status=failure_record.last_status,
                device=self.device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=f"Solution skipped after {failure_record.failure_count} failures. Last error: {failure_record.last_error}",
            )

        try:
            cache_key_str = self._get_or_compile_solution(bl.defn, sol)
        except Exception as e:
            return make_eval(
                status=EvaluationStatus.COMPILE_ERROR,
                device=self.device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=str(e),
            )
        
        # Ensure worker is healthy
        if not self._check_worker_health():
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self.device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error="Worker is not healthy",
            )
        
        # Send evaluation request
        eval_msg = {
            "cmd": WorkerCommand.RUN_SOLUTION.value,
            "cache_key": cache_key_str,
            "inputs": bl.inputs_dev,
            "ref_outputs": bl.ref_outputs_dev,
            "ref_mean_latency_ms": bl.ref_mean_latency_ms,
            "config": cfg,
            "solution_name": sol.name
        }
        
        try:
            self._parent_conn.send(eval_msg)
            response = self._parent_conn.recv()
            
            if response.get("cmd") == WorkerResponse.EVALUATION.value:
                evaluation = response["evaluation"]
                if evaluation.status == EvaluationStatus.PASSED:
                    self._clear_failure_record(cache_key)
                elif evaluation.status in (EvaluationStatus.RUNTIME_ERROR, EvaluationStatus.INCORRECT_SHAPE, 
                                          EvaluationStatus.INCORRECT_DTYPE):
                    self._record_failure(cache_key, evaluation.error or "Evaluation failed", evaluation.status)
                return evaluation
            elif response.get("cmd") == WorkerResponse.ERROR.value:
                error_msg = response.get("error", "Unknown evaluation error")
                self._record_failure(cache_key, error_msg, EvaluationStatus.RUNTIME_ERROR)
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=self.device,
                    log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                    error=error_msg,
                )
            else:
                error_msg = f"Unexpected evaluation response: {response}"
                self._record_failure(cache_key, error_msg, EvaluationStatus.RUNTIME_ERROR)
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=self.device,
                    log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                    error=error_msg,
                )
                
        except Exception as e:
            self._worker_healthy = False
            error_msg = f"Failed to evaluate solution: {e}"
            self._record_failure(cache_key, error_msg, EvaluationStatus.RUNTIME_ERROR)
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self.device,
                log_file=os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log"),
                error=error_msg,
            )

    def release(self, baseline: BaselineHandle) -> None:
        """Release baseline."""
        self._baselines.pop(baseline, None)

    def close(self) -> None:
        """Cleanup resources."""
        self._shutdown_worker()
        self._baselines.clear()
        self._solution_cache.clear()
        self._failure_records.clear()


def _persistent_worker_main(
    conn: mp.connection.Connection,
    device: str,
    log_dir: str,
) -> None:
    """
    Long-lived worker process that handles multiple solution evaluations.
    Caches compiled solutions to avoid recompilation.
    """
    try:
        torch.cuda.set_device(int(device.split(":")[1]))
        registry = get_registry()
        
        # Worker-local solution cache
        solution_cache: Dict[str, Runnable] = {}
        
        conn.send({"cmd": WorkerResponse.READY.value})
        
        while True:
            try:
                msg = conn.recv()
                cmd = msg.get("cmd")
                
                if cmd == WorkerCommand.SHUTDOWN.value:
                    break
                
                elif cmd == WorkerCommand.HEALTH_CHECK.value:
                    try:
                        # Simple GPU health check
                        test_tensor = torch.zeros(1, device=device)
                        test_tensor += 1
                        torch.cuda.synchronize(device=device)
                        del test_tensor
                        conn.send({"cmd": WorkerResponse.HEALTHY.value})
                    except Exception:
                        conn.send({"cmd": WorkerResponse.CORRUPTED.value})
                        break
                
                elif cmd == WorkerCommand.COMPILE_SOLUTION.value:
                    defn = msg["definition"]
                    sol = msg["solution"]
                    cache_key = msg["cache_key"]
                    
                    try:
                        if cache_key not in solution_cache:
                            # Compile solution
                            runnable_sol = registry.build(defn, sol)
                            solution_cache[cache_key] = runnable_sol
                        
                        conn.send({"cmd": WorkerResponse.COMPILED.value})
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        conn.send({"cmd": WorkerResponse.ERROR.value, "error": error_msg})
                
                elif cmd == WorkerCommand.RUN_SOLUTION.value:
                    cache_key = msg["cache_key"]
                    inputs_bl = msg["inputs"]
                    ref_outputs_bl = msg["ref_outputs"]
                    ref_mean_latency_ms = msg["ref_mean_latency_ms"]
                    cfg = msg["config"]
                    solution_name = msg["solution_name"]
                    
                    log_path = os.path.join(log_dir, f"{solution_name}_{time.time()}.log")
                    
                    try:
                        if cache_key not in solution_cache:
                            raise RuntimeError(f"Solution not found in cache: {cache_key}")
                        
                        runnable_sol = solution_cache[cache_key]
                        
                        evaluation = _evaluate_solution_worker(
                            runnable_sol=runnable_sol,
                            inputs_bl=inputs_bl,
                            ref_outputs_bl=ref_outputs_bl,
                            ref_mean_latency_ms=ref_mean_latency_ms,
                            cfg=cfg,
                            device=device,
                            log_path=log_path
                        )
                        
                        conn.send({"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation})
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        evaluation = make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=device,
                            log_file=log_path,
                            error=error_msg,
                        )
                        conn.send({"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation})
                
                else:
                    conn.send({"cmd": WorkerResponse.ERROR.value, "error": f"Unknown command: {cmd}"})
                    
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
    log_path: str
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
    
    # Check correctness
    for t, inp in enumerate(inputs):
        try:
            with torch.no_grad():
                out = runnable_sol(**inp)
            torch.cuda.synchronize(device=device)
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=device,
                log_file=log_path,
                error=error_msg,
            )

        out_t = normalize_outputs(
            out,
            device=torch.device(device),
            output_names=output_names,
            output_dtypes=output_dtypes,
        )
        ref_t = ref_outputs_bl[t]
        
        for k in ref_t.keys():
            if k not in out_t:
                return make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE,
                    device=device,
                    log_file=log_path,
                )
            if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                return make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE,
                    device=device,
                    log_file=log_path,
                )
            if out_t[k].dtype != ref_t[k].dtype:
                return make_eval(
                    status=EvaluationStatus.INCORRECT_DTYPE,
                    log_file=log_path,
                    device=device,
                )

            diff = (out_t[k] - ref_t[k]).abs()
            abs_err = float(diff.max().item()) if diff.numel() > 0 else 0.0
            denom = ref_t[k].abs().max()
            denom_v = float(denom.item()) if denom.numel() > 0 else 0.0
            rel_err = abs_err / (denom_v + 1e-12)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

    correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)
    if max_abs > cfg.atol or max_rel > cfg.rtol:
        return make_eval(
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
    
    return make_eval(
        status=EvaluationStatus.PASSED,
        device=device,
        log_file=log_path,
        correctness=correctness,
        performance=performance,
    )



