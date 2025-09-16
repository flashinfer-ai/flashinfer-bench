from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class MultiProcessRunner(Runner):
    """Each instance binds to a CUDA device; the baseline resides in the main process; each Solution starts an independent Worker process for strong isolation."""

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._baselines: Dict[BaselineHandle, DeviceBaseline] = {}
        self._registry = get_registry()

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
        if baseline not in self._baselines:
            raise RunnerError(f"Baseline handle not found: {baseline}")
        bl = self._baselines[baseline]

        log_path = os.path.join(self._log_dir, f"{sol.name}_{time.time()}.log")
        # New process for each solution run
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_solution_worker_main,
            args=(child_conn, self.device, bl.defn, sol, cfg, log_path),
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
                        device=self.device,
                        log_file=log_path,
                        error=error_msg,
                    )
                    break

                else:
                    print(f"Unknown worker command: {cmd}")
                    continue

        except EOFError as e:
            print(f"Worker crashed (EOF) running {sol.name}: {e}")
        except Exception as e:
            print(f"Unknown error running {sol.name}: {e}")
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
                device=self.device,
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
        for t, inp in enumerate(inputs):
            try:
                with torch.no_grad():
                    out = runnable_sol(**inp)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                import traceback

                error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                ev = make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=device,
                    log_file=log_path,
                    error=error_msg,
                )
                conn.send({"cmd": "EVAL", "evaluation": ev})
                return

            out_t = normalize_outputs(
                out,
                device=torch.device(device),
                output_names=output_names,
                output_dtypes=output_dtypes,
            )
            ref_t = ref_outputs_bl[t]
            for k in ref_t.keys():
                if k not in out_t:
                    ev = make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        device=device,
                        log_file=log_path,
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return
                if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                    ev = make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        log_file=log_path,
                        device=device,
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return
                if out_t[k].dtype != ref_t[k].dtype:
                    ev = make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE,
                        log_file=log_path,
                        device=device,
                    )
                    conn.send({"cmd": "EVAL", "evaluation": ev})
                    return

                diff = (out_t[k] - ref_t[k]).abs()
                abs_err = float(diff.max().item()) if diff.numel() > 0 else 0.0
                denom = ref_t[k].abs().max()
                denom_v = float(denom.item()) if denom.numel() > 0 else 0.0
                rel_err = abs_err / (denom_v + 1e-12)
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)
        if max_abs > cfg.atol or max_rel > cfg.rtol:
            ev = make_eval(
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
        ev = make_eval(
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
