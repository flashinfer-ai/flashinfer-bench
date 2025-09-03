from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from triton.testing import do_bench

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import AxisConst, AxisVar, Definition
from flashinfer_bench.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import env_snapshot, torch_dtype_from_def


class BaselineHandle(str):
    pass


@dataclass
class _DeviceBaseline:
    handle: BaselineHandle
    device: str
    num_trials: int
    output_names: List[str]
    output_dtypes: Dict[str, torch.dtype]
    inputs_dev: List[Dict[str, Any]]
    ref_outputs_dev: List[Dict[str, torch.Tensor]]
    ref_latencies_ms: List[float]
    ref_mean_latency_ms: float


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

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


class Runner:
    """Single-device runner that builds and reuses a baseline per (Definition, Workload)."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._baselines: Dict[BaselineHandle, _DeviceBaseline] = {}
        self._timing_lock = threading.Lock()

    def _gen_inputs(
        self,
        defn: Definition,
        wl: Workload,
        *,
        host_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        shapes = defn.get_input_shapes(wl.axes)
        dev = torch.device(self.device)
        out: Dict[str, Any] = {}

        for name, spec in defn.inputs.items():
            dtype = torch_dtype_from_def(spec.dtype)

            if name in wl.inputs and wl.inputs[name].type == "safetensors":
                if host_tensors is None or name not in host_tensors:
                    raise RuntimeError(f"Missing host tensor for safetensors input '{name}'")
                t_cpu = host_tensors[name]
                out[name] = t_cpu.to(device=dev, non_blocking=True)
            elif name in wl.inputs and wl.inputs[name].type == "scalar":
                out[name] = wl.inputs[name].value
            else:
                shape = shapes[name]
                out[name] = _rand_tensor(shape, dtype, dev)
        return out

    def _time_runnable(
        self,
        fn: Runnable,
        inputs: Dict[str, Any],
        warmup: int,
        iters: int,
    ) -> float:
        # Serialize timing on this device to avoid interference
        with self._timing_lock:
            with torch.no_grad():
                fn(**inputs)
            torch.cuda.synchronize(device=torch.device(self.device))

            return do_bench(
                lambda: fn(**inputs),
                warmup=warmup,
                rep=iters,
            )

    def run_reference(
        self,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        runnable_ref: Runnable,
        host_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> BaselineHandle:
        dev = torch.device(self.device)
        if dev.type != "cuda":
            raise RuntimeError("Runner currently supports CUDA devices only")

        inputs_all: List[Dict[str, Any]] = []
        ref_out_all: List[Dict[str, Any]] = []
        ref_lat_all: List[float] = []
        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}

        # Materialize per-trial inputs and reference outputs
        for _ in range(cfg.num_trials):
            inputs = self._gen_inputs(defn, workload, host_tensors=host_tensors)
            inputs_all.append(inputs)
            with torch.no_grad():
                out = runnable_ref(**inputs)
            torch.cuda.synchronize(device=torch.device(self.device))
            out_dict = _normalize_outputs(
                out,
                device=dev,
                output_names=list(defn.outputs.keys()),
                output_dtypes=output_dtypes,
            )
            ref_out_all.append(out_dict)

        # Timing for reference on each trial (mean over iterations)
        for t in range(cfg.num_trials):
            with torch.no_grad():
                mean_ms = self._time_runnable(
                    runnable_ref, inputs_all[t], cfg.warmup_runs, cfg.iterations
                )
            ref_lat_all.append(mean_ms)

        if not ref_lat_all or not ref_out_all:
            raise RuntimeError("Failed to collect reference outputs and latencies")

        handle = BaselineHandle(uuid.uuid4().hex)
        baseline = _DeviceBaseline(
            handle=handle,
            device=self.device,
            num_trials=cfg.num_trials,
            output_names=list(defn.outputs.keys()),
            output_dtypes=output_dtypes,
            inputs_dev=inputs_all,
            ref_outputs_dev=ref_out_all,
            ref_latencies_ms=ref_lat_all,
            ref_mean_latency_ms=sum(ref_lat_all) / float(len(ref_lat_all)),
        )
        self._baselines[handle] = baseline
        return handle

    def run_solution(
        self,
        solution_runnable: Runnable,
        baseline: BaselineHandle,
        cfg: BenchmarkConfig,
    ) -> Evaluation:
        if baseline not in self._baselines:
            raise KeyError(f"Baseline handle not found: {baseline}")
        bl = self._baselines[baseline]
        meta = solution_runnable.meta or {}
        name = meta.get("name") or meta.get("solution") or meta.get("entry_point") or "runnable"
        log_file = f"{name}.log"

        # First, run one trial for structural/numerical check
        dev = torch.device(self.device)
        try:
            inputs0 = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in bl.inputs_dev[0].items()
            }
            with torch.no_grad():
                out0 = solution_runnable(**inputs0)
        except Exception as e:
            # TODO(shanli): redirect error to log file
            return self._create_evaluation(
                status=EvaluationStatus.RUNTIME_ERROR,
                log_file=log_file,
            )

        out0_dict = _normalize_outputs(
            out0,
            device=dev,
            output_names=bl.output_names,
            output_dtypes=bl.output_dtypes,
        )
        ref0 = bl.ref_outputs_dev[0]

        max_abs = 0.0
        max_rel = 0.0
        for k in ref0.keys():
            if k not in out0_dict:
                return self._create_evaluation(
                    status=EvaluationStatus.INCORRECT_SHAPE, log_file=log_file
                )
            if tuple(out0_dict[k].shape) != tuple(ref0[k].shape):
                return self._create_evaluation(
                    status=EvaluationStatus.INCORRECT_SHAPE,
                    log_file=log_file,
                )
            if out0_dict[k].dtype != ref0[k].dtype:
                return self._create_evaluation(
                    status=EvaluationStatus.INCORRECT_DTYPE,
                    log_file=log_file,
                )

            # Passed structural checks
            diff = (out0_dict[k] - ref0[k]).abs()
            abs_err = float(diff.max().item()) if diff.numel() > 0 else 0.0
            denom = ref0[k].abs().max()
            denom_v = float(denom.item()) if denom.numel() > 0 else 0.0
            rel_err = abs_err / (denom_v + 1e-12)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

        correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)
        if max_abs > cfg.atol or max_rel > cfg.rtol:
            return self._create_evaluation(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                log_file=log_file,
                correctness=correctness,
            )

        # Passed numerical checks; now measure implementation performance
        impl_lats: List[float] = []
        for t in range(bl.num_trials):
            inputs = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in bl.inputs_dev[t].items()
            }
            mean_ms = self._time_runnable(
                solution_runnable, inputs, cfg.warmup_runs, cfg.iterations
            )
            impl_lats.append(mean_ms)

        impl_mean = sum(impl_lats) / float(len(impl_lats)) if impl_lats else 0.0
        performance = Performance(
            latency_ms=impl_mean,
            reference_latency_ms=bl.ref_mean_latency_ms,
            speedup_factor=(bl.ref_mean_latency_ms / impl_mean) if impl_mean > 0 else 0.0,
        )
        return self._create_evaluation(
            status=EvaluationStatus.PASSED,
            log_file=log_file,
            correctness=correctness,
            performance=performance,
        )

    def _create_evaluation(
        self,
        status: EvaluationStatus,
        log_file: str,
        correctness: Optional[Correctness] = None,
        performance: Optional[Performance] = None,
    ) -> Evaluation:
        return Evaluation(
            status=status,
            log_file=log_file,
            environment=env_snapshot(self.device),
            timestamp=datetime.now().isoformat(),
            correctness=correctness,
            performance=performance,
        )

    def release(self, baseline: BaselineHandle) -> None:
        self._baselines.pop(baseline, None)
