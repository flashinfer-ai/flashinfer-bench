import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.runner import BaselineHandle, DeviceBaseline
from flashinfer_bench.bench.runner.runner_utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    make_eval,
    normalize_outputs,
)
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile.registry import get_builder_registry
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import torch_dtype_from_def

from .evaluator import Evaluator


class DefaultEvaluator(Evaluator):
    @classmethod
    def can_evaluate(cls, defn: Definition) -> bool:
        return True

    @classmethod
    def build_baseline(
        cls,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        traceset_root: Optional[Path] = None,
    ) -> DeviceBaseline:
        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}
        ref_runnable = get_builder_registry().build_reference(defn)
        loaded_stensors = (
            load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: List[Dict[str, Any]] = []
        outputs: List[Dict[str, torch.Tensor]] = []

        for _ in range(cfg.num_trials):
            inp = gen_inputs(defn, workload, device=device, stensors=loaded_stensors)
            inputs.append(inp)

            with torch.no_grad():
                out = ref_runnable(**inp)
            torch.cuda.synchronize(device)
            out = normalize_outputs(
                out,
                device=device,
                output_names=list(defn.outputs.keys()),
                output_dtypes=output_dtypes,
            )
            outputs.append(out)

        latencies: List[float] = []
        for inp in inputs:
            ms = time_runnable(ref_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
            latencies.append(ms)

        mean_latency_ms = sum(latencies) / float(len(latencies))

        handle = BaselineHandle(uuid.uuid4().hex)

        return DeviceBaseline(
            handle=handle,
            defn=defn,
            device=device,
            inputs=inputs,
            outputs=outputs,
            mean_latency_ms=mean_latency_ms,
        )

    @classmethod
    def check_correctness(
        cls,
        defn: Definition,
        sol_runnable: Runnable,
        inputs: List[Dict[str, Any]],
        ref_outputs: List[Dict[str, torch.Tensor]],
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        output_names = list(ref_outputs[0].keys())
        output_dtypes = {k: v.dtype for k, v in ref_outputs[0].items()}

        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False

        for trial, inp in enumerate(inputs):
            try:
                with torch.no_grad():
                    out = sol_runnable(**inp)
                torch.cuda.synchronize(device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            out = normalize_outputs(
                out, device=device, output_names=output_names, output_dtypes=output_dtypes
            )
            ref_out = ref_outputs[trial]

            for k in ref_out.keys():
                # Shape validation
                if k not in out:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                if tuple(out[k].shape) != tuple(ref_out[k].shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                # Dtype validation
                if out[k].dtype != ref_out[k].dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                    )

                # Non-finite values check
                non_finite_err_val = None
                if torch.isinf(out[k]).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(out[k]).any().item():
                    non_finite_err_val = float("nan")

                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
                    )
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                    )

                # Compute error statistics
                abs_err, rel_err, exceeds_tol, _ = compute_error_stats(out[k], ref_out[k], cfg)

                if exceeds_tol:
                    numerical_incorrect = True

                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None

    @classmethod
    def eval_performance(
        cls,
        sol_runnable: Runnable,
        inputs: List[Dict[str, Any]],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Performance, Optional[Evaluation]]:
        sol_latencies: List[float] = []

        try:
            for inp in inputs:
                ms = time_runnable(sol_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
                sol_latencies.append(ms)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        if not sol_latencies:
            print("Failed to collect solution latencies", file=sys.stderr)
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        sol_mean_latency_ms = sum(sol_latencies) / float(len(sol_latencies))
        performance = Performance(
            latency_ms=sol_mean_latency_ms,
            reference_latency_ms=ref_mean_latency_ms,
            speedup_factor=(ref_mean_latency_ms / sol_mean_latency_ms),
        )

        return performance, None
