from __future__ import annotations

import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.benchmark_config import BenchmarkConfig
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
from flashinfer_bench.bench.runner.runner_utils import (
    compute_error_stats,
    make_eval,
    normalize_outputs,
)
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import Correctness, Evaluation, EvaluationStatus


class LowBitEvaluator(DefaultEvaluator):
    @override
    @classmethod
    def can_evaluate(cls, defn: Definition) -> bool:
        return "moe_fp8_block_scale" in defn.name

    @override
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
        min_matched_ratio = 1.0

        if cfg.required_matched_ratio is None:
            cfg = replace(cfg, required_matched_ratio=0.95)

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
                if k not in out:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                if tuple(out[k].shape) != tuple(ref_out[k].shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                if out[k].dtype != ref_out[k].dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                    )

                non_finite_err_val: Optional[float] = None
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

                abs_err, rel_err, exceeds_tol, matched_ratio = compute_error_stats(
                    out[k], ref_out[k], cfg
                )

                if exceeds_tol:
                    numerical_incorrect = True

                min_matched_ratio = min(min_matched_ratio, matched_ratio)
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(
            max_relative_error=max_rel,
            max_absolute_error=max_abs,
            extra={"matched_ratio": min_matched_ratio},
        )

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None
