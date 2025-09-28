from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import Correctness, Definition, Evaluation, EvaluationStatus
from flashinfer_bench.utils import torch_dtype_from_def

from .runner_utils import (
    compute_error_stats,
    compute_frequency_distribution,
    detect_sampling_type,
    make_eval,
    normalize_outputs,
    validate_sampling_tokens,
)


class SamplingValidator:
    @staticmethod
    def validate_correctness(
        runnable_sol: Runnable,
        inputs: List[Dict[str, Any]],
        ref_outputs_bl: List[Dict[str, torch.Tensor]],
        cfg: BenchmarkConfig,
        device: str,
        log_path: str,
        defn: Definition,
    ) -> Tuple[Optional[Evaluation], float, float, bool]:
        sampling_type = detect_sampling_type(defn)
        ref_freq = ref_outputs_bl[0]["frequency_distribution"]
        vocab_size = ref_freq.shape[0]

        inp = inputs[0]
        params = {k: inp[k] for k in ["top_k", "top_p"] if k in inp}

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}

        # Validate correct sampling token set
        for trial_idx in range(100):
            try:
                with torch.no_grad():
                    out = runnable_sol(**inp)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                return (
                    make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=device,
                        log_file=log_path,
                        error=error_msg,
                    ),
                    0.0,
                    0.0,
                    False,
                )

            out_normalized = normalize_outputs(
                out,
                device=torch.device(device),
                output_names=output_names,
                output_dtypes=output_dtypes,
            )
            samples = out_normalized["samples"]

            # Check vocabulary range
            if (samples < 0).any() or (samples >= vocab_size).any():
                invalid_samples = samples[(samples < 0) | (samples >= vocab_size)]
                correctness = Correctness(max_relative_error=1.0, max_absolute_error=1.0)
                return (
                    make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_file=log_path,
                        correctness=correctness,
                        error=f"Samples {invalid_samples.tolist()} out of vocabulary range [0, {vocab_size})",
                    ),
                    1.0,
                    1.0,
                    True,
                )

            # Validate top-p top-k sampling constraints
            probs = inp["probs"]
            if not validate_sampling_tokens(samples, probs, sampling_type, params):
                correctness = Correctness(max_relative_error=1.0, max_absolute_error=1.0)
                return (
                    make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_file=log_path,
                        correctness=correctness,
                        error=f"Samples {samples.tolist()} violate {sampling_type} constraints",
                    ),
                    1.0,
                    1.0,
                    True,
                )

        # Compute frequency distribution and similarity
        try:
            sol_freq = compute_frequency_distribution(
                runnable_sol, [inp], device, defn, num_trials=50000
            )
        except Exception:
            print(traceback.format_exc())
            raise

        similarity = torch.cosine_similarity(sol_freq.unsqueeze(0), ref_freq.unsqueeze(0)).item()
        max_abs, max_rel, _, _ = compute_error_stats(sol_freq, ref_freq, cfg, defn)

        numerical_incorrect = similarity < 0.95

        return None, max_abs, max_rel, numerical_incorrect


class FusedMoeValidator:
    """Validator for fused mixture-of-experts operations."""

    @staticmethod
    def validate_correctness(
        runnable_sol: Runnable,
        inputs: List[Dict[str, Any]],
        ref_outputs_bl: List[Dict[str, torch.Tensor]],
        cfg: BenchmarkConfig,
        device: str,
        log_path: str,
        defn: Definition,
    ) -> Tuple[Optional[Evaluation], float, float, bool, Optional[float]]:
        output_names = list(ref_outputs_bl[0].keys())
        output_dtypes = {k: v.dtype for k, v in ref_outputs_bl[0].items()}

        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        max_percentage = None

        for t, inp in enumerate(inputs):
            try:
                with torch.no_grad():
                    out = runnable_sol(**inp)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                return (
                    make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=device,
                        log_file=log_path,
                        error=error_msg,
                    ),
                    0.0,
                    0.0,
                    False,
                    None,
                )

            out_t = normalize_outputs(
                out,
                device=torch.device(device),
                output_names=output_names,
                output_dtypes=output_dtypes,
            )
            ref_t = ref_outputs_bl[t]

            for k in ref_t.keys():
                # Shape validation
                if k not in out_t:
                    return (
                        make_eval(
                            status=EvaluationStatus.INCORRECT_SHAPE,
                            device=device,
                            log_file=log_path,
                        ),
                        0.0,
                        0.0,
                        False,
                        None,
                    )

                if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                    return (
                        make_eval(
                            status=EvaluationStatus.INCORRECT_SHAPE,
                            device=device,
                            log_file=log_path,
                        ),
                        0.0,
                        0.0,
                        False,
                        None,
                    )

                # Dtype validation
                if out_t[k].dtype != ref_t[k].dtype:
                    return (
                        make_eval(
                            status=EvaluationStatus.INCORRECT_DTYPE,
                            device=device,
                            log_file=log_path,
                        ),
                        0.0,
                        0.0,
                        False,
                        None,
                    )

                # Non-finite values check
                non_finite_err_val = None
                if torch.isinf(out_t[k]).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(out_t[k]).any().item():
                    non_finite_err_val = float("nan")

                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val,
                        max_absolute_error=non_finite_err_val
                    )
                    return (
                        make_eval(
                            status=EvaluationStatus.INCORRECT_NUMERICAL,
                            device=device,
                            log_file=log_path,
                            correctness=correctness,
                        ),
                        non_finite_err_val,
                        non_finite_err_val,
                        True,
                        None,
                    )

                # Compute error statistics
                abs_err, rel_err, exceeds_tol, percentage_used = compute_error_stats(
                    out_t[k], ref_t[k], cfg, defn
                )

                if exceeds_tol:
                    numerical_incorrect = True

                if percentage_used is not None:
                    max_percentage = percentage_used

                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        return None, max_abs, max_rel, numerical_incorrect, max_percentage


def validate_operation_correctness(
    op_type: str,
    runnable_sol: Runnable,
    inputs: List[Dict[str, Any]],
    ref_outputs_bl: List[Dict[str, torch.Tensor]],
    cfg: BenchmarkConfig,
    device: str,
    log_path: str,
    defn: Definition,
) -> Tuple[Optional[Evaluation], float, float, bool, Optional[float]]:
    """
    Args:
        op_type: "sampling" or "fused_moe"
        runnable_sol: Solution runnable
        inputs: Input data
        ref_outputs_bl: Ref outputs
        cfg: Benchmark configuration
        device: Device string
        log_path: Logging path
        defn: FI-B Definition
    Returns:
        (evaluation_result, max_abs_error, max_rel_error, numerical_incorrect, max_percentage)
    """
    if op_type == "sampling":
        eval_result, max_abs, max_rel, numerical_incorrect = SamplingValidator.validate_correctness(
            runnable_sol, inputs, ref_outputs_bl, cfg, device, log_path, defn
        )
        return eval_result, max_abs, max_rel, numerical_incorrect, None

    return FusedMoeValidator.validate_correctness(
        runnable_sol, inputs, ref_outputs_bl, cfg, device, log_path, defn
    )
