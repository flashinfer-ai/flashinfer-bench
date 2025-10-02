from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import Correctness, Definition, Evaluation, EvaluationStatus, Performance
from flashinfer_bench.utils import torch_dtype_from_def

from .runner_utils import (
    compute_error_stats,
    compute_frequency_distribution,
    is_sampling_operation,
    make_eval,
    normalize_outputs,
)


def _detect_sampling_type(defn: Definition) -> str:
    """Detect the type of sampling operation from the definition name.

    Parameters
    ----------
    defn : Definition
        Operation definition.

    Returns
    -------
    str
        Sampling type: "top_k_top_p", "top_k", "top_p", or "basic".
    """
    name = defn.name.lower()
    if "top_k_top_p" in name:
        return "top_k_top_p"
    elif "top_k" in name:
        return "top_k"
    elif "top_p" in name:
        return "top_p"
    else:
        return "basic"  # vanilla sampling


def _validate_sampling_tokens(
    samples: torch.Tensor, probs: torch.Tensor, sampling_type: str, params: Dict[str, Any]
) -> bool:
    """Validate that sampled tokens conform to sampling constraints.

    Parameters
    ----------
    samples : torch.Tensor
        Sampled token indices.
    probs : torch.Tensor
        Probability distribution used for sampling.
    sampling_type : str
        Type of sampling: "top_k", "top_p", "top_k_top_p", or "basic".
    params : Dict[str, Any]
        Sampling parameters (top_k, top_p values).

    Returns
    -------
    bool
        True if samples are valid, False otherwise.
    """
    batch_size, vocab_size = probs.shape
    device = probs.device

    for i in range(batch_size):
        prob_row = probs[i]
        sample = samples[i].item()

        if sampling_type == "top_k":
            if "top_k" not in params:
                return True
            k = (
                int(params["top_k"][i].item())
                if params["top_k"].dim() > 0
                else int(params["top_k"].item())
            )
            if 0 < k < vocab_size:
                sorted_prob_desc, _ = torch.sort(prob_row, descending=True)
                pivot = sorted_prob_desc[k - 1]
                mask_top_k = (prob_row >= pivot).int()
                if mask_top_k[sample] != 1:
                    return False
        elif sampling_type == "top_p":
            if "top_p" not in params:
                return True
            p = (
                float(params["top_p"][i].item())
                if params["top_p"].dim() > 0
                else float(params["top_p"].item())
            )
            if 0 < p < 1:
                eps = 1e-4  # numerical stability
                sorted_probs, indices = torch.sort(prob_row, descending=False)
                cdf = torch.cumsum(sorted_probs, dim=0)
                valid_mask = cdf > (1 - p) - eps
                valid_indices = indices[valid_mask]

                if sample not in valid_indices:
                    return False

        elif sampling_type == "top_k_top_p":
            if "top_k" not in params or "top_p" not in params:
                return True
            k = (
                int(params["top_k"][i].item())
                if params["top_k"].dim() > 0
                else int(params["top_k"].item())
            )
            p = (
                float(params["top_p"][i].item())
                if params["top_p"].dim() > 0
                else float(params["top_p"].item())
            )

            if 0 < k < vocab_size:
                sorted_prob_desc, _ = torch.sort(prob_row, descending=True)
                pivot = sorted_prob_desc[k - 1]
                mask_top_k = (prob_row >= pivot).int()
            else:
                mask_top_k = torch.ones(vocab_size, dtype=torch.int32, device=device)

            if 0 < p < 1:
                eps = 1e-4
                sorted_probs_asc, indices = torch.sort(prob_row, descending=False)
                cdf = torch.cumsum(sorted_probs_asc, dim=0)
                mask_top_p = torch.zeros(vocab_size, dtype=torch.int32, device=device)
                valid_p_mask = cdf > (1 - p) - eps
                mask_top_p[indices[valid_p_mask]] = 1
            else:
                mask_top_p = torch.ones(vocab_size, dtype=torch.int32, device=device)

            joint_mask = torch.minimum(mask_top_k, mask_top_p)

            if joint_mask[sample] != 1:
                return False

    return True


def _validate_sampling_correctness(
    runnable_sol: Runnable,
    inputs: List[Dict[str, Any]],
    ref_outputs_bl: List[Dict[str, torch.Tensor]],
    cfg: BenchmarkConfig,
    device: str,
    log_path: str,
    defn: Definition,
) -> Tuple[Optional[Evaluation], float, float, bool]:
    """Validate correctness for sampling operations.

    Parameters
    ----------
    runnable_sol : Runnable
        The compiled solution runnable.
    inputs : List[Dict[str, Any]]
        List of input dictionaries.
    ref_outputs_bl : List[Dict[str, torch.Tensor]]
        List of reference output dictionaries.
    cfg : BenchmarkConfig
        Benchmark configuration.
    device : str
        Device string (e.g. "cuda:0").
    log_path : str
        Path to log file.
    defn : Definition
        Operation definition.

    Returns
    -------
    Tuple[Optional[Evaluation], float, float, bool]
        A tuple containing:
        - Optional evaluation object if error occurred, None otherwise
        - Maximum absolute error
        - Maximum relative error
        - Whether numerical correctness check failed
    """
    sampling_type = _detect_sampling_type(defn)
    ref_freq = ref_outputs_bl[0]["frequency_distribution"]
    vocab_size = ref_freq.shape[0]

    inp = inputs[0]
    params = {k: inp[k] for k in ["top_k", "top_p"] if k in inp}

    output_names = list(defn.outputs.keys())
    output_dtypes = {k: torch_dtype_from_def(v.dtype) for k, v in defn.outputs.items()}

    # Validate correct sampling token set
    for trial_idx in range(cfg.sampling_validation_trials):
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
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
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
        if not _validate_sampling_tokens(samples, probs, sampling_type, params):
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

    numerical_incorrect = similarity < cfg.sampling_similarity_threshold

    return None, max_abs, max_rel, numerical_incorrect


def _validate_default_correctness(
    runnable_sol: Runnable,
    inputs: List[Dict[str, Any]],
    ref_outputs_bl: List[Dict[str, torch.Tensor]],
    cfg: BenchmarkConfig,
    device: str,
    log_path: str,
    defn: Definition,
) -> Tuple[Optional[Evaluation], float, float, bool, Optional[float]]:
    """Validate correctness for non-sampling operations.

    Parameters
    ----------
    runnable_sol : Runnable
        The compiled solution runnable.
    inputs : List[Dict[str, Any]]
        List of input dictionaries.
    ref_outputs_bl : List[Dict[str, torch.Tensor]]
        List of reference output dictionaries.
    cfg : BenchmarkConfig
        Benchmark configuration.
    device : str
        Device string (e.g. "cuda:0").
    log_path : str
        Path to log file.
    defn : Definition
        Operation definition.

    Returns
    -------
    Tuple[Optional[Evaluation], float, float, bool, Optional[float]]
        A tuple containing:
        - Optional evaluation object if error occurred, None otherwise
        - Maximum absolute error
        - Maximum relative error
        - Whether numerical correctness check failed
        - Optional matched ratio used for comparison
    """
    output_names = list(ref_outputs_bl[0].keys())
    output_dtypes = {k: v.dtype for k, v in ref_outputs_bl[0].items()}

    max_abs = 0.0
    max_rel = 0.0
    numerical_incorrect = False
    matched_ratio_used = None

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
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )
        ref_t = ref_outputs_bl[t]

        for k in ref_t.keys():
            # Shape validation
            if k not in out_t:
                return (
                    make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_file=log_path
                    ),
                    0.0,
                    0.0,
                    False,
                    None,
                )

            if tuple(out_t[k].shape) != tuple(ref_t[k].shape):
                return (
                    make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_file=log_path
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
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_file=log_path
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
                    max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
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
            abs_err, rel_err, exceeds_tol, ratio_used = compute_error_stats(
                out_t[k], ref_t[k], cfg, defn
            )

            if exceeds_tol:
                numerical_incorrect = True

            if ratio_used is not None:
                matched_ratio_used = ratio_used

            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

    return None, max_abs, max_rel, numerical_incorrect, matched_ratio_used


class SolutionEvaluator:
    @staticmethod
    def evaluate(
        runnable_sol: Runnable,
        inputs: List[Dict[str, Any]],
        ref_outputs: List[Dict[str, torch.Tensor]],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        device: str,
        log_path: str,
        defn: Definition,
    ) -> Evaluation:
        """Evaluate a solution against reference outputs.

        Parameters
        ----------
        runnable_sol : Runnable
            The compiled solution runnable.
        inputs : List[Dict[str, Any]]
            List of input dictionaries.
        ref_outputs : List[Dict[str, torch.Tensor]]
            List of reference output dictionaries.
        ref_mean_latency_ms : float
            Reference implementation mean latency.
        cfg : BenchmarkConfig
            Benchmark configuration.
        device : str
            Device string (e.g. "cuda:0").
        log_path : str
            Path to log file.
        defn : Definition
            Operation definition.

        Returns
        -------
        Evaluation
            Evaluation object with status, correctness, and performance metrics.
        """
        if not ref_outputs:
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=device,
                log_file=log_path,
                error="No reference outputs provided",
            )

        is_sampling = is_sampling_operation(defn)

        if is_sampling:
            eval_result, max_abs, max_rel, numerical_incorrect = _validate_sampling_correctness(
                runnable_sol, inputs, ref_outputs, cfg, device, log_path, defn
            )
            matched_ratio_used = None
        else:
            eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio_used = (
                _validate_default_correctness(
                    runnable_sol, inputs, ref_outputs, cfg, device, log_path, defn
                )
            )

        if eval_result is not None:
            return eval_result

        correctness = Correctness(
            max_relative_error=max_rel, max_absolute_error=max_abs, matched_ratio=matched_ratio_used
        )

        if numerical_incorrect:
            return make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                log_file=log_path,
                correctness=correctness,
                device=device,
            )

        # Measure performance
        try:
            soln_lats: List[float] = []
            for inp in inputs:
                lat_ms = time_runnable(runnable_sol, inp, cfg.warmup_runs, cfg.iterations, device)
                soln_lats.append(lat_ms)

            if not soln_lats:
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=device,
                    log_file=log_path,
                    error="Failed to collect solution latencies",
                )

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

        except Exception as e:
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=device,
                log_file=log_path,
                error=f"Performance measurement failed: {str(e)}",
            )
