import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.benchmark_config import BenchmarkConfig
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
from flashinfer_bench.bench.runner.runner import BaselineHandle, DeviceBaseline
from flashinfer_bench.bench.runner.runner_utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    make_eval,
    normalize_outputs,
)
from flashinfer_bench.bench.utils import time_runnable
from flashinfer_bench.compile.builder_registry import get_builder_registry
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import Correctness, Evaluation, EvaluationStatus, Workload
from flashinfer_bench.utils import dtype_str_to_python_dtype


class SamplingEvaluator(DefaultEvaluator):

    @override
    @classmethod
    def can_evaluate(cls, defn: Definition) -> bool:
        return is_sampling_op(defn)

    @override
    @classmethod
    def build_baseline(
        cls,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        traceset_root: Optional[Path] = None,
    ) -> DeviceBaseline:
        ref_runnable = get_builder_registry().build_reference(defn)
        loaded_stensors = (
            load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: List[Dict[str, Any]] = []
        outputs: List[Dict[str, torch.Tensor]] = []

        inp = gen_inputs(defn, workload, device=device, stensors=loaded_stensors)
        if "probs" in inp:
            inp["probs"] = torch.softmax(
                inp["probs"], dim=-1
            )  # convert logits to probs for sampling
        inputs.append(inp)

        freq_dist = _compute_frequency_distribution(
            ref_runnable, inp, device, defn, num_trials=50000
        )
        outputs.append({"frequency_distribution": freq_dist})

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
        ref_freq = ref_outputs[0]["frequency_distribution"]
        vocab_size = ref_freq.shape[0]

        inp = inputs[0]
        params = {k: inp[k] for k in ["top_k", "top_p"] if k in inp}

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: dtype_str_to_python_dtype(v.dtype) for k, v in defn.outputs.items()}

        # Validate correct sampling token set
        for _ in range(cfg.sampling_validation_trials):
            try:
                with torch.no_grad():
                    out = sol_runnable(**inp)
                torch.cuda.synchronize(device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            out_normalized = normalize_outputs(
                out, device=device, output_names=output_names, output_dtypes=output_dtypes
            )
            samples = out_normalized["samples"]

            # Check vocabulary range
            if (samples < 0).any() or (samples >= vocab_size).any():
                invalid_samples = samples[(samples < 0) | (samples >= vocab_size)]
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                message = (
                    f"Samples {invalid_samples.tolist()} out of vocabulary range [0, {vocab_size})"
                )
                print(message, file=sys.stderr)
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                )

            # Validate thresholding
            thresholding_method = _detect_thresholding_method(defn)
            probs = inp["probs"]
            if not _check_thresholding(samples, probs, thresholding_method, params):
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                message = (
                    f"Samples {samples.tolist()} does not meet {thresholding_method} thresholding"
                )
                print(message, file=sys.stderr)
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                )

        try:
            sol_freq = _compute_frequency_distribution(
                sol_runnable, inp, device, defn, num_trials=50000
            )
            torch.cuda.synchronize(device)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        # total variation distance
        tvd = 0.5 * torch.sum(torch.abs(sol_freq - ref_freq)).item()
        max_abs, max_rel, _, _ = compute_error_stats(sol_freq, ref_freq, cfg)

        numerical_incorrect = tvd > cfg.sampling_tvd_threshold
        correctness = Correctness(
            max_relative_error=max_rel, max_absolute_error=max_abs, extra={"tvd": tvd}
        )
        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None


def is_sampling_op(defn: Definition) -> bool:
    return getattr(defn, "op_type", None) == "sampling"


def _detect_thresholding_method(defn: Definition) -> str:
    name = defn.name.lower()
    if "top_k_top_p" in name:
        return "top_k_top_p"
    elif "top_k" in name:
        return "top_k"
    elif "top_p" in name:
        return "top_p"
    else:
        return "none"  # no thresholding


def _compute_frequency_distribution(
    runnable: Runnable,
    inputs: Dict[str, Any],
    device: str,
    defn: Definition,
    num_trials: int = 10000,
) -> torch.Tensor:
    batch_size = inputs["probs"].shape[0] if inputs["probs"].dim() > 1 else 1
    vocab_size = inputs["probs"].shape[-1]
    counter = torch.zeros(vocab_size, dtype=torch.int64, device=torch.device(device))

    trials_needed = (num_trials + batch_size - 1) // batch_size
    total_samples_collected = 0

    for _ in range(trials_needed):
        with torch.no_grad():
            out = runnable(**inputs)

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: dtype_str_to_python_dtype(v.dtype) for k, v in defn.outputs.items()}

        out_normalized = normalize_outputs(
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )

        samples = out_normalized["samples"]

        if samples.dim() == 0:
            sample_idx = samples.item()
            counter[sample_idx] += 1
            total_samples_collected += 1
        else:  # Batch of samples
            for i in range(samples.numel()):
                sample_idx = samples.flatten()[i].item()
                counter[sample_idx] += 1
                total_samples_collected += 1

    frequency = counter.float() / total_samples_collected
    return frequency


def _check_thresholding(
    samples: torch.Tensor, probs: torch.Tensor, method: str, params: Dict[str, Any]
) -> bool:
    """Check if samples conform to the specified thresholding method.

    Parameters
    ----------
    samples : torch.Tensor
        Sampled token indices.
    probs : torch.Tensor
        Probability distribution used for sampling.
    method : str
        Thresholding method: "top_k", "top_p", "top_k_top_p", or "none".
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

        if method == "top_k":
            if "top_k" not in params:
                raise ValueError("top_k parameter is required for top_k thresholding but not found")
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

        elif method == "top_p":
            if "top_p" not in params:
                raise ValueError("top_p parameter is required for top_p thresholding but not found")
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

        elif method == "top_k_top_p":
            if "top_k" not in params or "top_p" not in params:
                raise ValueError(
                    "top_k and top_p parameters are both required for top_k_top_p thresholding but not found"
                )
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
