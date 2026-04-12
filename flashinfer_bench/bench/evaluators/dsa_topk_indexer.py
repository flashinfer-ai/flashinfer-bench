"""Evaluator for DSA top-k indexer kernels using sorted-score comparison."""

from __future__ import annotations

import sys
import traceback
import uuid as uuid_mod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.runner import BaselineHandle, DeviceBaseline
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors, make_eval
from flashinfer_bench.compile import BuilderRegistry, Runnable
from flashinfer_bench.data import Correctness, Definition, Evaluation, EvaluationStatus, Workload

from .default import DefaultEvaluator
from .utils import allocate_outputs, normalize_result


def _dequant_all_pages(k_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize all pages from deep_gemm FP8 packed layout.

    Input:  [num_pages, page_size, 1, head_dim+4] int8
    Output: [num_pages, page_size, head_dim] float32

    The packed layout stores fp8 data first, then per-token float32 scales,
    at the page level (not per-token).
    """
    k_uint8 = k_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_with_scale = k_uint8.shape
    head_dim = head_dim_with_scale - 4
    flat = k_uint8.view(num_pages, page_size * head_dim_with_scale)
    fp8_data = (
        flat[:, : page_size * head_dim]
        .contiguous()
        .view(num_pages, page_size, head_dim)
        .view(torch.float8_e4m3fn)
        .float()
    )
    scale = (
        flat[:, page_size * head_dim :]
        .contiguous()
        .view(num_pages, page_size, 4)
        .view(torch.float32)
    )
    return fp8_data * scale


def _compute_scores_at_indices(
    indices: torch.Tensor,
    k_all: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Compute weighted ReLU scores at given global token indices.

    All operations are batched — no Python loop over the batch dimension.

    Input:
      indices:  [B, N] int32, global page indices, -1 = padding
      k_all:    [num_pages, page_size, D] float32
      q_fp8:    [B, H, D] float8_e4m3fn
      weights:  [B, H] float32
    Output:
      scores:   [B, N] float32
    """
    valid_mask = indices >= 0
    safe = indices.clamp(min=0).long()
    page_id = safe // page_size
    offset = safe % page_size
    k_gathered = k_all[page_id, offset]  # [B, N, D]
    q_f32 = q_fp8.float()  # [B, H, D]
    per_head = torch.bmm(q_f32, k_gathered.transpose(1, 2))  # [B, H, N]
    per_head = torch.relu(per_head)
    scores = (per_head * weights.unsqueeze(2)).sum(dim=1)  # [B, N]
    scores[~valid_mask] = float("-inf")
    return scores


def _validate_indices(
    indices: torch.Tensor,
    num_pages: int,
    page_size: int,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized index validity check: reachability and duplicates.

    Uses [B, K, M] broadcasting instead of [B, num_pages*page_size] dense mask.

    Input:
      indices: [B, K] int32
      seq_lens: [B] int32
      block_table: [B, M] int32  (M = max_num_pages per sequence)
    Output: out_of_range [B] bool, has_dup [B] bool
    """
    valid_mask = indices >= 0
    safe = indices.clamp(min=0).long()

    index_page_id = safe // page_size  # [B, K]
    index_offset = safe % page_size  # [B, K]

    # Check page membership: [B, K, 1] == [B, 1, M] -> [B, K, M]
    page_match = index_page_id.unsqueeze(2) == block_table.long().unsqueeze(1)
    matched_slot = page_match.any(dim=2)  # [B, K] — page found in block_table

    # Which slot matched (argmax gives first match), used for offset range check
    slot_idx = page_match.long().argmax(dim=2)  # [B, K]
    token_position = slot_idx * page_size + index_offset  # [B, K] position within seq
    within_seq = token_position < seq_lens.unsqueeze(1)  # [B, K]

    reachable = matched_slot & within_seq
    out_of_range = (valid_mask & ~reachable).any(dim=1)

    # Duplicate check
    sorted_idx = indices.sort(dim=1).values
    consecutive_eq = sorted_idx[:, 1:] == sorted_idx[:, :-1]
    valid_pair = sorted_idx[:, 1:] >= 0
    has_dup = (consecutive_eq & valid_pair).any(dim=1)
    return out_of_range, has_dup


def _compute_sorted_score_error_stats(
    output: torch.Tensor, reference: torch.Tensor, cfg: BenchmarkConfig
) -> Tuple[float, float, bool, float]:
    """Compute score error stats while ignoring shared padding sentinels.

    DSA top-k outputs use `-1` padding. After score reconstruction and sorting,
    both reference and solution contain trailing `-inf` sentinels for those
    padded slots. Those positions should count as matched rather than producing
    `nan` through `-inf - -inf`.
    """
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    shared_nonfinite_mask = (torch.isnan(x) & torch.isnan(y)) | (
        torch.isinf(x) & torch.isinf(y) & (torch.signbit(x) == torch.signbit(y))
    )
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    invalid_mismatch_mask = (~shared_nonfinite_mask) & (~finite_mask)

    eps = 1e-8
    abs_error = torch.zeros_like(x)
    rel_error = torch.zeros_like(x)
    abs_error[finite_mask] = torch.abs(x[finite_mask] - y[finite_mask])
    rel_error[finite_mask] = abs_error[finite_mask] / (torch.abs(y[finite_mask]) + eps)

    total_elements = x.numel()
    if total_elements == 0:
        return 0.0, 0.0, False, 1.0

    required_matched_ratio = (
        cfg.required_matched_ratio if cfg.required_matched_ratio is not None else 1.0
    )
    exceeds_tol_mask = invalid_mismatch_mask | (
        finite_mask & (abs_error > cfg.atol) & (rel_error > cfg.rtol)
    )
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))
    exceeds_tol = matched_ratio < required_matched_ratio

    max_abs = float(abs_error.max().item()) if finite_mask.any().item() else 0.0
    max_rel = float(rel_error.max().item()) if finite_mask.any().item() else 0.0
    if invalid_mismatch_mask.any().item():
        max_abs = float("inf")
        max_rel = float("inf")
    return max_abs, max_rel, exceeds_tol, matched_ratio


def _pack_fp8_k_cache(
    key_cache_bfloat16: torch.Tensor, page_size: int, head_dim: int
) -> torch.Tensor:
    """Pack bfloat16 key cache into deep_gemm FP8 format.

    Input:  [num_pages, page_size, 1, head_dim] bfloat16
    Output: [num_pages, page_size, 1, head_dim+4] int8
    """
    absolute_max = key_cache_bfloat16.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = absolute_max / 448.0
    fp8_data = (key_cache_bfloat16 * (1.0 / scale)).to(torch.float8_e4m3fn)

    num_pages = key_cache_bfloat16.shape[0]
    packed = torch.empty(
        num_pages, page_size * (head_dim + 4), device=key_cache_bfloat16.device, dtype=torch.uint8
    )
    packed[:, : page_size * head_dim] = fp8_data.view(num_pages, page_size * head_dim).view(
        torch.uint8
    )
    packed[:, page_size * head_dim :] = scale.view(num_pages, page_size).view(torch.uint8)
    return packed.view(num_pages, page_size, 1, head_dim + 4).view(torch.int8)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


class DsaTopkIndexerEvaluator(DefaultEvaluator):
    @override
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return definition.name.startswith("dsa_topk_indexer")

    @override
    @classmethod
    def build_baseline(
        cls,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        trace_set_root: Optional[Path] = None,
    ) -> DeviceBaseline:
        ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
        loaded_safe_tensors = (
            load_safetensors(definition, workload, trace_set_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        page_size = definition.axes["page_size"].value
        head_dim = definition.axes["index_head_dim"].value
        input_names = list(definition.inputs.keys())
        k_cache_idx = input_names.index("k_index_cache_fp8")
        k_cache_is_random = (
            "k_index_cache_fp8" in workload.inputs
            and workload.inputs["k_index_cache_fp8"].type == "random"
        )

        inputs: List[List[Any]] = []
        outputs: List[List[torch.Tensor]] = []
        dev = torch.device(device)

        for _ in range(cfg.num_trials):
            inp = gen_inputs(definition, workload, device=device, safe_tensors=loaded_safe_tensors)

            if k_cache_is_random:
                num_pages = inp[k_cache_idx].shape[0]
                k_bf16 = torch.randn(
                    num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=dev
                )
                inp[k_cache_idx] = _pack_fp8_k_cache(k_bf16, page_size, head_dim)

            inputs.append(inp)
            with torch.no_grad():
                result = ref_runnable(*inp)
            torch.cuda.synchronize(device)
            outputs.append(normalize_result(definition, result, device))

        if cfg.profile_baseline:
            latencies: List[float] = []
            for inp in inputs:
                ms = time_runnable(ref_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
                latencies.append(ms)
            mean_latency_ms = sum(latencies) / float(len(latencies))
        else:
            mean_latency_ms = 0.0

        handle = BaselineHandle(uuid_mod.uuid4().hex)
        return DeviceBaseline(
            handle=handle,
            definition=definition,
            device=device,
            inputs=inputs,
            outputs=outputs,
            mean_latency_ms=mean_latency_ms,
        )

    @override
    @classmethod
    def check_correctness(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_outputs: List[List[torch.Tensor]],
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        is_dps = sol_runnable.metadata.destination_passing_style

        page_size = definition.axes["page_size"].value
        topk = definition.axes["topk"].value

        for trial, inp in enumerate(inputs):
            try:
                if is_dps:
                    out = allocate_outputs(definition, inp, device)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    torch.cuda.synchronize(device)
                else:
                    with torch.no_grad():
                        result = sol_runnable(*inp)
                    torch.cuda.synchronize(device)
                    out = normalize_result(definition, result, device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            sol_indices = out[0]
            ref_indices = ref_outputs[trial][0]

            if tuple(sol_indices.shape) != tuple(ref_indices.shape):
                return None, make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                )
            if sol_indices.dtype != ref_indices.dtype:
                return None, make_eval(
                    status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                )

            input_names = list(definition.inputs.keys())
            inputs_by_name = dict(zip(input_names, inp))
            q_fp8 = inputs_by_name["q_index_fp8"]
            k_cache_fp8 = inputs_by_name["k_index_cache_fp8"]
            weights = inputs_by_name["weights"]
            seq_lens = inputs_by_name["seq_lens"]
            block_table = inputs_by_name["block_table"]
            num_pages = k_cache_fp8.shape[0]

            out_of_range, has_dup = _validate_indices(
                sol_indices, num_pages, page_size, seq_lens, block_table
            )

            if out_of_range.any().item():
                bad_batches = out_of_range.nonzero(as_tuple=True)[0].tolist()
                msg = f"out-of-range indices in batch {bad_batches}"
                _log(f"ERROR: {msg}")
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                    extra_msg=msg,
                )

            if has_dup.any().item():
                bad_batches = has_dup.nonzero(as_tuple=True)[0].tolist()
                msg = f"duplicate indices in batch {bad_batches}"
                _log(f"ERROR: {msg}")
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                    extra_msg=msg,
                )

            k_all = _dequant_all_pages(k_cache_fp8)
            combined = torch.cat([ref_indices, sol_indices], dim=1)
            all_scores = _compute_scores_at_indices(combined, k_all, q_fp8, weights, page_size)
            ref_scores, sol_scores = all_scores.split(topk, dim=1)

            ref_sorted = ref_scores.sort(dim=1, descending=True).values
            sol_sorted = sol_scores.sort(dim=1, descending=True).values
            abs_err, rel_err, exceeds_tol, matched_ratio = _compute_sorted_score_error_stats(
                sol_sorted, ref_sorted, cfg
            )

            _log(
                f"trial {trial}: max_abs={abs_err:.6f} max_rel={rel_err:.6f} "
                f"matched_ratio={matched_ratio:.4f}"
            )

            if exceeds_tol:
                msg = (
                    f"score mismatch: max_abs={abs_err:.6f} "
                    f"max_rel={rel_err:.6f} matched_ratio={matched_ratio:.4f}"
                )
                _log(f"ERROR: {msg}")
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
