"""Evaluator for DSA sparse attention kernels with optional lse output."""

from __future__ import annotations

import traceback
from typing import Any, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import compute_error_stats, make_eval
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import Correctness, Definition, Evaluation, EvaluationStatus

from .default import DefaultEvaluator
from .utils import allocate_outputs

_DSA_SPARSE_ATTENTION_DEFS = {
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1",
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
}


class DsaSparseAttentionEvaluator(DefaultEvaluator):
    @override
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return definition.name in _DSA_SPARSE_ATTENTION_DEFS

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
                    out = _flexible_normalize(result, device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            ref_out = ref_outputs[trial]
            if len(ref_out) != 2:
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=device,
                    log_path=log_path,
                    extra_msg="DSA sparse attention reference must return exactly 2 outputs.",
                )

            if len(out) not in (1, 2):
                return None, make_eval(
                    status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                )

            for sol_tensor, ref_tensor in zip(out, ref_out[: len(out)], strict=True):
                if tuple(sol_tensor.shape) != tuple(ref_tensor.shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                if sol_tensor.dtype != ref_tensor.dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                    )

                non_finite_err_val = None
                if torch.isinf(sol_tensor).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(sol_tensor).any().item():
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

                abs_err, rel_err, exceeds_tol, _ = compute_error_stats(sol_tensor, ref_tensor, cfg)

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


def _flexible_normalize(result: Any, device: str) -> List[torch.Tensor]:
    """Convert solution result to tensor list, allowing fewer outputs than definition expects."""

    def to_tensor(v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if str(v.device) != device else v
        return torch.tensor(v, device=device)

    if result is None:
        return []

    if isinstance(result, (tuple, list)):
        return [to_tensor(v) for v in result]
    return [to_tensor(result)]
