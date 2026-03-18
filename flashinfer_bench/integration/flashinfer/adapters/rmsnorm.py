from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.apply.runtime import ApplyRuntime
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder


def _def_name_resolver(weight):
    return f"fused_add_rmsnorm_h{weight.shape[0]}"


def _active_flashinfer_baseline_solution() -> bool:
    active_solution = ApplyRuntime.current_solution_name()
    return bool(active_solution and active_solution.startswith("flashinfer_wrapper_"))


class RMSNormAdapter:
    """Adapter for fused add + RMSNorm entrypoints used by FlashInfer and SGLang."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.norm.fused_add_rmsnorm",
                kind="function",
                name="fused_add_rmsnorm",
                ctx_key="rmsnorm",
            ),
            PatchSpec(
                path="flashinfer.fused_add_rmsnorm",
                kind="function",
                name="fused_add_rmsnorm",
                ctx_key="rmsnorm",
            ),
            PatchSpec(
                path="sgl_kernel.fused_add_rmsnorm",
                kind="function",
                name="fused_add_rmsnorm",
                ctx_key="rmsnorm",
            ),
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        binder = ArgBinder.from_callable(orig)

        def wrapper(*args, **kwargs):
            if spec.path.startswith("flashinfer") and _active_flashinfer_baseline_solution():
                # FlashInfer baseline wrappers call back into flashinfer.norm.fused_add_rmsnorm.
                # Bypass apply here so baseline_only truly executes the native FlashInfer kernel.
                return orig(*args, **kwargs)

            bound = binder.bind(args, kwargs)
            input_tensor: torch.Tensor = bound["input"]
            residual: torch.Tensor = bound["residual"]
            weight: torch.Tensor = bound["weight"]
            eps = float(bound.get("eps", 1e-6))

            # Compatibility checks
            if (
                input_tensor.dtype != torch.bfloat16
                or residual.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16
            ):
                return orig(*args, **kwargs)
            if input_tensor.ndim != 2 or residual.ndim != 2 or weight.ndim != 1:
                return orig(*args, **kwargs)
            if input_tensor.shape != residual.shape or input_tensor.shape[1] != weight.shape[0]:
                return orig(*args, **kwargs)
            # The current trace definitions fix epsilon at 1e-5.
            if not math.isclose(eps, 1e-5, rel_tol=0.0, abs_tol=1e-12):
                return orig(*args, **kwargs)

            def_name = _def_name_resolver(weight)
            rk: Dict[str, Any] = {
                "hidden_states": input_tensor,
                "residual": residual,
                "weight": weight,
            }

            def _fallback(**_rk):
                return orig(*args, **kwargs)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)
            if ret is None:
                return None
            if ret is input_tensor:
                # Some baseline wrappers already implement the in-place contract and return the
                # mutated input tensor for convenience. Avoid re-applying the semantics here.
                return None

            # Match FlashInfer/SGLang in-place semantics:
            # 1. residual += input
            # 2. input = rmsnorm(residual) * weight
            residual.add_(input_tensor)
            input_tensor.copy_(ret)
            return None

        return wrapper
