from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

import flashinfer_bench.apply as apply
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder


def _def_name_resolver(weight):
    return f"fused_add_rmsnorm_h{weight.shape[0]}"


class RMSNormAdapter:
    """Adapter for flashinfer.norm.fused_add_rmsnorm."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.norm.fused_add_rmsnorm",
                kind="method",
                name="fused_add_rmsnorm",
                ctx_key="rmsnorm",
            )
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        binder = ArgBinder.from_callable(orig)

        def wrapper(*args, **kwargs):
            bound = binder.bind(args, kwargs)
            input_tensor: torch.Tensor = bound["input"]
            residual: torch.Tensor = bound["residual"]
            weight: torch.Tensor = bound["weight"]

            # Compatibility checks
            if (
                input_tensor.dtype != torch.bfloat16
                or residual.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16
            ):
                logger.warning("RMSNormAdapter: dtype not bfloat16, skipping")
                return orig(*args, **kwargs)
            if input_tensor.shape != residual.shape or input_tensor.shape[1] != weight.shape[0]:
                logger.warning("RMSNormAdapter: shape mismatch, skipping")
                return orig(*args, **kwargs)

            def_name = _def_name_resolver(weight)
            rk: Dict[str, Any] = {
                "hidden_states": input_tensor,
                "residual": residual,
                "weight": weight,
            }

            def _fb(**_rk):
                return orig(*args, **kwargs)

            ret = apply(def_name, runtime_kwargs=rk, fallback=_fb)
            return ret

        return wrapper
