from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply  # , get_apply_runtime
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder

# from flashinfer_bench.tracing import get_tracing_runtime


def _def_name_resolver(weight):
    return f"fused_add_rmsnorm_h{weight.shape[0]}"


class RMSNormAdapter:
    """Adapter for flashinfer.norm.fused_add_rmsnorm."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.norm.fused_add_rmsnorm",
                kind="function",
                name="fused_add_rmsnorm",
                ctx_key="fused_add_rmsnorm",
            )
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        binder = ArgBinder.from_callable(orig)

        def wrapper(*args, **kwargs):
            # rt_trace = get_tracing_runtime()
            # rt_apply = get_apply_runtime()
            # if rt_trace is None and rt_apply is None:
            #     return orig(*args, **kwargs)
            print("args:", args)
            print("kwargs:", kwargs)

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

            # if rt_trace is not None:
            #     rt_trace.collect(def_name, rk)
            #     return orig(*args, **kwargs)
            # elif rt_apply is not None:
            #     # Fallback if no definition found
            #     if def_name not in rt_apply._trace_set.definitions:
            #         return orig(*args, **kwargs)

            #     def _fb(**_rk):
            #         return orig(*args, **kwargs)

            #     ret = apply(def_name, runtime_kwargs=rk, fallback=_fb)
            #     return ret
            # else:
            #     return orig(*args, **kwargs)
            ret = apply(def_name, runtime_kwargs=rk, fallback=_fb)
            return ret

        return wrapper
