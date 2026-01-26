"""MoE (Mixture of Experts) adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _infer_moe_def_name(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    implementation: str = "batched",
) -> str:
    """Infer definition name for MoE operation."""
    return f"moe_{implementation}_e{num_experts}_h{hidden_dim}_i{intermediate_dim}_topk{top_k}"


class MoEAdapter:
    """Adapter for Mixture of Experts operations.

    Traces MoE forward functions used in models like Qwen3-MoE, GPT-OSS:
    - transformers.integrations.moe.batched_mm_experts_forward
    - transformers.integrations.moe.grouped_mm_experts_forward
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="transformers.integrations.moe.batched_mm_experts_forward",
                kind="function",
                name="batched_mm_experts_forward",
                ctx_key="moe_batched",
            ),
            PatchSpec(
                path="transformers.integrations.moe.grouped_mm_experts_forward",
                kind="function",
                name="grouped_mm_experts_forward",
                ctx_key="moe_grouped",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces MoE calls."""

        implementation = "batched" if "batched" in spec.name else "grouped"

        def wrapper(
            self_module: torch.nn.Module,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            # Validate inputs
            if not isinstance(hidden_states, torch.Tensor):
                return orig(self_module, hidden_states, top_k_index, top_k_weights)

            # Only trace on CUDA with supported dtypes
            if not hidden_states.is_cuda:
                return orig(self_module, hidden_states, top_k_index, top_k_weights)

            if hidden_states.dtype not in (torch.float16, torch.bfloat16):
                return orig(self_module, hidden_states, top_k_index, top_k_weights)

            # Extract dimensions from the module
            num_experts = self_module.gate_up_proj.size(0)
            hidden_dim = hidden_states.size(-1)

            # Infer intermediate dim from gate_up_proj shape
            # gate_up_proj shape depends on is_transposed:
            # - is_transposed=False: (num_experts, 2 * intermediate_dim, hidden_dim)
            # - is_transposed=True: (num_experts, hidden_dim, 2 * intermediate_dim)
            is_transposed = getattr(self_module, "is_transposed", False)
            if is_transposed:
                intermediate_dim = self_module.gate_up_proj.size(2) // 2
            else:
                intermediate_dim = self_module.gate_up_proj.size(1) // 2

            top_k = top_k_index.size(-1)

            def_name = _infer_moe_def_name(
                num_experts, hidden_dim, intermediate_dim, top_k, implementation
            )

            rk: Dict[str, Any] = {
                "hidden_states": hidden_states,
                "top_k_index": top_k_index,
                "top_k_weights": top_k_weights,
                "gate_up_proj": self_module.gate_up_proj,
                "down_proj": self_module.down_proj,
            }

            def _fallback(**_rk):
                return orig(self_module, hidden_states, top_k_index, top_k_weights)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            return ret

        return wrapper
