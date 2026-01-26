"""MoE (Mixture of Experts) adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.transformers.common import (
    SUPPORTED_ACTIVATION_DTYPES,
    infer_moe_def_name as _infer_moe_def_name,
)


class MoEAdapter:
    """Adapter for Mixture of Experts operations.

    Traces MoE forward functions used in models like Qwen3-MoE, GPT-OSS:
    - transformers.integrations.moe.batched_mm_experts_forward
    - transformers.integrations.moe.grouped_mm_experts_forward
    - Model-specific MoE implementations (GptOssExperts, Qwen3MoeSparseMoeBlock, etc.)
    """

    def targets(self) -> List[PatchSpec]:
        specs = [
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
        
        # Add model-specific MoE MLP forward methods
        moe_mlp_paths = [
            "transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP",
            "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
            "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
            "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
        ]
        
        for path in moe_mlp_paths:
            model_name = path.split(".")[-1]
            specs.append(
                PatchSpec(
                    path=f"{path}.forward",
                    kind="method",
                    name="forward",
                    ctx_key=f"moe_{model_name}",
                )
            )
        
        return specs

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces MoE calls."""
        
        if spec.kind == "function":
            return self._make_experts_forward_wrapper(spec, orig)
        else:
            return self._make_moe_mlp_wrapper(spec, orig)
    
    def _make_experts_forward_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Wrapper for batched/grouped experts forward functions."""
        
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

            if hidden_states.dtype not in SUPPORTED_ACTIVATION_DTYPES:
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
                num_experts, hidden_dim, intermediate_dim, top_k, implementation, hidden_states.dtype
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
    
    def _make_moe_mlp_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Wrapper for MoE MLP forward methods (GptOssMLP, Qwen3MoeSparseMoeBlock, etc.)."""

        def wrapper(
            self_module: torch.nn.Module,
            hidden_states: torch.Tensor,
        ) -> Any:
            # Validate inputs
            if not isinstance(hidden_states, torch.Tensor):
                return orig(self_module, hidden_states)

            # Only trace on CUDA with supported dtypes
            if not hidden_states.is_cuda:
                return orig(self_module, hidden_states)

            if hidden_states.dtype not in SUPPORTED_ACTIVATION_DTYPES:
                return orig(self_module, hidden_states)

            # Try to extract MoE config from the module
            # GptOssMLP has: router, experts
            # Qwen3MoeSparseMoeBlock has: gate, experts
            experts = getattr(self_module, "experts", None)
            if experts is None:
                return orig(self_module, hidden_states)
            
            # Get number of experts and dimensions
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is None:
                # Try to infer from gate_up_proj shape
                gate_up_proj = getattr(experts, "gate_up_proj", None)
                if gate_up_proj is not None:
                    num_experts = gate_up_proj.size(0)
                else:
                    return orig(self_module, hidden_states)
            
            hidden_dim = hidden_states.size(-1)
            intermediate_dim = getattr(experts, "intermediate_size", hidden_dim)
            
            # Get top_k from router/gate
            router = getattr(self_module, "router", None) or getattr(self_module, "gate", None)
            top_k = getattr(router, "top_k", 4) if router else 4
            
            def_name = _infer_moe_def_name(
                num_experts, hidden_dim, intermediate_dim, top_k, "batched", hidden_states.dtype
            )

            # Reshape for tracing
            batch_size, seq_len, _ = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)

            rk: Dict[str, Any] = {
                "hidden_states": hidden_flat,
            }
            
            # Add expert weights if available
            gate_up_proj = getattr(experts, "gate_up_proj", None)
            down_proj = getattr(experts, "down_proj", None)
            if gate_up_proj is not None:
                rk["gate_up_proj"] = gate_up_proj
            if down_proj is not None:
                rk["down_proj"] = down_proj

            def _fallback(**_rk):
                return orig(self_module, hidden_states)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # The original returns (hidden_states, router_scores) for some models
            if isinstance(ret, tuple):
                return ret
            return ret

        return wrapper
