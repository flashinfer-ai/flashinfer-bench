"""RoPE (Rotary Position Embedding) adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _infer_rope_def_name(q: torch.Tensor) -> str:
    """Infer definition name for RoPE operation.
    
    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch, num_heads, seq_len, head_dim]
        
    Returns
    -------
    str
        Definition name like "rope_h32_d128"
    """
    num_heads = q.shape[1]
    head_dim = q.shape[-1]
    return f"rope_h{num_heads}_d{head_dim}"


class RoPEAdapter:
    """Adapter for Rotary Position Embedding operations.

    Patches apply_rotary_pos_emb functions in various model files:
    - transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    - transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb
    - transformers.models.qwen3_moe.modeling_qwen3_moe.apply_rotary_pos_emb
    - transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
    - transformers.models.gpt_oss.modeling_gpt_oss.apply_rotary_pos_emb
    """

    def targets(self) -> List[PatchSpec]:
        # RoPE is defined in each model file, so we need to patch multiple paths
        model_paths = [
            "transformers.models.llama.modeling_llama",
            "transformers.models.llama4.modeling_llama4",
            "transformers.models.qwen3.modeling_qwen3",
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "transformers.models.qwen2.modeling_qwen2",
            "transformers.models.qwen2_moe.modeling_qwen2_moe",
            "transformers.models.mistral.modeling_mistral",
            "transformers.models.mixtral.modeling_mixtral",
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "transformers.models.gemma.modeling_gemma",
            "transformers.models.gemma2.modeling_gemma2",
            "transformers.models.phi3.modeling_phi3",
            "transformers.models.cohere.modeling_cohere",
            "transformers.models.cohere2.modeling_cohere2",
        ]
        
        specs = []
        for path in model_paths:
            specs.append(
                PatchSpec(
                    path=f"{path}.apply_rotary_pos_emb",
                    kind="function",
                    name="apply_rotary_pos_emb",
                    ctx_key=f"rope_{path.split('.')[-1]}",
                )
            )
        return specs

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces RoPE calls."""

        def wrapper(
            q: torch.Tensor,
            k: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            unsqueeze_dim: int = 1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Validate inputs
            if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
                return orig(q, k, cos, sin, unsqueeze_dim)

            # Only trace on CUDA with supported dtypes
            if not q.is_cuda:
                return orig(q, k, cos, sin, unsqueeze_dim)

            if q.dtype not in (torch.float16, torch.bfloat16):
                return orig(q, k, cos, sin, unsqueeze_dim)

            # Validate tensor dimensions (expect 4D: batch, heads, seq, head_dim)
            if q.dim() != 4 or k.dim() != 4:
                return orig(q, k, cos, sin, unsqueeze_dim)

            def_name = _infer_rope_def_name(q)

            # Extract dimensions
            batch_size = q.shape[0]
            num_q_heads = q.shape[1]
            num_k_heads = k.shape[1]
            seq_len = q.shape[2]
            head_dim = q.shape[3]

            rk: Dict[str, Any] = {
                "q": q,
                "k": k,
                "cos": cos,
                "sin": sin,
            }

            def _fallback(**_rk):
                return orig(q, k, cos, sin, unsqueeze_dim)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # Expect tuple of (q_embed, k_embed)
            if isinstance(ret, tuple) and len(ret) == 2:
                return ret

            return ret

        return wrapper
