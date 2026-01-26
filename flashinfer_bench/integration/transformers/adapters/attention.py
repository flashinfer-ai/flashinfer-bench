"""Attention adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.transformers.common import (
    SUPPORTED_ACTIVATION_DTYPES,
    infer_attention_def_name,
)


class AttentionAdapter:
    """Adapter for transformers attention functions.

    Patches the following attention implementations:
    - transformers.integrations.sdpa_attention.sdpa_attention_forward
    - transformers.integrations.flash_attention.flash_attention_forward
    - transformers.integrations.flex_attention.flex_attention_forward
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="transformers.integrations.sdpa_attention.sdpa_attention_forward",
                kind="function",
                name="sdpa_attention_forward",
                ctx_key="transformers_sdpa",
            ),
            PatchSpec(
                path="transformers.integrations.flash_attention.flash_attention_forward",
                kind="function",
                name="flash_attention_forward",
                ctx_key="transformers_flash",
            ),
            PatchSpec(
                path="transformers.integrations.flex_attention.flex_attention_forward",
                kind="function",
                name="flex_attention_forward",
                ctx_key="transformers_flex",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces attention calls."""

        def wrapper(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            dropout: float = 0.0,
            scaling: float | None = None,
            is_causal: bool | None = None,
            **kwargs,
        ) -> Any:
            # Validate tensor dimensions
            if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
                return orig(
                    module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
                )

            # Check for supported dtypes
            if query.dtype not in SUPPORTED_ACTIVATION_DTYPES:
                return orig(
                    module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
                )

            # Infer is_causal from module if not provided
            effective_is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

            # Build runtime kwargs for tracing
            def_name = infer_attention_def_name(query, key, effective_is_causal)

            # Prepare runtime kwargs matching expected definition inputs
            # Shape: query [batch, num_q_heads, seq_len, head_dim]
            # For ragged prefill, we need to reshape to [total_tokens, num_heads, head_dim]
            batch_size = query.shape[0]
            num_q_heads = query.shape[1]
            seq_len_q = query.shape[2]
            head_dim = query.shape[3]
            num_kv_heads = key.shape[1]
            seq_len_kv = key.shape[2]

            # Create indptr for ragged format (assuming contiguous sequences)
            # Each sequence in the batch contributes seq_len tokens
            qo_indptr = torch.arange(
                0, (batch_size + 1) * seq_len_q, seq_len_q,
                dtype=torch.int32, device=query.device
            )
            kv_indptr = torch.arange(
                0, (batch_size + 1) * seq_len_kv, seq_len_kv,
                dtype=torch.int32, device=key.device
            )

            # Reshape tensors to ragged format: [total_tokens, num_heads, head_dim]
            q_ragged = query.transpose(1, 2).reshape(-1, num_q_heads, head_dim)
            k_ragged = key.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
            v_ragged = value.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

            # Get sm_scale
            sm_scale = scaling if scaling is not None else (1.0 / (head_dim ** 0.5))

            rk: Dict[str, Any] = {
                "q": q_ragged,
                "k": k_ragged,
                "v": v_ragged,
                "qo_indptr": qo_indptr,
                "kv_indptr": kv_indptr,
                "sm_scale": sm_scale,
            }

            def _fallback(**_rk):
                return orig(
                    module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
                )

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # If apply returned a tensor result, reshape it back to expected format
            if isinstance(ret, torch.Tensor):
                # Reshape from [total_tokens, num_q_heads, head_dim] to [batch, seq_len, num_q_heads * head_dim]
                # and return as (attn_output, attn_weights)
                attn_output = ret.reshape(batch_size, seq_len_q, num_q_heads, head_dim)
                attn_output = attn_output.reshape(batch_size, seq_len_q, -1).contiguous()
                return attn_output, None

            return ret

        return wrapper
