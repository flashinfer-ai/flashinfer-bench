"""Embedding adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _infer_embedding_def_name(weight: torch.Tensor) -> str:
    """Infer definition name for embedding operation."""
    num_embeddings, embedding_dim = weight.shape
    return f"embedding_v{num_embeddings}_d{embedding_dim}"


class EmbeddingAdapter:
    """Adapter for torch.nn.functional.embedding.

    Traces embedding lookups used in LLM token embeddings.
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="torch.nn.functional.embedding",
                kind="function",
                name="embedding",
                ctx_key="torch_embedding",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces embedding calls."""

        def wrapper(
            input: torch.Tensor,
            weight: torch.Tensor,
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
        ) -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
                return orig(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            # Only trace on CUDA with supported dtypes
            if not weight.is_cuda:
                return orig(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            if weight.dtype not in (torch.float16, torch.bfloat16):
                return orig(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            def_name = _infer_embedding_def_name(weight)

            # Flatten input for tracing
            input_flat = input.reshape(-1)

            rk: Dict[str, Any] = {
                "input_ids": input_flat,
                "weight": weight,
            }

            def _fallback(**_rk):
                return orig(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # Reshape output back to original shape + embedding_dim
            if isinstance(ret, torch.Tensor):
                return ret.reshape(*input.shape, weight.shape[1])

            return ret

        return wrapper
