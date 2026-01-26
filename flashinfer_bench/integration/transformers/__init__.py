"""Transformers integration for automatic tracing of model operations."""

from __future__ import annotations

from flashinfer_bench.integration.patch_manager import get_manager

from .adapters.activation import ActivationAdapter
from .adapters.attention import AttentionAdapter
from .adapters.embedding import EmbeddingAdapter
from .adapters.moe import MoEAdapter
from .adapters.rmsnorm import RMSNormAdapter
from .adapters.rope import RoPEAdapter
from .adapters.sampling import SamplingAdapter


def install_transformers_integrations() -> None:
    """
    Install patches for transformers operations. If a target does not exist in
    the current environment, skip silently. Idempotent.

    Patches the following operations:
    - Attention: SDPA, Flash Attention, Flex Attention
    - RMSNorm: torch.nn.functional.rms_norm
    - RoPE: apply_rotary_pos_emb (multiple model implementations)
    - Embedding: torch.nn.functional.embedding
    - Activations: SiLU, GELU
    - MoE: batched_mm and grouped_mm expert forwards
    - Sampling: multinomial, softmax, top-k
    - Linear: Already covered by flashinfer integration (torch.nn.functional.linear)
    """
    print("Installing transformers integrations...")
    mgr = get_manager()

    adapters = [
        AttentionAdapter(),
        RMSNormAdapter(),
        RoPEAdapter(),
        EmbeddingAdapter(),
        ActivationAdapter(),
        MoEAdapter(),
        SamplingAdapter(),
        # LinearAdapter already installed via flashinfer integration
    ]

    for adp in adapters:
        try:
            targets = adp.targets()
        except Exception:
            continue
        for spec in targets:
            mgr.patch(spec, adp.make_wrapper)


__all__ = ["install_transformers_integrations"]
