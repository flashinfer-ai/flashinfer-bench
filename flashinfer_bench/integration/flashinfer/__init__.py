"""FlashInfer integration for automatic tracing and apply functionality."""

from __future__ import annotations

import os

from flashinfer_bench.integration.patch_manager import get_manager

from .adapters.gqa_paged_decode import GQAPagedDecodeAdapter
from .adapters.gqa_paged_prefill import GQAPagedPrefillAdapter
from .adapters.linear import LinearAdapter
from .adapters.mla_paged import MLAPagedAdapter
from .adapters.ragged_prefill import RaggedPrefillAdapter
from .adapters.rmsnorm import RMSNormAdapter


def _select_adapters(scope: str):
    attention_adapters = [
        GQAPagedPrefillAdapter(),
        RaggedPrefillAdapter(),
        GQAPagedDecodeAdapter(),
        MLAPagedAdapter(),
    ]
    gemm_adapters = [LinearAdapter()]
    rmsnorm_adapters = [RMSNormAdapter()]

    if scope == "gemm_only":
        return gemm_adapters
    if scope == "rmsnorm_only":
        return rmsnorm_adapters
    if scope == "attention_only":
        return attention_adapters
    return [*attention_adapters, *gemm_adapters, *rmsnorm_adapters]


def install_flashinfer_integrations(scope: str | None = None) -> None:
    """
    Install patches for a set of adapters. If a target does not exist in
    the current environment, skip silently. Idempotent.
    """
    resolved_scope = scope or os.environ.get("FIB_APPLY_ADAPTER_SCOPE", "all")
    print(f"Installing flashinfer integrations (scope={resolved_scope})...")
    mgr = get_manager()
    adapters = _select_adapters(resolved_scope)

    for adp in adapters:
        try:
            targets = adp.targets()
        except Exception:
            continue
        for spec in targets:
            mgr.patch(spec, adp.make_wrapper)


__all__ = ["install_flashinfer_integrations"]
