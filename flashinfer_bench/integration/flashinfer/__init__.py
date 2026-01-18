"""FlashInfer integration for automatic tracing and apply functionality."""

from __future__ import annotations

from flashinfer_bench.integration.patch_manager import get_manager

from .adapters.gqa_paged_decode import GQAPagedDecodeAdapter
from .adapters.gqa_paged_prefill import GQAPagedPrefillAdapter
from .adapters.linear import LinearAdapter
from .adapters.mla_paged import MLAPagedAdapter
from .adapters.ragged_prefill import RaggedPrefillAdapter
from .adapters.rmsnorm import RMSNormAdapter


def install_flashinfer_integrations() -> None:
    """
    Install patches for a set of adapters. If a target does not exist in
    the current environment, skip silently. Idempotent.
    """
    print("Installing flashinfer integrations...")
    mgr = get_manager()

    adapters = [
        GQAPagedPrefillAdapter(),
        RaggedPrefillAdapter(),
        GQAPagedDecodeAdapter(),
        MLAPagedAdapter(),
        LinearAdapter(),
        RMSNormAdapter(),
    ]

    for adp in adapters:
        try:
            targets = adp.targets()
        except Exception:
            continue
        for spec in targets:
            mgr.patch(spec, adp.make_wrapper)


__all__ = ["install_flashinfer_integrations"]
