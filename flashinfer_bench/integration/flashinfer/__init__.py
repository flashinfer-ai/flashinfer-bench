from __future__ import annotations

from ..patch_manager import get_manager
from .adapters.prefill_gqa import PrefillGqaPagedAdapter


def install_flashinfer_integrations() -> None:
    """
    Install patches for a set of adapters. If a target does not exist in
    the current environment, skip silently. Idempotent.
    """
    mgr = get_manager()

    adapters = [PrefillGqaPagedAdapter()]

    for adp in adapters:
        try:
            targets = adp.targets()
        except Exception:
            continue
        for spec in targets:
            mgr.patch(spec, adp.make_wrapper)
