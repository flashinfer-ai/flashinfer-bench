from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Mapping, Tuple
from weakref import WeakKeyDictionary

import torch


class ArgBinder:
    """Cache inspect.signature and bind once per callable."""

    def __init__(self, fn) -> None:
        self._sig = inspect.signature(fn)

    @classmethod
    def from_callable(cls, fn) -> "ArgBinder":
        return cls(fn)

    def bind(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        ba = self._sig.bind_partial(*args, **kwargs)
        ba.apply_defaults()
        return dict(ba.arguments)


class ContextStore:
    """Per-instance loose store; adapter decides fields."""

    def __init__(self) -> None:
        self._store: "WeakKeyDictionary[object, Dict[str, Any]]" = WeakKeyDictionary()

    def get(self, inst: object) -> Dict[str, Any]:
        d = self._store.get(inst)
        if d is None:
            d = {}
            self._store[inst] = d
        return d


def infer_kv_layout(inst) -> str:
    layout = getattr(inst, "kv_layout", None)
    if isinstance(layout, str) and layout.upper() in ("NHD", "HND"):
        return layout.upper()
    return "NHD"


def split_paged_kv_to_nhd(paged_kv_cache, kv_layout: str):
    if isinstance(paged_kv_cache, tuple):
        k, v = paged_kv_cache
        if kv_layout == "NHD":
            return k, v
        else:  # HND: [P, H, S, D]
            return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    x: torch.Tensor = paged_kv_cache
    assert x.dim() == 5, "paged_kv_cache must be 5D when passed as a single tensor"
    if kv_layout == "NHD":
        k = x[:, 0]
        v = x[:, 1]
        return k, v
    else:
        k = x[:, 0].permute(0, 2, 1, 3)
        v = x[:, 1].permute(0, 2, 1, 3)
        return k, v


def pick_sm_scale(head_dim: int, maybe: Any) -> float:
    if maybe is None:
        return 1.0 / math.sqrt(float(head_dim))
    if isinstance(maybe, torch.Tensor):
        return float(maybe.item())
    return float(maybe)


def write_back_outputs(
    *, output: torch.Tensor, lse: torch.Tensor, want_lse: bool, out_buf=None, lse_buf=None
):
    if out_buf is not None:
        out_buf.copy_(output)
        output = out_buf
    if want_lse:
        if lse_buf is not None:
            lse_buf.copy_(lse)
            lse = lse_buf
        return output, lse
    return output
