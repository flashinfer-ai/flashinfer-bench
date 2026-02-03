"""Common utilities for FlashInfer integration adapters."""

import math
from typing import Any, Optional

import torch


def infer_kv_layout_from_args(inst) -> Optional[str]:
    layout = getattr(inst, "kv_layout", None)
    if isinstance(layout, str) and layout.upper() in ("NHD", "HND"):
        return layout.upper()
    return None


def infer_paged_kv_layout_from_tensors(paged_kv_cache, num_kv_heads: int) -> Optional[str]:
    # tuple: (k, v) 4D
    if isinstance(paged_kv_cache, tuple):
        k0 = paged_kv_cache[0]
        ndim = getattr(k0, "ndim", None)
        if ndim == 4:
            # NHD: [P, S, H, D]  => dim2 == H
            if k0.shape[2] == num_kv_heads:
                return "NHD"
            # HND: [P, H, S, D]  => dim1 == H
            if k0.shape[1] == num_kv_heads:
                return "HND"
        return None

    # single 5D: [P, 2, S/H, H/S, D]
    x = paged_kv_cache
    ndim = getattr(x, "ndim", None)
    if ndim == 5:
        # NHD: x[:, 0] -> [P, S, H, D] => dim3 == H
        if x.shape[3] == num_kv_heads:
            return "NHD"
        # HND: x[:, 0] -> [P, H, S, D] => dim2 == H
        if x.shape[2] == num_kv_heads:
            return "HND"
    return None


def infer_ragged_kv_layout_from_tensors(ragged_k_or_v, num_kv_heads: int) -> Optional[str]:
    if ragged_k_or_v.dim() != 3:
        return None
    if ragged_k_or_v.shape[1] == num_kv_heads:
        return "NHD"
    elif ragged_k_or_v.shape[0] == num_kv_heads:
        return "HND"
    return None


def normalize_paged_kv_to_nhd(paged_kv_cache, kv_layout: str):
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


def normalize_ragged_kv_to_nhd(ragged_k_or_v, kv_layout: str):
    if kv_layout == "NHD":
        return ragged_k_or_v
    else:
        return ragged_k_or_v.permute(1, 0, 2)


def pick_sm_scale_gqa(head_dim: int, maybe: Any) -> float:
    if maybe is None:
        return 1.0 / math.sqrt(float(head_dim))
    if isinstance(maybe, torch.Tensor):
        return float(maybe.item())
    return float(maybe)


def pick_sm_scale_mla(head_dim_qk_nope: int, head_dim_qk_pe: int, maybe: Any) -> float:
    if maybe is None:
        return 1.0 / math.sqrt(float(head_dim_qk_nope + head_dim_qk_pe))
    if isinstance(maybe, torch.Tensor):
        return float(maybe.item())
    return float(maybe)


# TODO(shanli): make kernels to take pre-allocated buffers and write in-place
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
