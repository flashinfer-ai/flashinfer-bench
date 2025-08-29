from pathlib import Path
from random import random
from typing import Any, Dict, Hashable, List

import torch

from flashinfer_bench.tracer import TraceEntry, TracingConfig, TracingRule

# ============================================================================
# Dedup Policy Presets
# ============================================================================
def policy_keep_all():
    """
    Keep all entries as unique.
    """
    def _policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        return entries
    return _policy

def policy_keep_first_k(k: int):
    """
    Keep first k entries as unique.
    """
    def policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        if k <= 0:
            raise ValueError("k must be > 0")
        entries_sorted = sorted(entries, key=lambda e: e.order)
        return entries_sorted[: min(k, len(entries_sorted))]
    return policy

def policy_keep_random_k(k: int):
    """
    Keep random k entries as unique.
    """
    def _policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if k > len(entries):
            return entries
        return random.sample(entries, k)
    return _policy

def policy_dedup_by_axes(k: int = 1):
    """
    Policy that deduplicates by same axes values.

    k: The number of entries with the same axes values to keep.
    """
    def _policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not entries:
            return []
        entries_sorted = sorted(entries, key=lambda e: e.order)
        kept: List["TraceEntry"] = []
        counts = {}
        for e in entries_sorted:
            key = tuple(sorted(e.axes.items()))
            c = counts.get(key, 0)
            if c < k:
                kept.append(e)
                counts[key] = c + 1
        return kept
    return _policy

def policy_dedup_by_avg_seq_len(k: int = 1):
    """Deduplicate by rounded average sequence length inferred from `kv_indptr`.

    - For each entry, if `picked['kv_indptr']` or `picked['seq_indptr']` exists and is valid (a 1-D tensor with
    length >= 2), we compute:
    avg_seq_len = int(round(indptr[-1].item() / len(indptr) - 1))
    Entries with the same `avg_seq_len` are considered duplicates, and at most
    `k` entries are kept for each `avg_seq_len` value.

    Args:
    k: max number of entries to keep per distinct average seq length (>0).


    Returns:
    A policy(entries) -> subset(entries), stable by entry.order.
    """
    def _policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        if k <= 0:
            raise ValueError("k must be > 0")
        kept: List["TraceEntry"] = []
        counts: Dict[int, int] = {}
        for e in entries:
            ten = e.picked.get("kv_indptr") or e.picked.get("seq_indptr")
            if not ten or ten.dim != 1 or ten.numel < 2:
                raise ValueError("indptr tensor doesn't exist or has invalid shape")
            total = ten[-1].item()
            bs = int(ten.numel - 1)
            avg = int(round(total / bs))
            c = counts.get(avg, 0)
            if c < k:
                kept.append(e)
                counts[avg] = c + 1
        return kept
    return _policy

# ============================================================================
# Dedup Keys Presets
# ============================================================================
def key_axes():
    """Key function for grouping entries by their axes."""
    def _key(entry: "TraceEntry") -> Hashable:
        return tuple(sorted(entry.axes.items()))
    return _key

# ============================================================================
# tensors_to_dump Presets
# ============================================================================
def dump_all():
    """Dump all tensors."""
    def _dump(inputs: Dict[str, Any]) -> List[str]:
        return [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]
    return _dump

def dump_none():
    """Dump no tensors."""
    def _dump(_: Dict[str, Any]) -> List[str]:
        return []
    return _dump

def dump_int32():
    """Select only int32 tensors for dumping. These inputs are usually indptrs."""
    def _dump(inputs: Dict[str, Any]) -> List[str]:
        picks: List[str] = []
        for name, val in inputs.items():
            if isinstance(val, torch.Tensor) and val.dtype == torch.int32:
                picks.append(name)
        return picks
    return _dump

# ============================================================================
# TracingRule Presets
# ============================================================================
gemm_rule = TracingRule(
    tensors_to_dump=dump_none,
    dedup_policy=policy_dedup_by_axes,
    dedup_keys=key_axes
)

mla_paged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

mla_ragged_prefill_rule = TracingRule(
    tensors_to_dump=["seq_indptr", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

mla_paged_decode_rule = TracingRule(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

gqa_paged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

gqa_ragged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

gqa_paged_decode_rule = TracingRule(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=policy_dedup_by_avg_seq_len,
    dedup_keys=key_axes
)

all_dump_rule = TracingRule(
    tensors_to_dump=dump_all,
    dedup_policy=policy_keep_all
)

axes_only_rule = TracingRule(
    tensors_to_dump=dump_none,
    dedup_policy=policy_dedup_by_axes,
)


# ============================================================================
# Config Presets
# ============================================================================
fib_full_tracing = TracingConfig(
    out_dir=Path("/tmp/traces"),
    blob_dir=Path("/tmp/blob"),
    rules={
        "gemm_n_28672_k_4096": gemm_rule,
        "gemm_n_4096_k_14336": gemm_rule,
        "gemm_n_4096_k_4096": gemm_rule,
        "gemm_n_6144_k_4096": gemm_rule,
        
        "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_rule,
        "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_rule,
        "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_rule,
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_rule,
        "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_rule,
        "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_rule,

        "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_rule,
        "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_rule,
        "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule
    }
)

fib_attn_tracing = TracingConfig(
    out_dir=Path("/tmp/traces"),
    blob_dir=Path("/tmp/blob"),
    rules={
        "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_rule,
        "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_rule,
        "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_rule,
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_rule,
        "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_rule,
        "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_rule,

        "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_rule,
        "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_rule,
        "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule
    }
)