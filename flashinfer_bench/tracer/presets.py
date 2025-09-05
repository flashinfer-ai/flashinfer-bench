import random
from typing import Any, Dict, List, Optional

import torch

from .types import TraceEntry, TracingRule


# ============================================================================
# Dedup Policy Presets
# ============================================================================
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


def policy_keep_random_k(k: int, seed: Optional[int] = None):
    """
    Keep random k entries as unique.
    """

    def _policy(entries: List["TraceEntry"]) -> List["TraceEntry"]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if k > len(entries):
            return entries
        if seed is not None:
            random.seed(seed)
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
    avg_seq_len = int(round(indptr[-1].item() / (len(indptr) - 1)))
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
            ten = e.picked.get("kv_indptr")
            if not ten:
                ten = e.picked.get("seq_indptr")
            if not ten or ten.dim() != 1 or ten.numel() < 2:
                raise ValueError("indptr tensor doesn't exist or has invalid shape")
            total = ten[-1].item()
            bs = len(ten) - 1
            avg = int(round(total / bs))
            c = counts.get(avg, 0)
            if c < k:
                kept.append(e)
                counts[avg] = c + 1
        return kept

    return _policy


# Keep all entries as unique.
KEEP_ALL = lambda entries: entries
# Keep one random entry as unique.
KEEP_RANDOM_ONE = policy_keep_random_k(1)
# Deduplicate by axes values and keep one entry for exact same axes.
DEDUP_BY_AXES = policy_dedup_by_axes()
# Deduplicate by average sequence length and keep one entry for exact same length.
DEDUP_BY_AVG_SEQ_LEN = policy_dedup_by_avg_seq_len()

# ============================================================================
# Dedup Keys Presets
# ============================================================================

# Generate key by entry axis values.
KEY_AXES = lambda entry: tuple(sorted(entry.axes.items()))


# ============================================================================
# tensors_to_dump Presets
# ============================================================================
def dump_int32():
    """Select only int32 tensors for dumping. These inputs are usually indptrs."""

    def _dump(inputs: Dict[str, Any]) -> List[str]:
        picks: List[str] = []
        for name, val in inputs.items():
            if isinstance(val, torch.Tensor) and val.dtype == torch.int32:
                picks.append(name)
        return picks

    return _dump


# Dump all tensors.
DUMP_ALL = lambda inputs: [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]
# Dump no tensors.
DUMP_NONE = lambda _: []
DUMP_INT32 = dump_int32()

# ============================================================================
# TracingRule Presets
# ============================================================================
gemm_rule = TracingRule(tensors_to_dump=DUMP_NONE, dedup_policy=DEDUP_BY_AXES, dedup_keys=KEY_AXES)

mla_paged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

mla_ragged_prefill_rule = TracingRule(
    tensors_to_dump=["seq_indptr", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

mla_paged_decode_rule = TracingRule(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

gqa_paged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

gqa_ragged_prefill_rule = TracingRule(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

gqa_paged_decode_rule = TracingRule(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DEDUP_BY_AVG_SEQ_LEN,
    dedup_keys=KEY_AXES,
)

all_dump_rule = TracingRule(tensors_to_dump=DUMP_ALL, dedup_policy=KEEP_ALL)

axes_only_rule = TracingRule(
    tensors_to_dump=DUMP_NONE,
    dedup_policy=DEDUP_BY_AXES,
)
