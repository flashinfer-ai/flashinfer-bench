from typing import Any, Dict, Hashable, List, Optional

import torch

from .tracing_config import DedupPolicyFactory, TracingConfig, WorkloadEntry

# ============================================================================
# Dedup Policy Implementations
# ============================================================================


class KeepAllPolicy:
    """Keep all entries without deduplication."""

    def __init__(self):
        self.entries: List[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept all entries."""
        self.entries.append(entry)

    def drain(self) -> List[WorkloadEntry]:
        """Return all submitted entries."""
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        """Clear all buffered entries."""
        self.entries.clear()


class KeepFirstKPolicy:
    """Keep the first k entries by order."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.entries: List[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept entries until k entries are collected."""
        if len(self.entries) < self.k:
            self.entries.append(entry)

    def drain(self) -> List[WorkloadEntry]:
        """Return the first k entries."""
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        """Clear all buffered entries."""
        self.entries.clear()


class DedupByAxesPolicy:
    """Deduplicate by axes values.

    Maintains a count of how many entries have been seen for each unique
    axes combination, and keeps at most k entries per combination.

    Parameters
    ----------
    k : int
        Maximum number of entries to keep per unique axes combination.
    """

    def __init__(self, k: int = 1):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.seen_counts: Dict[Hashable, int] = {}
        self.entries: List[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept entries up to k per unique axes combination."""
        key = tuple(sorted(entry.axes.items()))
        count = self.seen_counts.get(key, 0)
        if count < self.k:
            self.seen_counts[key] = count + 1
            self.entries.append(entry)

    def drain(self) -> List[WorkloadEntry]:
        """Return all selected entries."""
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        """Clear the seen counts and buffered entries."""
        self.seen_counts.clear()
        self.entries.clear()


class DedupByAvgSeqLenPolicy:
    """Deduplicate by average sequence length with two-stage bucketing.

    This policy implements two-stage deduplication:
    1. Group entries by axes values
    2. Within each axes group, deduplicate by average sequence length computed from indptr tensors

    Parameters
    ----------
    k : int
        Maximum number of entries to keep per unique average sequence length within each axes group.
    indptr_names : List[str]
        Names of indptr tensors to use for computing average sequence length.
    """

    def __init__(self, k: int = 1, indptr_names: Optional[List[str]] = None):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.indptr_names = indptr_names or ["kv_indptr", "seq_indptr"]
        # Two-level state: axes_key -> (avg_len -> count)
        self.seen_counts: Dict[Hashable, Dict[int, int]] = {}
        self.entries: List[WorkloadEntry] = []

    def _compute_avg_seq_len(self, entry: WorkloadEntry) -> Optional[int]:
        """Compute average sequence length from indptr tensor."""
        for name in self.indptr_names:
            ten = entry.tensors_to_dump.get(name)
            if ten is not None:
                if ten.dim() != 1 or ten.numel() < 2:
                    continue
                total = ten[-1].item()
                bs = len(ten) - 1
                return int(round(total / bs))
        return None

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept entries up to k per unique (axes, avg_seq_len) combination."""
        # Compute axes key and avg seq len
        axes_key = tuple(sorted(entry.axes.items()))
        avg_len = self._compute_avg_seq_len(entry)

        if avg_len is None:
            return

        # Get or create the avg_len counts for this axes group
        if axes_key not in self.seen_counts:
            self.seen_counts[axes_key] = {}

        avg_len_counts = self.seen_counts[axes_key]
        count = avg_len_counts.get(avg_len, 0)

        if count < self.k:
            avg_len_counts[avg_len] = count + 1
            self.entries.append(entry)

    def drain(self) -> List[WorkloadEntry]:
        """Return all selected entries."""
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        """Clear all seen counts and buffered entries."""
        self.seen_counts.clear()
        self.entries.clear()


# ============================================================================
# Built-in Dedup Policy Factories
# ============================================================================

BUILTIN_DEDUP_POLICY_FACTORIES: Dict[str, DedupPolicyFactory] = {
    "keep_all": lambda: KeepAllPolicy(),
    "keep_first": lambda: KeepFirstKPolicy(k=1),
    "dedup_by_axes": lambda: DedupByAxesPolicy(k=1),
}


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
gemm_rule = TracingConfig(tensors_to_dump=DUMP_NONE, dedup_policy="dedup_by_axes")

mla_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1),
)

mla_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["seq_indptr", "sm_scale"], dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1)
)

mla_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1),
)

gqa_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1),
)

gqa_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"],
    dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1),
)

gqa_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: DedupByAvgSeqLenPolicy(k=1),
)

all_dump_rule = TracingConfig(tensors_to_dump=DUMP_ALL, dedup_policy="keep_all")

axes_only_rule = TracingConfig(tensors_to_dump=DUMP_NONE, dedup_policy="dedup_by_axes")

fib_full_tracing = {
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
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule,
}

fib_attn_tracing = {
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_rule,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_rule,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_rule,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_rule,
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule,
}
