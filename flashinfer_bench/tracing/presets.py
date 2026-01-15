"""Built-in policies and tracing configurations."""

from __future__ import annotations

import logging
from functools import cache
from typing import Any, Dict, Hashable, List, Optional, Tuple

import torch

from flashinfer_bench.utils import is_dtype_integer

from .config import TracingConfig, TracingConfigRegistry
from .policy import PolicyRegistry
from .workload_entry import WorkloadEntry

logger = logging.getLogger(__name__)

# ============================================================================
# Input Dump Policies
# ============================================================================


@PolicyRegistry.register_input_dump_policy("dump_all")
class DumpAllPolicy:
    """Dump all tensor inputs."""

    def dump(self, inputs: Dict[str, Any]) -> List[str]:
        """Return names of all tensor inputs."""
        return [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]


@PolicyRegistry.register_input_dump_policy("dump_none")
class DumpNonePolicy:
    """Dump no inputs."""

    def dump(self, inputs: Dict[str, Any]) -> List[str]:
        """Return empty list."""
        return []


@PolicyRegistry.register_input_dump_policy("dump_int")
class DumpIntPolicy:
    """Dump only integer and boolean tensor inputs. These are usually flags or indptrs."""

    def dump(self, inputs: Dict[str, Any]) -> List[str]:
        """Return names of integer and boolean tensor inputs."""

        return [
            name
            for name, val in inputs.items()
            if isinstance(val, torch.Tensor) and is_dtype_integer(val.dtype)
        ]


# ============================================================================
# Filter Policies
# ============================================================================


@PolicyRegistry.register_filter_policy("keep_all")
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


@PolicyRegistry.register_filter_policy("keep_first_k")
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


@PolicyRegistry.register_filter_policy("keep_first")
class KeepFirstPolicy(KeepFirstKPolicy):
    """Keep only the first entry."""

    def __init__(self):
        super().__init__(k=1)


@PolicyRegistry.register_filter_policy("keep_none")
class KeepNonePolicy:
    """Keep no entries."""

    def __init__(self):
        pass

    def submit(self, entry: WorkloadEntry) -> None:
        pass

    def drain(self) -> List[WorkloadEntry]:
        return []

    def reset(self) -> None:
        pass


@PolicyRegistry.register_filter_policy("keep_first_k_by_axes")
class KeepFirstKByAxesPolicy:
    """Keep first k entries per unique axes combination."""

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


@PolicyRegistry.register_filter_policy("attention")
class AttentionFilterPolicy:
    """Deduplicate by average sequence length computed from indptr tensors."""

    def __init__(self, k: int = 1):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.seen_counts: Dict[Hashable, int] = {}
        self.entries: List[WorkloadEntry] = []

    def _get_axes_key(self, entry: WorkloadEntry) -> Optional[Tuple]:
        """Convert axes to a hashable key with computed avg lengths."""
        axes = entry.axes.copy()
        num_pages = axes.pop("num_pages", None)
        total_q = axes.pop("total_q", None)
        len_indptr = axes.pop("len_indptr", None)
        num_kv_indices = axes.pop("num_kv_indices", None)

        if num_pages is None or len_indptr is None or num_kv_indices is None:
            logger.error(
                f"No num_pages or len_indptr or num_kv_indices found in workload entry for {entry.def_name}"
            )
            return None

        avg_kv_len = int(round(num_kv_indices / (len_indptr - 1))) if len_indptr > 1 else 0
        axes["avg_kv_len"] = avg_kv_len

        if total_q is not None:
            axes["avg_q_len"] = int(round(total_q / (len_indptr - 1))) if len_indptr > 1 else 0
        else:
            axes["avg_q_len"] = 1

        return tuple(sorted(axes.items()))

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept entries up to k per unique (axes, avg_seq_len) combination."""
        axes_key = self._get_axes_key(entry)
        if axes_key is None:
            return

        count = self.seen_counts.get(axes_key, 0)
        if count < self.k:
            self.seen_counts[axes_key] = count + 1
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
# Tracing Configs
# ============================================================================


@cache
def _get_shared_configs() -> Dict[str, TracingConfig]:
    """Get shared config instances (cached)."""
    gemm = TracingConfig(input_dump_policy="dump_none", filter_policy="keep_first_k_by_axes")
    axes_only = TracingConfig(input_dump_policy="dump_none", filter_policy="keep_first_k_by_axes")
    mla_paged_prefill = TracingConfig(
        input_dump_policy=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
        filter_policy="attention",
        filter_policy_kwargs={"k": 1},
    )
    mla_paged_decode = TracingConfig(
        input_dump_policy=["kv_indptr", "kv_indices", "sm_scale"],
        filter_policy="attention",
        filter_policy_kwargs={"k": 1},
    )
    gqa_paged_prefill = TracingConfig(
        input_dump_policy=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
        filter_policy="attention",
        filter_policy_kwargs={"k": 1},
    )
    gqa_ragged_prefill = TracingConfig(
        input_dump_policy=["qo_indptr", "kv_indptr", "sm_scale"],
        filter_policy="attention",
        filter_policy_kwargs={"k": 1},
    )
    gqa_paged_decode = TracingConfig(
        input_dump_policy=["kv_indptr", "kv_indices", "sm_scale"],
        filter_policy="attention",
        filter_policy_kwargs={"k": 1},
    )
    return {
        "gemm": gemm,
        "axes_only": axes_only,
        "mla_paged_prefill": mla_paged_prefill,
        "mla_paged_decode": mla_paged_decode,
        "gqa_paged_prefill": gqa_paged_prefill,
        "gqa_ragged_prefill": gqa_ragged_prefill,
        "gqa_paged_decode": gqa_paged_decode,
    }


@cache
def get_default_configs() -> TracingConfigRegistry:
    """Get the default tracing configs (cached)."""
    return TracingConfigRegistry(default=TracingConfig())


@cache
def get_full_configs() -> TracingConfigRegistry:
    """Get the full tracing configs with GEMM, attention, and normalization configs (cached)."""
    configs = _get_shared_configs()
    return TracingConfigRegistry(
        default=TracingConfig(),
        per_definition={
            "gemm_n128_k2048": configs["gemm"],
            "gemm_n256_k7168": configs["gemm"],
            "gemm_n2048_k4096": configs["gemm"],
            "gemm_n4096_k14336": configs["gemm"],
            "gemm_n4096_k4096": configs["gemm"],
            "gemm_n5120_k2048": configs["gemm"],
            "gemm_n6144_k4096": configs["gemm"],
            "gemm_n28672_k4096": configs["gemm"],
            "gqa_paged_decode_h32_kv4_d128_ps1": configs["gqa_paged_decode"],
            "gqa_paged_decode_h32_kv8_d128_ps1": configs["gqa_paged_decode"],
            "gqa_paged_prefill_causal_h32_kv4_d128_ps1": configs["gqa_paged_prefill"],
            "gqa_paged_prefill_causal_h32_kv8_d128_ps1": configs["gqa_paged_prefill"],
            "gqa_ragged_prefill_causal_h32_kv4_d128": configs["gqa_ragged_prefill"],
            "gqa_ragged_prefill_causal_h32_kv8_d128": configs["gqa_ragged_prefill"],
            "mla_paged_decode_h16_ckv512_kpe64_ps1": configs["mla_paged_decode"],
            "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": configs["mla_paged_prefill"],
            "fused_add_rmsnorm_h2048": configs["axes_only"],
            "fused_add_rmsnorm_h4096": configs["axes_only"],
            "fused_add_rmsnorm_h7168": configs["axes_only"],
        },
    )


@cache
def get_attention_only_configs() -> TracingConfigRegistry:
    """Get the attention-only tracing configs (cached)."""
    configs = _get_shared_configs()
    return TracingConfigRegistry(
        default=TracingConfig(),
        per_definition={
            "gqa_paged_decode_h32_kv4_d128_ps1": configs["gqa_paged_decode"],
            "gqa_paged_decode_h32_kv8_d128_ps1": configs["gqa_paged_decode"],
            "gqa_paged_prefill_causal_h32_kv4_d128_ps1": configs["gqa_paged_prefill"],
            "gqa_paged_prefill_causal_h32_kv8_d128_ps1": configs["gqa_paged_prefill"],
            "gqa_ragged_prefill_causal_h32_kv4_d128": configs["gqa_ragged_prefill"],
            "gqa_ragged_prefill_causal_h32_kv8_d128": configs["gqa_ragged_prefill"],
            "mla_paged_decode_h16_ckv512_kpe64_ps1": configs["mla_paged_decode"],
            "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": configs["mla_paged_prefill"],
        },
    )
