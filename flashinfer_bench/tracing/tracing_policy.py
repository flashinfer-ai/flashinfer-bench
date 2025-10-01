"""Deduplication policies and tensor dump functions for workload tracing.

This module contains:
- DedupPolicy protocol and all builtin implementations
- TensorsDumpFunction and all builtin implementations
- Type aliases and factory functions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Protocol

import torch

from flashinfer_bench.logging import get_logger

logger = get_logger("TracingPolicy")


# ============================================================================
# WorkloadEntry
# ============================================================================


@dataclass
class WorkloadEntry:
    """In-memory buffer entry for collected workloads."""

    def_name: str
    """Name of the definition this workload entry belongs to."""

    axes: Dict[str, int]
    """Dictionary mapping axis names to their concrete integer values."""

    inputs_to_dump: Dict[str, Any]
    """Inputs to dump. Maps input name to the tensor to dump. This field will be further stored
    to disk as a tensor blob."""

    order: int
    """Sequential order number for this entry in the collection process."""

    cuda_graph_snapshot: Optional[Dict[str, Any]] = None
    """CPU snapshot of tensors collected during CUDA Graph replay, if applicable."""


# ============================================================================
# TensorsDump Type Aliases and Functions
# ============================================================================


InputDumpPolicyFunction = Callable[[Dict[str, Any]], List[str]]
"""Function that selects which inputs to dump from runtime arguments."""


def dump_all(inputs: Dict[str, Any]) -> List[str]:
    """Dump all tensors."""
    return [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]


def dump_none(inputs: Dict[str, Any]) -> List[str]:
    """Dump no tensors."""
    return []


def dump_int32(inputs: Dict[str, Any]) -> List[str]:
    """Select only int32 tensors for dumping. These inputs are usually indptrs."""
    picks: List[str] = []
    for name, val in inputs.items():
        if isinstance(val, torch.Tensor) and val.dtype == torch.int32:
            picks.append(name)
    return picks


# Built-in input dump policy functions
BUILTIN_INPUT_DUMP_POLICIES: Dict[str, InputDumpPolicyFunction] = {
    "dump_all": dump_all,
    "dump_none": dump_none,
    "dump_int32": dump_int32,
}


# ============================================================================
# DedupPolicy Protocol and Type Aliases
# ============================================================================


class DedupPolicy(Protocol):
    """Protocol for workload deduplication policy.

    A dedup policy maintains internal state and supports both online and offline
    deduplication strategies. Entries are submitted one at a time via submit(),
    and selected entries are retrieved via drain().
    """

    def submit(self, entry: WorkloadEntry) -> None:
        """Submit a workload entry for deduplication consideration.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to submit.
        """
        ...

    def drain(self) -> List[WorkloadEntry]:
        """Drain and return all selected entries.

        Returns
        -------
        List[WorkloadEntry]
            List of entries that passed the deduplication policy.
            After calling this method, the internal buffer is cleared.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of the deduplication policy.

        This method should be called when starting a new deduplication session
        to clear any cached state or statistics.
        """
        ...


DedupPolicyFactory = Callable[[], DedupPolicy]
"""Factory function for dedup policy."""


# ============================================================================
# Builtin DedupPolicy Implementations
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


class KeepFirstByAxesPolicy:
    """Keep first k entries per unique axes combination.

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


class AttentionDedupPolicy:
    # This policy is not done and not need to be tested yet.
    """Deduplicate by average sequence length computed from indptr tensors.

    This policy implements two-stage bucketing:
    1. Group entries by axes values
    2. Within each axes group, keep first k entries per average sequence length computed from indptr tensors

    Parameters
    ----------
    k : int
        Maximum number of entries to keep per unique average sequence length within each axes group.
    """

    def __init__(self, k: int = 1):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.seen_counts: Dict[Hashable, int] = {}
        self.entries: List[WorkloadEntry] = []

    def _get_axes_key(self, entry: WorkloadEntry) -> Dict[str, Any]:
        """Convert axes to a hashable key."""
        axes = entry.axes.copy()
        num_pages = axes.pop("num_pages", None)
        total_q = axes.pop("total_q", None)
        len_indptr = axes.pop("len_indptr", None)
        num_kv_indices = axes.pop("num_kv_indices", None)

        if num_pages is None or len_indptr is None or num_kv_indices is None:
            logger.error(
                f"No num_pages or len_indptr or num_kv_indices found in workload entry for {entry.def_name}"
            )
            return

        # Calculate average key-value length
        avg_kv_len = int(round(num_kv_indices / (len_indptr - 1)))
        axes["avg_kv_len"] = avg_kv_len

        # Calculate average query length
        if total_q is not None:
            axes["avg_q_len"] = int(round(total_q / (len_indptr - 1)))
        else:
            # Decode workload: avg_q_len == 1
            axes["avg_q_len"] = 1

        return tuple(sorted(axes.items()))

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept entries up to k per unique (axes, avg_seq_len) combination."""
        # Compute axes key and avg seq len
        axes_key = self._get_axes_key(entry)

        # Get or create the avg_len counts for this axes group
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


class KeepNonePolicy:
    """Keep no entries."""

    def __init__(self):
        pass

    def submit(self, entry: WorkloadEntry) -> None:
        """Accept no entries."""
        pass

    def drain(self) -> List[WorkloadEntry]:
        """Return no entries."""
        return []

    def reset(self) -> None:
        """Clear all buffered entries."""
        pass


BUILTIN_DEDUP_POLICIES: Dict[str, DedupPolicyFactory] = {
    "keep_all": lambda: KeepAllPolicy(),
    "keep_first": lambda: KeepFirstKPolicy(k=1),
    "keep_first_by_axes": lambda: KeepFirstByAxesPolicy(k=1),
    "keep_none": lambda: KeepNonePolicy(),
}
