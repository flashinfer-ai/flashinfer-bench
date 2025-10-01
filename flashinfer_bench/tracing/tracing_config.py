from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional, Protocol, Union


@dataclass
class WorkloadEntry:
    """In-memory buffer entry for collected workloads."""

    def_name: str
    """Name of the definition this workload entry belongs to."""

    axes: Dict[str, int]
    """Dictionary mapping axis names to their concrete integer values."""

    tensors_to_dump: Dict[str, Any]
    """Tensors to dump. Maps input name to the tensor to dump."""

    order: int
    """Sequential order number for this entry in the collection process."""

    cuda_graph_snapshot: Optional[Dict[str, Any]] = None
    """CPU snapshot of tensors collected during CUDA Graph replay, if applicable."""


class DedupPolicy(Protocol):
    """Protocol for online workload deduplication policy.

    A dedup policy maintains internal state and decides whether to keep
    each workload entry as it arrives, enabling streaming deduplication.
    """

    def __call__(self, entry: WorkloadEntry) -> bool:
        """Decide whether to keep this entry.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to evaluate.

        Returns
        -------
        bool
            True if the entry should be kept, False to drop it.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of the deduplication policy.

        This method should be called when starting a new deduplication session
        to clear any cached state or statistics.
        """
        ...


# ============================================================================
# Dedup Policy Classes
# ============================================================================


class KeepAllPolicy:
    """Keep all entries without deduplication."""

    def __call__(self, entry: WorkloadEntry) -> bool:
        return True

    def reset(self) -> None:
        """No-op since this policy has no state."""
        pass


class KeepFirstKPolicy:
    """Keep the first k entries by order."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.count = 0

    def __call__(self, entry: WorkloadEntry) -> bool:
        if self.count < self.k:
            self.count += 1
            return True
        return False

    def reset(self) -> None:
        """Reset the counter."""
        self.count = 0


class DedupByAxesPolicy:
    """Deduplicate by axes values online.

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

    def __call__(self, entry: WorkloadEntry) -> bool:
        key = tuple(sorted(entry.axes.items()))
        count = self.seen_counts.get(key, 0)
        if count < self.k:
            self.seen_counts[key] = count + 1
            return True
        return False

    def reset(self) -> None:
        """Clear the seen counts."""
        self.seen_counts.clear()


class DedupByAvgSeqLenPolicy:
    """Deduplicate by average sequence length with two-stage bucketing online.

    This policy implements two-stage online deduplication:
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

    def __call__(self, entry: WorkloadEntry) -> bool:
        # Compute axes key and avg seq len
        axes_key = tuple(sorted(entry.axes.items()))
        avg_len = self._compute_avg_seq_len(entry)

        if avg_len is None:
            return False

        # Get or create the avg_len counts for this axes group
        if axes_key not in self.seen_counts:
            self.seen_counts[axes_key] = {}

        avg_len_counts = self.seen_counts[axes_key]
        count = avg_len_counts.get(avg_len, 0)

        if count < self.k:
            avg_len_counts[avg_len] = count + 1
            return True
        return False

    def reset(self) -> None:
        """Clear all seen counts."""
        self.seen_counts.clear()


# Literal types for built-in dedup policies
DedupPolicyLiteral = Literal["keep_all", "keep_first", "dedup_by_axes"]

# Singleton cache for built-in policy instances
_BUILTIN_POLICY_CACHE: Dict[str, DedupPolicy] = {}


def _get_builtin_policy(literal: str) -> DedupPolicy:
    """Get or create a built-in policy instance (cached singleton).

    Parameters
    ----------
    literal : str
        The policy literal name.

    Returns
    -------
    DedupPolicy
        The policy instance.

    Raises
    ------
    ValueError
        If the literal is unknown.
    """
    if literal not in _BUILTIN_POLICY_CACHE:
        if literal == "keep_all":
            _BUILTIN_POLICY_CACHE[literal] = KeepAllPolicy()
        elif literal == "keep_first":
            _BUILTIN_POLICY_CACHE[literal] = KeepFirstKPolicy(k=1)
        elif literal == "dedup_by_axes":
            _BUILTIN_POLICY_CACHE[literal] = DedupByAxesPolicy(k=1)
        else:
            raise ValueError(
                f"Unknown dedup_policy literal: {literal}. "
                f"Must be one of ['keep_all', 'keep_first', 'dedup_by_axes']"
            )
    return _BUILTIN_POLICY_CACHE[literal]


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    """Which inputs to persist. List[str] for static selection, Callable for dynamic."""

    dedup_policy: Union[DedupPolicy, DedupPolicyLiteral]
    """Deduplication policy. Can be a DedupPolicy object or a string literal for built-in policies."""

    def __post_init__(self):
        """Convert literal dedup policy strings to actual policy objects."""
        if isinstance(self.dedup_policy, str):
            self.dedup_policy = _get_builtin_policy(self.dedup_policy)

    def get_tensors_to_dump(self, runtime_args: Dict[str, Any]) -> List[str]:
        """Get the tensors to dump from the runtime arguments. The validity of the result is
        checked, so every returned tensor name must exist in the runtime arguments.

        Parameters
        ----------
        runtime_args : Dict[str, Any]
            The runtime arguments to get the tensors to dump from.

        Returns
        -------
        List[str]
            The tensors to dump.

        Raises
        ------
        ValueError
            If tensors_to_dump is not a list of strings or a callable, or the result is not valid.
        """
        if isinstance(self.tensors_to_dump, list):
            result = self.tensors_to_dump
        elif callable(self.tensors_to_dump):
            result = self.tensors_to_dump(runtime_args)
        else:
            raise ValueError("tensors_to_dump must be a list of strings or a callable")

        # Check the validity of the result
        if not isinstance(result, list):
            raise ValueError("tensors_to_dump callable must return a list of strings")
        for name in result:
            if not isinstance(name, str):
                raise ValueError(
                    f"tensors_to_dump callable must return a list of strings, but got "
                    f"{type(name).__name__}"
                )
            if name not in runtime_args:
                raise ValueError(
                    f"tensors_to_dump callable returned {name} which is not in runtime_args"
                )
        return result
