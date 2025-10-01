from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Union


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


# Type alias for dedup policy factory function
DedupPolicyLiteral = Literal["keep_all", "keep_first", "dedup_by_axes"]
"""Possible dedup policy literals. See builtin_config.py for more the implementation of these
policies."""

DedupPolicyFactory = Callable[[], DedupPolicy]
"""Factory function for dedup policy."""


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    """Which inputs to persist. List[str] for static selection, Callable for dynamic."""

    dedup_policy: Union[DedupPolicyLiteral, DedupPolicyFactory]
    """Deduplication policy factory. Can be a factory function or a string literal for built-in policies."""

    def __post_init__(self):
        """Convert literal dedup policy strings to factory functions."""
        if isinstance(self.dedup_policy, str):
            # Lazy import to avoid circular dependency
            from .builtin_config import BUILTIN_DEDUP_POLICY_FACTORIES

            factory = BUILTIN_DEDUP_POLICY_FACTORIES.get(self.dedup_policy)
            if factory is None:
                raise ValueError(
                    f"Unknown dedup_policy literal: {self.dedup_policy}. "
                    f"Must be one of {list(BUILTIN_DEDUP_POLICY_FACTORIES.keys())}"
                )
            self.dedup_policy = factory

    def create_dedup_policy(self) -> DedupPolicy:
        """Create a new dedup policy instance.

        Returns
        -------
        DedupPolicy
            A new policy instance with independent state.
        """
        if callable(self.dedup_policy):
            return self.dedup_policy()
        else:
            raise TypeError(
                f"dedup_policy must be callable after __post_init__, got {type(self.dedup_policy)}"
            )

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
