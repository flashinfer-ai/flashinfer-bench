from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Union


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


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    """Which inputs to persist. List[str] for static selection, Callable for dynamic."""

    dedup_policy: Callable[[List["WorkloadEntry"]], List["WorkloadEntry"]]
    """Final in-group deduplication decision. Returns the representatives."""

    dedup_keys: Optional[Callable[["WorkloadEntry"], Hashable]] = None
    """Blocking function for candidate partitioning during dedup."""

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
