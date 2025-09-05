from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Set, Union


@dataclass
class TracingRule:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    """Which inputs to persist. List[str] for static selection, Callable for dynamic."""

    dedup_policy: Callable[[List["TraceEntry"]], List["TraceEntry"]]
    """Final in-group deduplication decision. Returns the representatives."""

    dedup_keys: Optional[Callable[["TraceEntry"], Hashable]] = None
    """Blocking function for candidate partitioning during dedup."""


@dataclass
class TraceEntry:
    """In-memory buffer entry for collected workloads."""

    def_name: str
    axes: Dict[str, int]
    definition_input_names: Set[str]
    picked: Dict[str, Any]
    order: int
    cuda_graph_snapshot: Optional[Dict[str, Any]] = None
