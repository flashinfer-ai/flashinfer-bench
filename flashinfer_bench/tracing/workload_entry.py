"""Collected workload entries."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
