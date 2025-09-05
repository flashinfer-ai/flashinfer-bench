from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Union


@dataclass
class TracingRule:
    """Lightweight rule definition usable for static typing and imports."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    dedup_policy: Callable[[List["TraceEntry"]], List["TraceEntry"]]
    dedup_keys: Optional[Callable[["TraceEntry"], Hashable]] = None


# Forward-declared type for typing tools; actual class is defined in tracer.py
class TraceEntry:  # pragma: no cover - typing helper only
    ...
