from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Mapping


@dataclass(frozen=True)
class ApplyEvent:
    definition: str
    runtime_args: Mapping[str, Any]


class HookManager:
    """Placeholder: global on_apply event management."""

    def __init__(self):
        self._lock = RLock()
        self._listeners: Dict[str, Callable[[ApplyEvent], None]] = {}

    def register_apply_listener(self, cb: Callable[[ApplyEvent], None]) -> str:
        raise NotImplementedError

    def unregister(self, token: str) -> None:
        raise NotImplementedError

    def emit_apply(self, event: ApplyEvent) -> None:
        raise NotImplementedError


# Singleton
hook_manager = HookManager()
