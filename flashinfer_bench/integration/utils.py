"""Utility classes for integration adapters."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Tuple
from weakref import WeakKeyDictionary


class ArgBinder:
    """Cache inspect.signature and bind once per callable."""

    def __init__(self, fn) -> None:
        self._sig = inspect.signature(fn)

    @classmethod
    def from_callable(cls, fn) -> "ArgBinder":
        return cls(fn)

    def bind(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        ba = self._sig.bind_partial(*args, **kwargs)
        ba.apply_defaults()
        return dict(ba.arguments)


class ContextStore:
    """Per-instance loose store; adapter decides fields."""

    def __init__(self) -> None:
        self._store: "WeakKeyDictionary[object, Dict[str, Any]]" = WeakKeyDictionary()

    def get(self, inst: object) -> Dict[str, Any]:
        d = self._store.get(inst)
        if d is None:
            d = {}
            self._store[inst] = d
        return d
