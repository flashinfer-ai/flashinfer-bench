from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution


@dataclass
class Runnable:
    """
    A built implementation that can be invoked directly.
    - entry: callable entry point; signature will align with Definition inputs/outputs
    - meta: metadata for build information/flags/versions etc.
    """

    entry: Callable[..., Any]
    meta: dict[str, Any]


# Backward-compat alias for early adopters during refactor
RunnableSolution = Runnable


class Builder(ABC):
    """Standardized builder abstraction: Solution -> Runnable"""

    @abstractmethod
    def can_build(self, sol: Solution) -> bool:  # pragma: no cover - skeleton
        ...

    @abstractmethod
    def fingerprint(self, sol: Solution) -> str:  # pragma: no cover - skeleton
        """A fingerprint for cache and reuse (not strictly stable)."""
        ...

    @abstractmethod
    def build(self, defn: Definition, sol: Solution) -> Runnable:  # pragma: no cover - skeleton
        """Build/load and return an invokable entry point."""
        ...
