from __future__ import annotations

import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data import Definition, Solution, SourceFile


def write_sources_to_dir(dir: str, sources: list[SourceFile]) -> None:
    os.makedirs(dir, exist_ok=True)
    for src in sources:
        abspath = os.path.join(dir, src.path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", encoding="utf-8") as f:
            f.write(src.content)


def write_sources_to_temp(base: str, sources: list[SourceFile], pkg: Optional[str] = None) -> str:
    os.makedirs(base, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=base)
    if pkg:
        tmpdir = os.path.join(tmpdir, pkg)
        os.makedirs(tmpdir, exist_ok=True)
    write_sources_to_dir(tmpdir, sources)
    return tmpdir


def create_pkg_name(sol: Solution, prefix: str = "") -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", sol.name)
    if not s or s[0].isdigit():
        s = "_" + s
    return prefix + s


class BuildError(RuntimeError):
    """Raised when a builder fails to construct a runnable implementation."""


class Builder(ABC):
    """Builder abstraction: (Definition, Solution) -> Runnable with hidden cache."""

    def __init__(self) -> None:
        self._cache: Dict[str, Runnable] = {}

    @abstractmethod
    def can_build(self, solution: Solution) -> bool:
        """Build guard to check if this builder can handle the given solution."""
        ...

    @abstractmethod
    def _build(self, definition: Definition, solution: Solution) -> Runnable:
        """Perform a real build and return a Runnable; raise BuildError on failure."""
        ...

    @abstractmethod
    def _make_closer(self, *args, **kwargs) -> Callable[[], None]:
        """Factory for a resource closer used by the concrete builder."""
        ...

    @abstractmethod
    def _make_key(self, solution: Solution) -> str:
        """Cache key for a solution."""
        ...

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Public entry with per-solution cache keyed by solution.name."""
        key = self._make_key(solution)
        if key in self._cache:
            return self._cache[key]
        runnable = self._build(definition, solution)
        self._cache[key] = runnable
        return runnable

    def clear_cache(self) -> None:
        """Close all cached runnables and clear the cache."""
        for r in list(self._cache.values()):
            try:
                r.close()
            except Exception:
                # Best-effort cleanup; keep going
                pass
        self._cache.clear()
