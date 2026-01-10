"""Patch manager for monkey-patching external library functions."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

Kind = Literal["function", "method", "callable"]


@dataclass
class PatchSpec:
    path: str
    kind: Kind
    name: str
    ctx_key: Optional[str] = None


class PatchManager:
    """Responsible for: resolve target, replace attr, restore original."""

    def __init__(self) -> None:
        # (owner_obj, attr_name) -> original
        self._originals: Dict[Tuple[object, str], Any] = {}

    def _resolve(self, path: str) -> Tuple[object, str, Any]:
        """
        Resolve a dotted path to (owner, attr, original_attr).
        Works for module functions or class attributes (methods).
        """
        parts = path.split(".")
        # greedily import the longest module prefix
        for i in range(len(parts), 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                mod = importlib.import_module(mod_name)
                owner: object = mod
                rest = parts[i:]
                break
            except Exception:
                continue
        else:
            raise ImportError(f"Cannot import any module from '{path}'")

        # descend attributes to find owner of the final attribute
        for j in range(len(rest) - 1):
            owner = getattr(owner, rest[j])

        attr_name = rest[-1] if rest else None
        if attr_name is None:
            raise AttributeError(f"Path '{path}' has no attribute segment")

        original = getattr(owner, attr_name)
        return owner, attr_name, original

    def patch(
        self,
        spec: PatchSpec,
        wrapper_factory: Callable[[PatchSpec, Callable[..., Any]], Callable[..., Any]],
    ) -> bool:
        """Install a wrapper on target; return True if success, False if target missing."""
        try:
            owner, attr, original = self._resolve(spec.path)
        except Exception:
            return False  # target not present in this env; silently ignore

        key = (owner, attr)
        if key in self._originals:
            return True  # already patched (idempotent)

        wrapper = wrapper_factory(spec, original)
        setattr(owner, attr, wrapper)
        self._originals[key] = original
        return True

    def unpatch_all(self) -> None:
        """Restore all originals."""
        for (owner, attr), original in list(self._originals.items()):
            try:
                setattr(owner, attr, original)
            except Exception:
                pass
        self._originals.clear()


_manager = PatchManager()


def get_manager() -> PatchManager:
    return _manager
