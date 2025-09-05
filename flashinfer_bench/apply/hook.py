from typing import Any, Callable, Mapping, Optional

# Global function pointer hook: (def_name, runtime_kwargs) -> None
_hook: Optional[Callable[[str, Mapping[str, Any]], None]] = None


def set_apply_hook(fn: Optional[Callable[[str, Mapping[str, Any]], None]]) -> None:
    global _hook
    _hook = fn


def get_apply_hook() -> Callable[[str, Mapping[str, Any]], None]:
    def _noop(def_name: str, runtime_kwargs: Mapping[str, Any]) -> None:  # type: ignore
        return None

    return _hook or _noop
