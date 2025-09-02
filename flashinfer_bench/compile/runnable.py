from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Runnable:
    def __init__(
        self,
        fn: Callable[..., Any],
        closer: Callable[[], None],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A runnable callable with a required resource closer.

        closer: must be provided by the builder and be idempotent.
        """
        self._fn = fn
        self._closer: Optional[Callable[[], None]] = closer
        self.meta: Dict[str, Any] = dict(meta or {})

    def __call__(self, **kwargs: Any) -> Any:
        """
        - Accept kwargs only (aligns with Definition.inputs naming)
        - Unpack a single-element tuple to a scalar value
        - No type/shape/count validation; errors surface naturally
        """
        ret = self._fn(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 1:
            return ret[0]
        return ret

    def close(self) -> None:
        """Release build artifacts/resources; must be idempotent."""
        if self._closer:
            try:
                self._closer()
            finally:
                self._closer = None
