from __future__ import annotations

from typing import Any, Mapping

from flashinfer_bench.apply.runtime import get_runtime


def make_tracing_hook(tracer):
    """Create a lightweight apply hook that forwards to tracer.collect.

    Hook signature: (def_name, runtime_kwargs) -> None
    """

    def _hook(def_name: str, runtime_kwargs: Mapping[str, Any]) -> None:
        try:
            # Ensure runtime is initialized and axes are computable
            rt = get_runtime()
            # This will raise if missing
            rt.infer_axes(def_name, runtime_kwargs)
        except Exception:
            # Silently ignore tracing on invalid inputs to avoid impacting hot path
            return
        try:
            tracer.collect(def_name, dict(runtime_kwargs))
        except Exception:
            # Do not propagate tracing errors to hot path
            return

    return _hook
