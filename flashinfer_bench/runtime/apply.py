from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from flashinfer_bench.trace_set import TraceSet


class ApplyRuntime:
    _instance: Optional["ApplyRuntime"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.apply_enabled: bool = bool(os.getenv("ENABLE_FIB_APPLY"))
        self.tracing_enabled: bool = bool(os.getenv("ENABLE_FIB_TRACING"))
        self.root: str = os.getenv("FIB_DATASET_PATH")
        if self.apply_enabled and not self.root:
            raise ValueError("FIB_DATASET_PATH is not set")

        self._ts: Optional[TraceSet] = None
        # Apply best op cache: (def_name, tuple(sorted(axis→val))) → callable
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], Callable] = {}
        # Stores what inputs have variable axes in the shape: def_name → tuple(input_name, dim_idx, axis)
        self._var_axes_table: Dict[str, Tuple[Tuple[str, int, str], ...]] = {}

        self._logger = logging.getLogger(__name__)

    def resolve(
        self,
        def_name: str,
        runtime_args: Dict[str, Any],
        fallback: Callable,
        max_abs_diff: float = 1e-5,
        max_rel_diff: float = 1e-5,
    ) -> Callable:
        self._logger.info(f"Resolving '{def_name}'")
        self._ensure_traceset()

        if def_name not in self._ts.definitions:
            self._logger.error(f"Definition '{def_name}' not found in traceset")
            return fallback

        try:
            axes = self.infer_axes(def_name, runtime_args)
        except Exception as e:
            self._logger.error(f"Error inferring axes for definition '{def_name}': {e}")
            return fallback

        cache_key = (def_name, tuple(sorted(axes.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        best = self._ts.get_best_op(def_name, axes, max_abs_diff, max_rel_diff)
        chosen: Callable = best or fallback

        self._cache[cache_key] = chosen
        return chosen

    def _ensure_traceset(self):
        if self._ts is None:
            self._ts = TraceSet.from_path(self.root)

        if not self._var_axes_table:
            for defn in self._ts.definitions.values():
                axes = []
                for inp_name, inp_spec in defn.inputs.items():
                    for dim_idx, axis in enumerate(inp_spec["shape"]):
                        if defn.axes[axis]["type"] == "var":
                            axes.append((inp_name, dim_idx, axis))
                self._var_axes_table[defn.name] = tuple(axes)

    @property
    def traceset(self) -> Optional[TraceSet]:
        """Get the current traceset for validation purposes."""
        return self._ts

    def infer_axes(self, def_name: str, runtime_args: Dict[str, Any]) -> Dict[str, int]:
        """
        Use input shape in definition + runtime tensor shapes to determine
        concrete values for every variable axis.
        """
        recs = self._var_axes_table.get(def_name)
        if not recs:
            return {}

        axes = {}
        for inp_name, dim_idx, axis in recs:
            tensor = runtime_args.get(inp_name)
            if tensor is None:
                raise ValueError(f"Missing input '{inp_name}' for definition '{def_name}'")
            if dim_idx >= len(tensor.shape):
                raise ValueError(
                    f"Input '{inp_name}' rank {len(tensor.shape)} < dim {dim_idx} expected by '{axis}'"
                )
            axes[axis] = int(tensor.shape[dim_idx])
        return axes


_runtime = ApplyRuntime()


def apply(def_name_fn: Callable[..., str]) -> Callable[[Callable], Callable]:
    """
    Parameters
    ----------
    def_name_fn : (*args, **kw) -> str
        Function that maps runtime arguments → Definition name.

    Usage
    -----
    >>> @flashinfer_bench.apply(lambda A, B: f"gemm_n_{B.shape[0]}_k_{B.shape[1]}")
    ... def gemm_bf16(A, B, bias=None):
    ...     return _gemm_module.gemm_bf16(A, B, bias)
    """

    def decorator(fallback: Callable) -> Callable:
        sig = inspect.signature(fallback)

        @functools.wraps(fallback)
        def wrapped(*args: Any, **kwargs: Any):
            bound = sig.bind_partial(*args, **kwargs)
            def_name = def_name_fn(*args, **kwargs)

            if _runtime.tracing_enabled:
                from flashinfer_bench.tracer import get_tracer

                tracer = get_tracer()
                tracer.collect(def_name, bound.arguments)

            if _runtime.apply_enabled:
                impl = _runtime.resolve(def_name, bound.arguments, fallback)
            else:
                impl = fallback

            return impl(*args, **kwargs)

        return wrapped

    return decorator
