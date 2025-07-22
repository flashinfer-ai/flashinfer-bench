from __future__ import annotations
import inspect, os, functools
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
        self.enabled: bool = bool(os.getenv("ENABLE_FLASHINFER_APPLY"))
        self.root: str = os.getenv("FLASHINFER_BENCH_PATH")
        if self.enabled and not self.root:
            raise ValueError("FLASHINFER_BENCH_PATH is not set")

        self._ts: Optional[TraceSet] = None
        # (def_name, tuple(sorted(axis→val))) → callable
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], Callable] = {}
        # def_name → tuple(input_name, dim_idx, axis)
        self._var_axes_table: Dict[str, Tuple[Tuple[str, int, str], ...]] = {}

    def resolve(
        self,
        def_name: str,
        runtime_args: Dict[str, Any],
        fallback: Callable,
        max_abs_diff: float = 1e-5,
        max_rel_diff: float = 1e-5,
    ) -> Callable:
        if not self.enabled:
            return fallback

        self._ensure_traceset()

        if def_name not in self._ts.definitions:
            print(f"Definition '{def_name}' not found in traceset")
            return fallback

        try:
            axes = self._infer_axes(def_name, runtime_args)
        except Exception as e:
            print(f"Error inferring axes for definition '{def_name}': {e}")
            return fallback

        cache_key = (def_name, tuple(sorted(axes.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        chosen = self._ts.get_best_op(def_name, axes, max_abs_diff, max_rel_diff) or fallback
        self._cache[cache_key] = chosen
        return chosen

    def _ensure_traceset(self):
        if self._ts is None:
            self._ts = TraceSet.from_path(self.root)

        if self._var_axes_table is None:
            for defn in self._ts.definitions.values():
                axes = []
                for inp_name, inp_spec in defn.inputs.items():
                    for dim_idx, axis in enumerate(inp_spec["shape"]):
                        if defn.axes[axis]["type"] == "var":
                            axes.append((inp_name, dim_idx, axis))
                self._var_axes_table[defn.name] = tuple(axes)


    def _infer_axes(self, def_name: str, runtime_args: Dict[str, Any]) -> Dict[str, int]:
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
    >>> @flashinfer_bench.apply(lambda A, B: f"gemm_m_dynamic_n_{B.shape[0]}_k_{B.shape[1]}")
    ... def gemm_bf16(A, B, bias=None):
    ...     return _gemm_module.gemm_bf16(A, B, bias)
    """
    def decorator(fallback: Callable) -> Callable:
        sig = inspect.signature(fallback)

        @functools.wraps(fallback)
        def wrapped(*args: Any, **kwargs: Any):
            bound = sig.bind_partial(*args, **kwargs)
            def_name = def_name_fn(*args, **kwargs)
            impl = _runtime.resolve(def_name, bound.arguments, fallback)
            return impl(*args, **kwargs)

        return wrapped

    return decorator