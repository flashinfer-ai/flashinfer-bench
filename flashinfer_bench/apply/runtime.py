from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Optional, Union

from flashinfer_bench.compile.registry import get_registry
from flashinfer_bench.data.traceset import TraceSet

from .config import ApplyConfig
from .hook import get_apply_hook
from .key import ApplyKeyBuilder, ApplyKeyFactory
from .table import ApplyTable

_runtime: Optional["ApplyRuntime"] = None


def get_runtime() -> "ApplyRuntime":
    global _runtime
    if _runtime is None:
        _maybe_init_from_env()
    return _runtime


def set_runtime(rt: Optional["ApplyRuntime"]) -> None:
    global _runtime
    _runtime = rt


def _maybe_init_from_env() -> None:
    global _runtime
    enable_apply = os.environ.get("FIB_ENABLE_APPLY")
    enable_tracing = os.environ.get("FIB_ENABLE_TRACING")
    dataset = os.environ.get("FIB_DATASET_PATH")

    if enable_apply is None or dataset is None:
        return

    rt = ApplyRuntime(dataset, ApplyConfig())

    # Optionally install tracing hook
    if enable_tracing:
        try:
            from flashinfer_bench.tracer import enable_tracing, make_tracing_hook

            from .hook import set_apply_hook

            tracer = enable_tracing()  # default to fib_full_tracing rules
            set_apply_hook(make_tracing_hook(tracer))
        except Exception as e:
            raise RuntimeError(f"Failed to enable tracing hook: {e}")

    _runtime = rt


class ApplyRuntime:
    def __init__(self, traceset: Union[TraceSet, str], config: ApplyConfig) -> None:
        if not traceset:
            raise ValueError("Dataset path is required for enabling apply")

        if isinstance(traceset, str):
            self._traceset = TraceSet.from_path(traceset)
        else:
            self._traceset = traceset

        self._config: ApplyConfig = config
        self._table: ApplyTable = ApplyTable.load_or_build(self._traceset, self._config)
        # def_name -> callable: (runtime_kwargs) -> ApplyKey
        self._key_builders: Dict[str, ApplyKeyBuilder] = {}

        # Install integrations
        from flashinfer_bench.integration.flashinfer import install_flashinfer_integrations

        install_flashinfer_integrations()

    def rebuild(
        self,
        traceset: Optional[Union[TraceSet, str]] = None,
        config: Optional[ApplyConfig] = None,
    ) -> None:
        if traceset is not None:
            if isinstance(traceset, str):
                self._traceset = TraceSet.from_path(traceset)
            else:
                self._traceset = traceset
        if config is not None:
            self._config = config
        self._table = ApplyTable.load_or_build(self._traceset, self._config)

    def dispatch(
        self,
        def_name: str,
        runtime_kwargs: Mapping[str, Any],
        fallback: Optional[Callable[..., Any]],
    ) -> Any:
        # Hook (no-op by default)
        hook = get_apply_hook()
        try:
            hook(def_name, runtime_kwargs)
        except Exception:
            pass

        defn = self._traceset.definitions.get(def_name)
        if defn is None:
            if fallback is None:
                raise RuntimeError(f"Definition '{def_name}' not found and no fallback provided")
            return fallback(**runtime_kwargs)

        # Build key
        builder = self._key_builders.get(defn.name)
        if builder is None:
            builder = ApplyKeyFactory.specialize(defn)
            self._key_builders[defn.name] = builder
        key = builder.build_from_runtime(runtime_kwargs)

        # Lookup solution
        sol_name = self._table.match_solution(def_name, key)
        runnable = None
        if sol_name:
            sol = self._traceset.get_solution(sol_name)
            if sol:
                runnable = get_registry().build(defn, sol)

        # Miss policy
        if runnable is None:
            if self._config.on_miss_policy == "use_def_best":
                best_sol_name = self._table.def_best.get(def_name)
                sol = self._traceset.get_solution(best_sol_name)
                if defn and sol:
                    runnable = get_registry().build(defn, sol)
                if runnable is not None:
                    return runnable(**runtime_kwargs)
            if fallback is None:
                raise RuntimeError(f"Apply miss for '{def_name}' and no fallback provided")
            return fallback(**runtime_kwargs)

        return runnable(**runtime_kwargs)
