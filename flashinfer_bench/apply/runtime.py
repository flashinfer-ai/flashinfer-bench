"""Runtime system for applying optimized implementations based on trace data.

This module provides the core runtime infrastructure for the FlashInfer benchmark apply system.
It manages the lifecycle of optimized implementations, handles dispatch logic, and provides
hooks for tracing and monitoring. The runtime uses trace data to select the best performing
implementation for each function call based on workload characteristics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from flashinfer_bench.compile import get_builder_registry
from flashinfer_bench.data import TraceSet
from flashinfer_bench.env import get_fib_dataset_path, get_fib_enable_apply
from flashinfer_bench.logging import get_logger

from .config import ApplyConfig
from .key import ApplyKeyBuilder, ApplyKeyFactory
from .table import ApplyTable

logger = get_logger("ApplyRuntime")


class ApplyRuntime:
    """Runtime system for dispatching optimized implementations based on trace data.

    The ApplyRuntime manages a collection of optimized implementations and selects
    the best one for each function call based on workload characteristics. It uses
    precomputed trace data to build lookup tables that map workload parameters to
    the most efficient implementation.

    The runtime supports fallback mechanisms when no optimized implementation is
    available and provides hooks for monitoring and tracing function calls.
    """

    def __init__(
        self,
        trace_set: TraceSet,
        apply_config: Optional[ApplyConfig] = None,
        prev_apply_runtime: Optional["ApplyRuntime"] = None,
    ) -> None:
        """Initialize the apply runtime.

        Parameters
        ----------
        trace_set : TraceSet
            A TraceSet object.
        apply_config : ApplyConfig
            Configuration object specifying runtime behavior and policies.
        prev_apply_runtime : Optional[ApplyRuntime], optional
            The previous apply runtime. Will be used in the __exit__ method. Default is None.
        """
        self._trace_set = trace_set
        self._apply_config = apply_config if apply_config is not None else ApplyConfig()
        self._prev_runtime = prev_apply_runtime

        self._table = ApplyTable.load_or_build(self._trace_set, self._apply_config)

        # def_name -> callable: (runtime_kwargs) -> ApplyKey
        self._key_builders: Dict[str, ApplyKeyBuilder] = {}

        # Install integrations
        from flashinfer_bench.integration.flashinfer import install_flashinfer_integrations

        install_flashinfer_integrations()

    def dispatch(
        self,
        def_name: str,
        runtime_kwargs: Mapping[str, Any],
        fallback: Optional[Callable[..., Any]],
    ) -> Any:
        """Dispatch a function call to the optimal implementation.

        Selects and executes the best performing implementation for the given
        function and workload parameters. Uses trace data to determine which
        implementation will perform best for the specific workload characteristics.

        The dispatch process follows these steps:
        1. Call any registered hooks for monitoring/tracing
        2. Look up the function definition in the trace dataset
        3. Build a workload key from the runtime parameters
        4. Find the optimal solution using the lookup table
        5. Build and execute the optimized implementation
        6. Fall back to default implementation or best available if needed

        Parameters
        ----------
        def_name : str
            Name of the function definition to dispatch.
        runtime_kwargs : Mapping[str, Any]
            Runtime parameters that characterize the workload.
        fallback : Optional[Callable[..., Any]]
            Fallback function to call if no optimized implementation is available.

        Returns
        -------
        Any
            The result of executing the selected implementation.

        Raises
        ------
        RuntimeError
            If the definition is not found and no fallback is provided, or if
            no suitable implementation is available and no fallback is provided.
        """
        defn = self._trace_set.definitions.get(def_name)
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
            sol = self._trace_set.get_solution(sol_name)
            if sol:
                runnable = get_builder_registry().build(defn, sol)

        # Miss policy
        if runnable is None:
            if self._apply_config.on_miss_policy == "use_def_best":
                best_sol_name = self._table.def_best.get(def_name)
                sol = self._trace_set.get_solution(best_sol_name)
                if defn and sol:
                    runnable = get_builder_registry().build(defn, sol)
                if runnable is not None:
                    return runnable(**runtime_kwargs)
            if fallback is None:
                raise RuntimeError(f"Apply miss for '{def_name}' and no fallback provided")
            return fallback(**runtime_kwargs)

        return runnable(**runtime_kwargs)

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        set_apply_runtime(self._prev_runtime)
        return False


def _init_apply_runtime_from_env() -> Optional["ApplyRuntime"]:
    """Initialize the global runtime from environment variables if configured."""
    fib_enable_apply = get_fib_enable_apply()
    if not fib_enable_apply:
        return
    fib_dataset_path = get_fib_dataset_path()
    trace_set = TraceSet.from_path(fib_dataset_path)
    apply_config = ApplyConfig()
    return ApplyRuntime(trace_set, apply_config, None)


_global_apply_runtime: Optional["ApplyRuntime"] = _init_apply_runtime_from_env()


def get_apply_runtime() -> Optional["ApplyRuntime"]:
    """Get the global ApplyRuntime instance.

    Returns the singleton runtime instance, initializing it from environment
    variables if it hasn't been created yet.

    Returns
    -------
    Optional[ApplyRuntime]
        The global runtime instance, or None if not initialized.
    """
    return _global_apply_runtime


def set_apply_runtime(rt: Optional["ApplyRuntime"]) -> None:
    """Set the global ApplyRuntime instance.

    Parameters
    ----------
    rt : Optional[ApplyRuntime]
        The runtime instance to set, or None to clear the global runtime.
    """
    global _global_apply_runtime
    _global_apply_runtime = rt
