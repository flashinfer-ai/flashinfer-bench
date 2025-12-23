"""Runtime system for applying optimized implementations based on trace data.

This module provides the core runtime infrastructure for the FlashInfer benchmark apply system.
It manages the lifecycle of optimized implementations, handles dispatch logic, and provides
hooks for tracing and monitoring. The runtime uses trace data to select the best performing
implementation for each function call based on workload characteristics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import TraceSet
from flashinfer_bench.env import get_fib_dataset_path, get_fib_enable_apply
from flashinfer_bench.logging import get_logger

from .config import ApplyConfig
from .key import ApplyKeyBuilder, ApplyKeyFactory
from .table import ApplyTable

logger = get_logger("ApplyRuntime")


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
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        fallback: Optional[Callable[..., Any]],
    ) -> Any:
        """Dispatch a function call to the optimal implementation.

        Selects and executes the best performing implementation for the given
        function and workload parameters. Uses trace data to determine which
        implementation will perform best for the specific workload characteristics.

        The calling convention is determined by comparing the number of arguments
        to the definition's inputs and outputs:
        - If len(args) == len(inputs): value-returning style
        - If len(args) == len(inputs) + len(outputs): destination-passing style

        Parameters
        ----------
        def_name : str
            Name of the function definition to dispatch.
        args : Tuple[Any, ...]
            Positional arguments. For value-returning style, only inputs.
            For destination-passing style, inputs followed by pre-allocated outputs.
        kwargs : Dict[str, Any]
            Keyword arguments to merge into positional arguments.
        fallback : Optional[Callable[..., Any]]
            Fallback function to call if no optimized implementation is available.

        Returns
        -------
        Any
            The result of executing the selected implementation.
            For value-returning style: the output tensor(s).
            For destination-passing style: None.

        Raises
        ------
        RuntimeError
            If the definition is not found and no fallback is provided, or if
            no suitable implementation is available and no fallback is provided.
        """
        definition = self._trace_set.definitions.get(def_name)
        if definition is None:
            if fallback is None:
                raise RuntimeError(f"Definition '{def_name}' not found and no fallback provided")
            return fallback(*args, **kwargs)

        # Merge kwargs into args based on definition's input/output order
        # Use a new variable to preserve original args/kwargs for fallback calls
        merged_args = args
        if kwargs:
            merged_args = definition.merge_kwargs_to_args(args, kwargs)

        # Determine calling convention based on argument count
        num_inputs = len(definition.inputs)
        num_outputs = len(definition.outputs)
        is_dps = len(merged_args) == num_inputs + num_outputs

        if not is_dps and len(merged_args) != num_inputs:
            raise ValueError(
                f"Invalid number of arguments for '{def_name}': expected {num_inputs} "
                f"(value-returning) or {num_inputs + num_outputs} (destination-passing), "
                f"got {len(merged_args)}"
            )

        # Extract only inputs for key building
        input_args = merged_args[:num_inputs]

        # Build key from input arguments
        builder = self._key_builders.get(definition.name)
        if builder is None:
            builder = ApplyKeyFactory.specialize(definition)
            self._key_builders[definition.name] = builder
        key = builder.build_from_args(input_args)

        # Lookup solution
        sol_name = self._table.match_solution(def_name, key)
        runnable = None
        if sol_name:
            solution = self._trace_set.get_solution(sol_name)
            if solution:
                runnable = BuilderRegistry.get_instance().build(definition, solution)

        # Miss policy
        if runnable is None:
            if self._apply_config.on_miss_policy == "use_def_best":
                best_sol_name = self._table.def_best.get(def_name)
                solution = self._trace_set.get_solution(best_sol_name)
                if definition and solution:
                    runnable = BuilderRegistry.get_instance().build(definition, solution)

        if runnable is None:
            if fallback is None:
                raise RuntimeError(f"Apply miss for '{def_name}' and no fallback provided")
            return fallback(*args, **kwargs)

        # Call the runnable with appropriate style
        if is_dps:
            runnable.call_destination_passing(*merged_args)
            return None
        else:
            return runnable.call_value_returning(*input_args)

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        set_apply_runtime(self._prev_runtime)
        return False
