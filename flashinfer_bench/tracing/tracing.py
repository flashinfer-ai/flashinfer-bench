"""Public API for enabling and disabling tracing."""

import logging
from typing import Dict, Optional, Union

from flashinfer_bench.data import TraceSet

from .config import TracingConfig, TracingConfigRegistry
from .runtime import TracingRuntime

logger = logging.getLogger(__name__)


def enable_tracing(
    dataset_path: Optional[str] = None,
    tracing_config: Union[
        TracingConfigRegistry, TracingConfig, Dict[str, TracingConfig], None
    ] = None,
) -> TracingRuntime:
    """Enable tracing by creating and pushing a new tracing runtime onto the global stack.

    The returned runtime can be used as a context manager to automatically
    flush and pop from the stack on exit.

    The tracing runtime is process-level. This function is recommended to be called in the main
    thread.

    Parameters
    ----------
    dataset_path : Optional[str]
        Path to the dataset/trace_set directory. If None, uses the environment
        variable FIB_DATASET_PATH or defaults to `~/.cache/flashinfer_bench/dataset`.
    tracing_config : Union[TracingConfigRegistry, TracingConfig, \
            Dict[str, TracingConfig], None], optional
        Configuration for the tracing runtime. Can be:

        - TracingConfig: A single config used as the default for all definitions.
        - Dict[str, TracingConfig]: A dict mapping the names of the kernels to trace and their
          configs.
        - TracingConfigRegistry: A registry with per-definition configs.
        - None: Uses the full configs preset.

    Returns
    -------
    TracingRuntime
        The newly created tracing runtime instance that has been pushed onto
        the global stack.

    Examples
    --------
    Basic usage with manual disable:

    >>> enable_tracing("/path/to/trace_set")
    >>> # Tracing is now enabled
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_tracing()
    >>> # Tracing is now disabled

    Context manager usage (recommended):

    >>> with enable_tracing("/path/to/trace_set"):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> # Tracing is automatically flushed and disabled

    Custom tracing configuration:

    >>> from flashinfer_bench.tracing import TracingConfig
    >>> configs = {
    ...     "rmsnorm_d4096": TracingConfig(input_dump_policy=["x", "weight"]),
    ...     "silu_and_mul": TracingConfig(
    ...         filter_policy="keep_first_k", filter_policy_kwargs={"k": 10}
    ...     ),
    ... }
    >>> with enable_tracing("/path/to/trace_set", configs):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)

    Nested tracing with different configs:

    >>> with enable_tracing("/path/to/trace_set", config_a):
    ...     # Tracing with config_a
    ...     with enable_tracing("/path/to/trace_set", config_b):
    ...         # Tracing with config_b (config_a is paused)
    ...     # Back to config_a
    """
    trace_set = TraceSet.from_path(dataset_path)
    runtime = TracingRuntime(trace_set, tracing_config)
    runtime.start()
    return runtime


def disable_tracing():
    """Disable tracing by popping the current runtime from the stack and flushing.

    Pops the top runtime from the global stack, flushes all buffered trace
    entries to disk, and restores the previous runtime (if any) as the active
    instance. Safe to call even if no tracing runtime is active.

    Notes
    -----
    This function logs errors but does not raise exceptions if flushing fails.
    When using enable_tracing() as a context manager, this is called automatically
    on exit.

    Examples
    --------
    Manual disable after enable:

    >>> enable_tracing("/path/to/trace_set")
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_tracing()
    >>> # Tracing is now disabled and all traces are flushed to disk

    Safe to call when tracing is not enabled:

    >>> disable_tracing()  # No-op if tracing is not active
    """
    tracing_runtime = TracingRuntime.get_instance()

    if tracing_runtime is not None:
        try:
            tracing_runtime.stop()
        except Exception as e:
            logger.error(f"Cannot stop existing tracing runtime: {e}, ignoring")
