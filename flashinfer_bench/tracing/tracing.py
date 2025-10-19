from typing import Dict, Optional

from flashinfer_bench.data import TraceSet
from flashinfer_bench.logging import get_logger
from flashinfer_bench.tracing.config import TracingConfig
from flashinfer_bench.tracing.runtime import (
    TracingRuntime,
    get_tracing_runtime,
    set_tracing_runtime,
)

logger = get_logger("Tracing")


def enable_tracing(
    dataset_path: Optional[str] = None, tracing_configs: Optional[Dict[str, TracingConfig]] = None
) -> TracingRuntime:
    """Enable tracing with the given tracing config set.

    Creates or replaces the process-wide singleton tracing runtime.
    If replacing, flushes the previous instance first. The returned runtime
    can be used as a context manager to automatically flush and restore the
    previous runtime on exit.

    Parameters
    ----------
    dataset_path : Optional[str]
        Path to the dataset/traceset directory. If None, uses the environment
        variable FIB_DATASET_PATH or defaults to `~/.cache/flashinfer_bench/dataset`.
    tracing_configs : Optional[Dict[str, TracingConfig]]
        Dictionary mapping definition names to their tracing configurations.
        If None, defaults to `fib.tracing.builtin_config.fib_full_tracing`.

    Returns
    -------
    TracingRuntime
        The newly created tracing runtime instance that has been set as the
        global runtime.

    Examples
    --------
    Basic usage with manual disable:

    >>> enable_tracing("/path/to/traceset")
    >>> # Tracing is now enabled
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_tracing()
    >>> # Tracing is now disabled

    Context manager usage (recommended):

    >>> with enable_tracing("/path/to/traceset"):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> # Tracing is automatically flushed and disabled

    Custom tracing configuration:

    >>> from flashinfer_bench.tracing import TracingConfig
    >>> configs = {"rmsnorm_d4096": TracingConfig(input_dump_policy=["x", "weight"], filter_policy="keep_all")}
    >>> with enable_tracing("/path/to/traceset", tracing_configs=configs):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    """

    prev_runtime = get_tracing_runtime()
    # Flush the previous runtime if it exists
    if prev_runtime is not None:
        prev_runtime.flush()
    trace_set = TraceSet.from_path(dataset_path)
    runtime = TracingRuntime(trace_set, tracing_configs, prev_runtime)
    set_tracing_runtime(runtime)
    return runtime


def disable_tracing():
    """Disable tracing and flush any pending data.

    Flushes the current global tracing runtime (if one exists) to persist
    all buffered trace entries to disk, then clears the global runtime
    instance. This is safe to call even if no tracing runtime is active.

    Notes
    -----
    This function logs errors but does not raise exceptions if flushing
    fails. The global runtime will be cleared regardless of flush status.
    When using enable_tracing() as a context manager, disable_tracing()
    is called automatically on exit.

    Examples
    --------
    Manual disable after enable:

    >>> enable_tracing("/path/to/traceset")
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_tracing()
    >>> # Tracing is now disabled and all traces are flushed to disk

    Safe to call when tracing is not enabled:

    >>> disable_tracing()  # No-op if tracing is not active
    """
    # Flush the current tracing runtime if it exists
    tracing_runtime = get_tracing_runtime()

    if tracing_runtime is not None:
        try:
            tracing_runtime.flush()
        except Exception as e:
            logger.error(f"Cannot flush existing tracing runtime: {e}, ignoring")

    set_tracing_runtime(None)
