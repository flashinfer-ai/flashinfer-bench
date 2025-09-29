from typing import Dict, Optional

from flashinfer_bench.data import TraceSet
from flashinfer_bench.logging import get_logger

from .tracing_config import TracingConfig
from .tracing_runtime import TracingRuntime, get_tracing_runtime, set_tracing_runtime

logger = get_logger("Tracing")


def enable_tracing(
    dataset_path: Optional[str] = None, tracing_configs: Optional[Dict[str, TracingConfig]] = None
) -> TracingRuntime:
    """
    Enable tracing with the given tracing config set.

    Creates or replaces the process-wide singleton tracing runtime.
    If replacing, flushes the previous instance first.

    Args:
        dataset_path: Path to the dataset/traceset directory
        tracing_configs: A set of tracing configs. Default is `tracing_configs.fib_full_tracing`

    Returns:
        The new tracing runtime instance

    Examples
    --------
    >>> enable_tracing("/path/to/traceset")
    >>> # Tracing is now enabled
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_tracing()
    >>> # Tracing is now disabled.

    >>> # Context manager usage
    >>> with enable_tracing("/path/to/traceset"):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> # Tracing is now disabled.
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

    Examples
    --------
    >>> disable_tracing()
    >>> # Tracing is now disabled.
    """
    # Flush the current tracing runtime if it exists
    tracing_runtime = get_tracing_runtime()

    if tracing_runtime is not None:
        try:
            tracing_runtime.flush()
        except Exception as e:
            logger.error(f"Cannot flush existing tracing runtime: {e}, ignoring")

    set_tracing_runtime(None)
