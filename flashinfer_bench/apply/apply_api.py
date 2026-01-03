"""Public API for the apply subsystem."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union, overload

from flashinfer_bench.data import TraceSet
from flashinfer_bench.tracing import get_tracing_runtime

from .config import ApplyConfig
from .runtime import ApplyRuntime, get_apply_runtime, set_apply_runtime


# Decorator mode
@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


# Function mode
@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    *,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    fallback: Optional[Callable[..., Any]] = None,
) -> Any: ...


def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    fallback: Optional[Callable[..., Any]] = None,
):
    """
    Decorator/function for routing to the best-performing kernel recorded in the
    FlashInfer Trace database.

    This API can be used in two modes:

    1) **Decorator mode** (only ``def_name_or_resolver`` provided): returns a decorator
       that wraps a kernel function with a router. The router selects the best-performing
       candidate according to the function's runtime arguments.
    2) **Function mode** (``args`` or ``kwargs`` provided, optionally ``fallback``):
       immediately resolves and calls the best-performing kernel and returns its result.

    The calling convention (value-returning vs destination-passing) is determined by the
    number of arguments:
    - If len(args) == len(inputs): value-returning style, solution returns outputs
    - If len(args) == len(inputs) + len(outputs): destination-passing style, outputs are
      pre-allocated and passed as arguments

    Parameters
    ----------
    def_name_or_resolver : Union[str, Callable[..., str]]
        The kernel name, or a resolver ``fn(*args, **kwargs) -> str`` that maps runtime
        arguments to a kernel name (definition name).
    args : Tuple[Any, ...], optional
        Only used in **function mode**. The positional runtime arguments to feed into
        the selected kernel. The number of arguments determines the calling convention.
    kwargs : Dict[str, Any], optional
        Only used in **function mode**. The keyword runtime arguments to feed into
        the selected kernel. The number of arguments determines the calling convention.
    fallback : Optional[Callable[..., Any]], optional
        Only used in **function mode**. A fallback function to invoke when no matching
        kernel is found in the Trace database.

    Returns
    -------
    Union[Callable[[Callable[..., Any]], Callable[..., Any]], Any]
        - **Decorator mode**: a decorator that transforms the target kernel function into
          a routed version.
        - **Function mode**: the return value produced by the selected (or fallback) kernel.
          For destination-passing style, returns None.

    Examples
    --------
    Decorator mode with a fixed name
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> @apply("gemm_bf16")
    ... def gemm_bf16(A, B):
    ...     return A @ B.T

    Decorator mode with a resolver
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> @apply(lambda A, B: f"gemm_n{B.shape[0]}_k{B.shape[1]}")
    ... def gemm_bf16(A, B):
    ...     return A @ B.T

    Function mode (value-returning)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> out = apply(
    ...     "gemm_bf16",
    ...     args=(A, B),
    ...     fallback=lambda A, B: A @ B.T,
    ... )

    Function mode (destination-passing)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    >>> apply(
    ...     "gemm_bf16",
    ...     args=(A, B, C),  # C is pre-allocated output
    ...     fallback=lambda *args: my_gemm_dps(*args),
    ... )

    Function mode with kwargs
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> out = apply(
    ...     "gemm_bf16",
    ...     kwargs={"A": A, "B": B},
    ...     fallback=lambda A, B: A @ B.T,
    ... )
    """
    # Imperative / Function mode
    if args is not None or kwargs is not None:
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}
        return _dispatch_apply_or_tracing(def_name_or_resolver, args, kwargs, fallback)

    # Decorator mode
    def decorator(fallback: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fallback)
        def wrapped(*args: Any, **kwargs: Any):
            return _dispatch_apply_or_tracing(def_name_or_resolver, args, kwargs, fallback)

        return wrapped

    return decorator


def _dispatch_apply_or_tracing(
    def_name_or_resolver: Union[str, Callable[..., str]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    fallback: Optional[Callable[..., Any]],
) -> Any:
    """Internal dispatch function that handles tracing and apply.

    Parameters
    ----------
    def_name_or_resolver : Union[str, Callable[..., str]]
        Definition name or a resolver function.
    args : Tuple[Any, ...]
        Positional arguments (inputs only, or inputs + outputs for DPS).
    kwargs : Dict[str, Any]
        Keyword arguments.
    fallback : Optional[Callable[..., Any]]
        Fallback function.

    Returns
    -------
    Any
        Result of the call (None for DPS).
    """
    # Resolve def_name
    def_name = (
        def_name_or_resolver
        if isinstance(def_name_or_resolver, str)
        else def_name_or_resolver(*args, **kwargs)
    )

    apply_rt = get_apply_runtime()

    # Apply
    if apply_rt is not None:
        return apply_rt.dispatch(def_name, args, kwargs, fallback)

    tracing_rt = get_tracing_runtime()

    # Tracing
    if tracing_rt is not None:
        tracing_rt.collect(def_name, args, kwargs)

    # No runtime enabled
    if fallback is None:
        raise RuntimeError("Apply or tracing is not enabled and no fallback provided")
    return fallback(*args, **kwargs)


def enable_apply(
    dataset_path: Optional[str] = None, apply_config: Optional[ApplyConfig] = None
) -> ApplyRuntime:
    """Enable apply functionality globally and return a ApplyRuntime instance that manages the
    apply functionality.

    There is only one global ApplyRuntime instance. This function must be called in the main thread.

    Parameters
    ----------
    dataset_path : str, optional
        Path to the dataset/trace_set directory
    apply_config : ApplyConfig, optional
        Configuration for the apply runtime

    Returns
    -------
    ApplyRuntime
        The global ApplyRuntime instance managing the apply functionality.

    Examples
    --------
    >>> # Direct usage
    >>> enable_apply("/path/to/trace_set", cfg)
    >>> # Apply is now enabled
    >>> out = apply("rmsnorm_d4096", args=(...), kwargs={...}, fallback=ref_fn)
    >>> disable_apply()
    >>> # Apply is now disabled.

    >>> # Context manager usage
    >>> with enable_apply("/path/to/trace_set", cfg):
    ...     out = apply("rmsnorm_d4096", args=(...), kwargs={...}, fallback=ref_fn)
    >>> # Apply is now disabled.
    """
    prev_runtime = get_apply_runtime()
    trace_set = TraceSet.from_path(dataset_path)
    apply_runtime = ApplyRuntime(trace_set, apply_config, prev_runtime)
    set_apply_runtime(apply_runtime)
    return apply_runtime


def disable_apply() -> None:
    """Disable global apply functionality.

    This function silently disables the global apply runtime by setting it to None.
    After calling this function, any subsequent calls to apply() will use fallback
    functions instead of the apply runtime.

    Check out the `enable_apply` function for examples.
    """
    set_apply_runtime(None)
