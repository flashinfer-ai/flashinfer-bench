from __future__ import annotations

import inspect
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
    runtime_kwargs: Dict[str, Any],
    fallback: Optional[Callable[..., Any]],
) -> Any: ...


def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    runtime_kwargs: Optional[Dict[str, Any]] = None,
    fallback: Optional[Callable[..., Any]] = None,
):
    """
    Decorator/function for routing to the best-performing kernel recorded in the
    FlashInfer Trace database.

    This API can be used in two modes:

    1) **Decorator mode** (only ``def_name_or_resolver`` provided): returns a decorator
       that wraps a kernel function with a router. The router selects the best-performing
       candidate according to the function's runtime arguments.
    2) **Function mode** (``runtime_kwargs`` provided, optionally ``fallback``):
       immediately resolves and calls the best-performing kernel and returns its result.

    Parameters
    ----------
    def_name_or_resolver : Union[str, Callable[..., str]]
        The kernel name, or a resolver ``fn(*args, **kwargs) -> str`` that maps runtime
        arguments to a kernel name (definition name).
    runtime_kwargs : Dict[str, Any], optional
        Only used in **function mode**. The runtime arguments to feed into the selected
        kernel. Use this to call the kernel immediately instead of returning a decorator.
    fallback : Optional[Callable[..., Any]], optional
        Only used in **function mode**. A fallback function to invoke when no matching
        kernel is found in the Trace database.

    Returns
    -------
    Union[Callable[[Callable[..., Any]], Callable[..., Any]], Any]
        - **Decorator mode**: a decorator that transforms the target kernel function into
          a routed version.
        - **Function mode**: the return value produced by the selected (or fallback) kernel.

    Examples
    --------
    Decorator mode with a fixed name
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> @apply("gemm_bf16")
    ... def gemm_bf16(A, B):
    ...     return torch.nn.functional.linear(A, B)

    Decorator mode with a resolver
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> @apply(lambda A, B: f"gemm_n{B.shape[0]}_k{B.shape[1]}")
    ... def gemm_bf16(A, B):
    ...     return torch.nn.functional.linear(A, B)

    Function mode
    ~~~~~~~~~~~~~
    >>> out = apply(
    ...     "gemm_bf16",
    ...     runtime_kwargs={"A": A, "B": B, "bias": None},
    ...     fallback=lambda **kw: torch.nn.functional.linear(**kw),
    ... )
    """
    # Imperative
    if runtime_kwargs is not None:
        kwargs = dict(runtime_kwargs)
        def_name = (
            def_name_or_resolver
            if isinstance(def_name_or_resolver, str)
            else def_name_or_resolver(**kwargs)
        )

        tracing_rt = get_tracing_runtime()
        if tracing_rt is not None:
            tracing_rt.collect(def_name, kwargs)
            tracing_rt.flush()

        apply_rt = get_apply_runtime()
        if apply_rt is None:
            if fallback is None:
                raise RuntimeError("Apply is not enabled and no fallback provided")
            return fallback(**kwargs)

        return apply_rt.dispatch(def_name, kwargs, fallback)

    # Decorator
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Inspect once
        sig = inspect.signature(fn)
        param_names = tuple(sig.parameters.keys())

        def wrapped(*args: Any, **kwargs: Any):
            tracing_rt = get_tracing_runtime()
            apply_rt = get_apply_runtime()
            if tracing_rt is None and apply_rt is None:
                return fn(*args, **kwargs)

            bound = _merge_args_to_kwargs(param_names, args, kwargs)
            def_name = (
                def_name_or_resolver
                if isinstance(def_name_or_resolver, str)
                else def_name_or_resolver(**bound)
            )
            if tracing_rt is not None:
                tracing_rt.collect(def_name, bound)
            if apply_rt is None:
                return fn(*args, **kwargs)
            return apply_rt.dispatch(def_name, bound, fn)

        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        wrapped.__wrapped__ = fn
        return wrapped

    return decorator


def enable_apply(
    dataset_path: Optional[str] = None, apply_config: Optional[ApplyConfig] = None
) -> ApplyRuntime:
    """Enable apply functionality globally and return a ApplyRuntime instance that manages the
    apply functionality.

    There is only one global ApplyRuntime instance. This function must be called in the main thread.

    Parameters
    ----------
    dataset_path : str, optional
        Path to the dataset/traceset directory
    apply_config : ApplyConfig, optional
        Configuration for the apply runtime

    Returns
    -------
    ApplyRuntime
        The global ApplyRuntime instance managing the apply functionality.

    Examples
    --------
    >>> # Direct usage
    >>> enable_apply("/path/to/traceset", cfg)
    >>> # Apply is now enabled
    >>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    >>> disable_apply()
    >>> # Apply is now disabled.

    >>> # Context manager usage
    >>> with enable_apply("/path/to/traceset", cfg):
    ...     out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
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


def _merge_args_to_kwargs(
    param_names: Tuple[str], args: Tuple[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if len(args) > len(param_names):
        raise TypeError("Too many positional arguments")
    merged: Dict[str, Any] = {}
    for i, val in enumerate(args):
        merged[param_names[i]] = val
    # Merge kwargs with conflict detection
    for k, v in kwargs.items():
        if k in merged:
            raise TypeError(f"Multiple values for argument '{k}'")
        merged[k] = v
    return merged
