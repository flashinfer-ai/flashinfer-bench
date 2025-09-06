from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, Mapping, Optional, Union, overload

from .config import ApplyConfig
from .runtime import ApplyRuntime, get_runtime, set_runtime

_SENTINEL = object()


# Decorator
@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    *,
    runtime_kwargs: Dict[str, Any],
    fallback: Optional[Callable[..., Any]],
) -> Any:
    ...


def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    runtime_kwargs: Dict[str, Any] = _SENTINEL,
    fallback: Optional[Callable[..., Any]] = _SENTINEL,
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
    ... def gemm_bf16(A, B, bias=None):
    ...     return torch.nn.functional.linear(A, B, bias)

    Decorator mode with a resolver
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> @apply(lambda A, B: f"gemm_n_{B.shape[0]}_k_{B.shape[1]}")
    ... def gemm_bf16(A, B, bias=None):
    ...     return torch.nn.functional.linear(A, B, bias)

    Function mode
    ~~~~~~~~~~~~~
    >>> out = apply(
    ...     "gemm_bf16",
    ...     runtime_kwargs={"A": A, "B": B, "bias": None},
    ...     fallback=lambda **kw: torch.nn.functional.linear(**kw),
    ... )
    """
    # Imperative
    if runtime_kwargs is not _SENTINEL:
        rt = get_runtime()
        if rt is None:
            if fallback is None:
                raise RuntimeError("Apply is not enabled and no fallback provided")
            return fallback(**runtime_kwargs)

        kwargs = dict(runtime_kwargs)
        def_name = (
            def_name_or_resolver
            if isinstance(def_name_or_resolver, str)
            else def_name_or_resolver(**kwargs)
        )
        return rt.dispatch(def_name, kwargs, fallback)

    # Decorator
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Inspect once
        sig = inspect.signature(fn)
        param_names = tuple(sig.parameters.keys())

        def wrapped(*args: Any, **kwargs: Any):
            rt = get_runtime()
            if rt is None:
                return fn(*args, **kwargs)

            bound = _merge_to_kwargs(param_names, args, kwargs)
            def_name = (
                def_name_or_resolver
                if isinstance(def_name_or_resolver, str)
                else def_name_or_resolver(**bound)
            )
            return rt.dispatch(def_name, bound, fn)

        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        wrapped.__wrapped__ = fn
        return wrapped

    return decorator


def enable_apply(
    dataset_path: Optional[str] = None, apply_config: Optional[ApplyConfig] = None
) -> _ApplyHandle:
    """
    Immediately enable global apply, and return a handle:
      - Use as a function:imperative apply
      - Use in a with block: contextually available, exiting restores the original state
    Usage:
      enable_apply("/path/to/traceset", cfg)
      out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)

      # Or
      with enable_apply("/path/to/traceset", cfg) as apply:
          out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)
    """
    return _ApplyHandle(dataset_path, apply_config)


def disable_apply() -> None:
    """Silently disable: set the global runtime to None."""
    set_runtime(None)


class _ApplyHandle:
    """Context manager for enabling apply."""

    def __init__(
        self, dataset_path: Optional[str] = None, config: Optional[ApplyConfig] = None
    ) -> None:
        # Record current runtime, then install new runtime
        self._prev: Optional[ApplyRuntime] = get_runtime()
        self._rt = ApplyRuntime(_resolve_dataset(dataset_path), _resolve_cfg(config))
        set_runtime(self._rt)

    # Imperative apply
    def __call__(
        self,
        def_name_or_resolver: Union[str, Callable[..., str]],
        *,
        runtime_kwargs: Dict[str, Any],
        fallback: Optional[Callable[..., Any]],
    ) -> Any:
        def_name = (
            def_name_or_resolver
            if isinstance(def_name_or_resolver, str)
            else def_name_or_resolver(**runtime_kwargs)
        )
        return self._rt.dispatch(def_name, runtime_kwargs, fallback)

    def __enter__(self) -> Callable[[Union[str, Callable[..., str]]], Any]:
        return self.__call__

    # Exit restores the runtime before entering
    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            set_runtime(self._prev)
        finally:
            self._prev = None
            self._rt = None
        return False


def _merge_to_kwargs(
    param_names: tuple[str, ...], args: tuple[Any, ...], kwargs: Mapping[str, Any]
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


def _resolve_dataset(dataset_path: Optional[str]) -> str:
    if dataset_path:
        return dataset_path
    env_ds = os.environ.get("FIB_DATASET_PATH")
    if env_ds:
        return env_ds
    raise ValueError("dataset_path is required (or set FIB_DATASET_PATH).")


def _resolve_cfg(cfg: Optional[ApplyConfig]) -> ApplyConfig:
    return cfg or ApplyConfig()
