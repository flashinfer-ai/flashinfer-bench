from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Union, overload

from flashinfer_bench.apply.runtime import get_runtime

_SENTINEL = object()


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


# Decorator
@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


# Imperative
@overload
def apply(
    def_or_resolver: Union[str, Callable[..., str]],
    *,
    runtime_kwargs: Dict[str, Any],
    fallback: Callable[..., Any],
) -> Any: ...


def apply(
    def_name_or_resolver: Union[str, Callable[..., str]],
    runtime_kwargs: Dict[str, Any] = _SENTINEL,
    fallback: Callable[..., Any] = _SENTINEL,
):
    """
    Unified apply API:
    - Decorator: @apply(resolver)
    - Imperative: apply(def_name_or_resolver, runtime_kwargs=..., fallback=...)
    """
    # Imperative
    if runtime_kwargs is not _SENTINEL:
        rt = get_runtime()
        if rt is None:
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
        rt = get_runtime()
        if rt is None:
            return fn

        # Inspect once
        sig = inspect.signature(fn)
        param_names = tuple(sig.parameters.keys())

        def wrapped(*args: Any, **kwargs: Any):
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
