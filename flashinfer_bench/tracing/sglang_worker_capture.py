"""Install thin capture hooks in SGLang parent and worker processes."""

from __future__ import annotations

import atexit
import functools
import importlib
import inspect
import json
import os
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .runtime import TracingRuntime
from .sglang_dumper_adapter import (
    SGLangCaptureSpec,
    _capture_spec,
    _create_shard_runtime,
)
from .sglang_inventory import _layer_index_from_path

_MODE_ENV = "FIB_SGLANG_CAPTURE_MODE"
_ROOT_ENV = "FIB_SGLANG_CAPTURE_ROOT"
_DEFINITIONS_ENV = "FIB_SGLANG_DEFINITIONS_DIR"
_INVENTORY_PREFIXES = ("sglang.", "transformers_modules.")


@dataclass
class _ProcessState:
    pid: int
    mode: str
    root: Path
    definitions_dir: Path | None
    specs_by_module: dict[str, list[SGLangCaptureSpec]] = field(default_factory=dict)
    specs_by_callable: dict[str, list[SGLangCaptureSpec]] = field(default_factory=dict)
    runtime: TracingRuntime | None = None
    inventory_seen: set[tuple[str, str]] = field(default_factory=set)
    module_binding_seen: set[tuple[int, str, str]] = field(default_factory=set)
    module_parents: dict[int, list[tuple[torch.nn.Module, str]]] = field(
        default_factory=dict
    )
    captures: dict[str, int] = field(default_factory=dict)
    bindings: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    callable_patches: list[tuple[Any, str, Any]] = field(default_factory=list)
    calls_since_flush: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATES: dict[int, _ProcessState] = {}
_HOOK_HANDLE: Any = None
_HOOK_PID: int | None = None
_REGISTRATION_HOOK_HANDLE: Any = None
_REGISTRATION_HOOK_PID: int | None = None
_ATEXIT_REGISTERED = False
_IN_HOOK = threading.local()


@contextmanager
def sglang_capture_environment(
    capture_root: Path, *, mode: str, definitions_dir: Path | None = None
) -> Iterator[None]:
    """Install the same SGLang hook in the parent process and spawned workers."""
    if mode not in {"inventory", "workloads"}:
        raise ValueError("SGLang capture mode must be inventory or workloads")
    if mode == "workloads" and definitions_dir is None:
        raise ValueError("workload capture requires definitions_dir")

    shutil.rmtree(capture_root, ignore_errors=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = capture_root / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    (bootstrap_dir / "sitecustomize.py").write_text(
        "from flashinfer_bench.tracing.sglang_worker_capture import install_from_env\n"
        "install_from_env()\n",
        encoding="utf-8",
    )

    updates = {
        _MODE_ENV: mode,
        _ROOT_ENV: str(capture_root),
        _DEFINITIONS_ENV: str(definitions_dir or ""),
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(bootstrap_dir), os.environ.get("PYTHONPATH", "")) if part
        ),
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    install_from_env()
    try:
        yield
    finally:
        uninstall_current_process()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def install_from_env() -> None:
    """Worker bootstrap entrypoint called by the temporary ``sitecustomize``."""
    mode = os.environ.get(_MODE_ENV)
    root = os.environ.get(_ROOT_ENV)
    if mode not in {"inventory", "workloads"} or not root:
        return
    state = _state_for_current_process()
    _install_module_registration_hook()
    _install_global_module_hook()
    if state.mode == "workloads" and not state.callable_patches:
        _install_callable_wrappers(state)
    _register_atexit()


def uninstall_current_process() -> None:
    """Flush and remove hooks installed in the current process."""
    global _HOOK_HANDLE, _HOOK_PID, _REGISTRATION_HOOK_HANDLE, _REGISTRATION_HOOK_PID
    state = _STATES.pop(os.getpid(), None)
    if state is not None:
        _flush_state(state)
        for owner, attribute, original in reversed(state.callable_patches):
            setattr(owner, attribute, original)
    if _HOOK_HANDLE is not None and _HOOK_PID == os.getpid():
        _HOOK_HANDLE.remove()
        _HOOK_HANDLE = None
        _HOOK_PID = None
    if _REGISTRATION_HOOK_HANDLE is not None and _REGISTRATION_HOOK_PID == os.getpid():
        _REGISTRATION_HOOK_HANDLE.remove()
        _REGISTRATION_HOOK_HANDLE = None
        _REGISTRATION_HOOK_PID = None


def _state_for_current_process() -> _ProcessState:
    pid = os.getpid()
    existing = _STATES.get(pid)
    if existing is not None:
        return existing
    mode = os.environ[_MODE_ENV]
    root = Path(os.environ[_ROOT_ENV])
    raw_definitions = os.environ.get(_DEFINITIONS_ENV, "")
    definitions_dir = Path(raw_definitions) if raw_definitions else None
    state = _ProcessState(pid=pid, mode=mode, root=root, definitions_dir=definitions_dir)
    if mode == "workloads":
        if definitions_dir is None:
            raise RuntimeError("SGLang workload capture has no definitions directory")
        state = _prepare_workload_state(state)
    _STATES[pid] = state
    return state


def _prepare_workload_state(state: _ProcessState) -> _ProcessState:
    assert state.definitions_dir is not None
    specs: list[SGLangCaptureSpec] = []
    for path in sorted(state.definitions_dir.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        spec = _capture_spec(value)
        if spec.modules or spec.callables:
            specs.append(spec)
            for module_path in spec.modules:
                state.specs_by_module.setdefault(module_path, []).append(spec)
            for callable_path in spec.callables:
                state.specs_by_callable.setdefault(callable_path, []).append(spec)

    callable_specs = [spec for spec in specs if spec.callables]
    if callable_specs:
        state.runtime = _create_shard_runtime(
            state.root / "shards" / str(state.pid),
            definitions_dir=state.definitions_dir,
            specs=callable_specs,
        )
    return state


def _install_global_module_hook() -> None:
    global _HOOK_HANDLE, _HOOK_PID
    if _HOOK_HANDLE is not None:
        _HOOK_PID = os.getpid()
        return
    register = torch.nn.modules.module.register_module_forward_hook
    try:
        _HOOK_HANDLE = register(_module_forward_hook, with_kwargs=True)
    except TypeError:
        _HOOK_HANDLE = register(_module_forward_hook_without_kwargs)
    _HOOK_PID = os.getpid()


def _install_module_registration_hook() -> None:
    global _REGISTRATION_HOOK_HANDLE, _REGISTRATION_HOOK_PID
    if _REGISTRATION_HOOK_HANDLE is not None:
        _REGISTRATION_HOOK_PID = os.getpid()
        return
    register = torch.nn.modules.module.register_module_module_registration_hook
    _REGISTRATION_HOOK_HANDLE = register(_module_registration_hook)
    _REGISTRATION_HOOK_PID = os.getpid()


def _module_registration_hook(
    parent: torch.nn.Module, name: str, child: torch.nn.Module | None
) -> None:
    if child is None:
        return
    state = _state_for_current_process()
    parents = state.module_parents.setdefault(id(child), [])
    if not any(existing is parent and existing_name == name for existing, existing_name in parents):
        parents.append((parent, name))


def _module_forward_hook(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    _handle_module_call(module, args, kwargs, output)


def _module_forward_hook_without_kwargs(
    module: torch.nn.Module, args: tuple[Any, ...], output: Any
) -> None:
    _handle_module_call(module, args, {}, output)


def _handle_module_call(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    if getattr(_IN_HOOK, "active", False):
        return
    _IN_HOOK.active = True
    try:
        state = _state_for_current_process()
        class_path = f"{type(module).__module__}.{type(module).__qualname__}"
        if state.mode == "inventory":
            _record_inventory(state, module, class_path, args, kwargs, output)
            return
        specs = state.specs_by_module.get(class_path, [])
        for spec in specs:
            _record_module_binding(state, spec, module)
    finally:
        _IN_HOOK.active = False


def _install_callable_wrappers(state: _ProcessState) -> None:
    for path, specs in state.specs_by_callable.items():
        owner, attribute, original = _resolve_attribute(path)

        @functools.wraps(original)
        def wrapped(*args: Any, __original=original, __specs=tuple(specs), **kwargs: Any) -> Any:
            current = _state_for_current_process()
            if not getattr(_IN_HOOK, "active", False):
                _IN_HOOK.active = True
                try:
                    for spec in __specs:
                        _collect_call(current, spec, __original, args, kwargs, module=None)
                    if current.runtime is not None:
                        _flush_incrementally(current)
                finally:
                    _IN_HOOK.active = False
            return __original(*args, **kwargs)

        setattr(owner, attribute, wrapped)
        state.callable_patches.append((owner, attribute, original))


def _collect_call(
    state: _ProcessState,
    spec: SGLangCaptureSpec,
    callable_value: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    module: torch.nn.Module | None,
) -> None:
    if state.runtime is None:
        return
    try:
        signature = inspect.signature(callable_value)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        definition = state.runtime._trace_set.definitions[spec.name]
        values: dict[str, Any] = {}
        for input_name in definition.inputs:
            kind, source_name = spec.input_sources.get(input_name, ("arg", input_name))
            if kind == "arg":
                if source_name.isdecimal():
                    position = int(source_name)
                    if position >= len(args):
                        raise KeyError(f"forward positional argument {position} is missing")
                    values[input_name] = args[position]
                else:
                    if source_name not in bound.arguments:
                        raise KeyError(f"forward argument {source_name!r} is missing")
                    values[input_name] = bound.arguments[source_name]
            else:
                if module is None:
                    raise KeyError(f"attribute source {source_name!r} requires sglang_module")
                values[input_name] = _nested_attribute(module, source_name)
        state.runtime.collect(spec.name, args=(), kwargs=values)
        state.captures[spec.name] = state.captures.get(spec.name, 0) + 1
        state.calls_since_flush += 1
    except Exception as exc:  # noqa: BLE001 - capture must never break model inference
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")


def _record_module_binding(
    state: _ProcessState, spec: SGLangCaptureSpec, module: torch.nn.Module
) -> None:
    """Persist only Definition/path/attribute metadata that SGLang's dumper lacks."""
    module_paths = _module_instance_paths(state, module)
    if not module_paths:
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: module instance path is unavailable")
        return
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        signature = None
    argument_positions: dict[str, int] = {}
    if signature is not None:
        position = 0
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                argument_positions[parameter.name] = position
                position += 1

    attributes: dict[str, Any] = {}
    try:
        for input_name, (kind, source_name) in spec.input_sources.items():
            if kind == "attr":
                attributes[input_name] = _portable_binding_value(
                    _nested_attribute(module, source_name)
                )
    except Exception as exc:  # noqa: BLE001 - capture metadata cannot break inference
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
        return

    for module_path in module_paths:
        identity = (id(module), module_path, spec.name)
        if identity in state.module_binding_seen:
            continue
        state.module_binding_seen.add(identity)
        destination = (
            state.root
            / "module_bindings"
            / str(state.pid)
            / f"{len(state.module_binding_seen):05d}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "definition": spec.name,
                "class_path": f"{type(module).__module__}.{type(module).__qualname__}",
                "module_path": module_path,
                "argument_positions": argument_positions,
                "attributes": attributes,
            },
            destination,
        )
        state.bindings[spec.name] = state.bindings.get(spec.name, 0) + 1


def _portable_binding_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_portable_binding_value(item) for item in value)
    if isinstance(value, list):
        return [_portable_binding_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _portable_binding_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported module attribute type: {type(value).__name__}")


def _record_inventory(
    state: _ProcessState,
    module: torch.nn.Module,
    class_path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> None:
    if not class_path.startswith(_INVENTORY_PREFIXES):
        return
    module_paths = _module_instance_paths(state, module)
    identities = {(class_path, path) for path in module_paths} or {(class_path, "")}
    if identities <= state.inventory_seen:
        return
    state.inventory_seen.update(identities)
    forward = module.forward
    try:
        signature = inspect.signature(forward)
        bound = signature.bind_partial(*args, **kwargs)
        inputs: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            parameter = signature.parameters.get(name)
            if parameter is not None and parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                inputs.update(
                    {f"arg_{index}": _describe_value(item) for index, item in enumerate(value)}
                )
            elif parameter is not None and parameter.kind is inspect.Parameter.VAR_KEYWORD:
                inputs.update({key: _describe_value(item) for key, item in value.items()})
            else:
                inputs[name] = _describe_value(value)
    except (TypeError, ValueError):
        signature = "unavailable"
        inputs = {f"arg_{index}": _describe_value(value) for index, value in enumerate(args)}
    try:
        source = inspect.getsource(type(module))[:12000]
    except (OSError, TypeError):
        source = ""
    try:
        source_file = inspect.getsourcefile(type(module)) or ""
    except TypeError:
        source_file = ""
    item = {
        "class_path": class_path,
        "module_paths": module_paths,
        "module_observations": [
            {
                "pid": state.pid,
                "module_path": path,
                "layer_index": _layer_index_from_path(path),
            }
            for path in module_paths
        ],
        "layer_indices": sorted(
            {
                layer_index
                for path in module_paths
                if (layer_index := _layer_index_from_path(path)) is not None
            }
        ),
        "forward_signature": str(signature),
        "sample_inputs": inputs,
        "sample_output": _describe_value(output),
        "source_file": source_file,
        "source": source,
        "pid": state.pid,
    }
    path = state.root / "inventory" / f"{state.pid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def _module_instance_paths(state: _ProcessState, module: torch.nn.Module) -> list[str]:
    """Recover paths relative to the root model from module registration events."""

    def visit(current: torch.nn.Module, seen: set[int]) -> list[str]:
        current_id = id(current)
        if current_id in seen:
            return []
        parents = state.module_parents.get(current_id, [])
        if not parents:
            return [""]
        paths: list[str] = []
        for parent, child_name in parents:
            for parent_path in visit(parent, {*seen, current_id}):
                paths.append(".".join(part for part in (parent_path, child_name) if part))
        return paths

    return sorted(set(path for path in visit(module, set()) if path))


def _describe_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "items": [_describe_value(item) for item in value[:8]],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {str(key): _describe_value(item) for key, item in list(value.items())[:8]},
        }
    return {"type": type(value).__name__, "value": repr(value)[:200]}


def _resolve_attribute(path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    for index in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:index]))
        except ImportError:
            continue
        for part in parts[index:-1]:
            value = getattr(value, part)
        attribute = parts[-1]
        original = getattr(value, attribute)
        if not callable(original):
            raise TypeError(f"SGLang capture target is not callable: {path}")
        return value, attribute, original
    raise ImportError(f"cannot import SGLang capture target: {path}")


def _nested_attribute(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _write_state_summary(state: _ProcessState) -> None:
    path = state.root / "status" / f"{state.pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": state.pid,
                "captures": state.captures,
                "bindings": state.bindings,
                "errors": state.errors,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _flush_state(state: _ProcessState) -> None:
    if state.runtime is not None:
        state.runtime.flush()
        state.calls_since_flush = 0
    _write_state_summary(state)


def _flush_incrementally(state: _ProcessState) -> None:
    """Bound buffered captures without flushing after every module call."""
    if state.calls_since_flush >= 256:
        _flush_state(state)


def _flush_current_process() -> None:
    state = _STATES.get(os.getpid())
    if state is not None:
        _flush_state(state)


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_flush_current_process)
        _ATEXIT_REGISTERED = True
