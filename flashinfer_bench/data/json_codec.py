"""Unified JSON encoding/decoding for all dataclasses."""

import json
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union, get_args, get_origin

from .definition import AxisConst, AxisVar, Definition, TensorSpec
from .solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Trace,
    Workload,
)

T = TypeVar("T")

# Fields that preserve None in serialization
_PRESERVE_NULL_FIELDS = {
    Trace: {"solution", "evaluation"},
    Evaluation: {"correctness", "performance"},
    TensorSpec: {"shape"},
}

# Field output order for JSON serialization
_FIELD_ORDER = {
    Definition: (
        "name",
        "type",
        "tags",
        "description",
        "axes",
        "inputs",
        "outputs",
        "reference",
        "constraints",
    ),
    Solution: ("name", "definition", "description", "author", "spec", "sources"),
    Trace: ("definition", "solution", "workload", "evaluation"),
}


# Field-level decoders
def _decode_axes(v):
    """Decode Definition.axes."""
    out = {}
    for name, ax in v.items():
        if isinstance(ax, dict):
            t = ax.get("type")
            if t == "const":
                out[name] = dict_to_dataclass(ax, AxisConst)
            elif t == "var":
                out[name] = dict_to_dataclass(ax, AxisVar)
            else:
                raise ValueError(f"Unsupported axis type '{t}'")
        else:
            raise ValueError(f"Unsupported axis field '{ax}'")
    return out


def _decode_tensor_specs(v):
    """Decode Definition.inputs/outputs."""
    out = {}
    for k, x in v.items():
        if not isinstance(x, dict):
            raise ValueError(f"Unsupported tensor spec field '{x}'")
        out[k] = dict_to_dataclass(x, TensorSpec)
    return out


def _decode_workload_inputs(v):
    """Decode Workload.inputs."""
    out = {}
    for k, x in v.items():
        if isinstance(x, dict):
            t = x.get("type")
            if t == "random":
                out[k] = dict_to_dataclass(x, RandomInput)
            elif t == "safetensors":
                out[k] = dict_to_dataclass(x, SafetensorsInput)
            elif t == "scalar":
                out[k] = dict_to_dataclass(x, ScalarInput)
            else:
                raise ValueError(f"Unsupported workload input type '{t}'")
        else:
            raise ValueError(f"Unsupported workload input field '{x}'")
    return out


def _decode_sources(v):
    """Decode Solution.sources."""
    out = []
    for x in v:
        if not isinstance(x, dict):
            raise ValueError(f"Unsupported source file field '{x}'")
        out.append(dict_to_dataclass(x, SourceFile))
    return out


def _decode_spec(v):
    """Decode Solution.spec."""
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported build spec field '{v}'")
    return dict_to_dataclass(v, BuildSpec)


def _decode_evaluation(v):
    """Decode Trace.evaluation."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported evaluation field '{v}'")
    return dict_to_dataclass(v, Evaluation)


def _decode_workload(v):
    """Decode Trace.workload."""
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported workload field '{v}'")
    return dict_to_dataclass(v, Workload)


def _decode_correctness(v):
    """Decode Evaluation.correctness."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported correctness field '{v}'")
    return dict_to_dataclass(v, Correctness)


def _decode_performance(v):
    """Decode Evaluation.performance."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported performance field '{v}'")
    return dict_to_dataclass(v, Performance)


def _decode_environment(v):
    """Decode Evaluation.environment."""
    if not isinstance(v, dict):
        raise ValueError(f"Unsupported environment field '{v}'")
    return dict_to_dataclass(v, Environment)


def _decode_language(v):
    """Decode BuildSpec.language."""
    if isinstance(v, str):
        try:
            return SupportedLanguages(v.strip().lower())
        except ValueError as e:
            raise ValueError(
                f"Unsupported language '{v}', must be one of {[x.value for x in SupportedLanguages]}"
            ) from e
    raise ValueError(f"Unsupported language field '{v}'")


def _decode_status(v):
    """Decode Evaluation.status."""
    if isinstance(v, str):
        try:
            return EvaluationStatus(v.strip().upper())
        except ValueError as e:
            raise ValueError(
                f"Unsupported evaluation status '{v}', must be one of {[x.value for x in EvaluationStatus]}"
            ) from e
    raise ValueError(f"Unsupported evaluation status field '{v}'")


# Field decoder dispatch table
_FIELD_DECODERS = {
    (Definition, "axes"): _decode_axes,
    (Definition, "inputs"): _decode_tensor_specs,
    (Definition, "outputs"): _decode_tensor_specs,
    (Workload, "inputs"): _decode_workload_inputs,
    (Solution, "sources"): _decode_sources,
    (Solution, "spec"): _decode_spec,
    (Trace, "evaluation"): _decode_evaluation,
    (Trace, "workload"): _decode_workload,
    (Evaluation, "correctness"): _decode_correctness,
    (Evaluation, "performance"): _decode_performance,
    (Evaluation, "environment"): _decode_environment,
    (BuildSpec, "language"): _decode_language,
    (Evaluation, "status"): _decode_status,
}


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass instance to a dictionary, handling nested dataclasses."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result: Dict[str, Any] = {}
        preserve = _PRESERVE_NULL_FIELDS.get(type(obj), set())

        all_fields = list(fields(obj))
        custom_order = _FIELD_ORDER.get(type(obj))
        if custom_order:
            by_name = {f.name: f for f in all_fields}
            ordered_fields = [by_name[n] for n in custom_order if n in by_name]
            # Append remaining fields not in custom order
            ordered_fields += [f for f in all_fields if f.name not in custom_order]
        else:
            ordered_fields = all_fields

        # Preserve None
        for f in ordered_fields:
            v = getattr(obj, f.name)
            if f.name in preserve:
                result[f.name] = None if v is None else dataclass_to_dict(v)
            else:
                if v is None:
                    continue
                result[f.name] = dataclass_to_dict(v)

        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (SupportedLanguages, EvaluationStatus)):
        return obj.value
    else:
        return obj


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance."""
    if not is_dataclass(cls):
        return data

    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        v = data[f.name]

        # Handle None values
        if v is None:
            kwargs[f.name] = None
            continue

        # Check for field-specific decoder
        decoder = _FIELD_DECODERS.get((cls, f.name))
        if decoder:
            kwargs[f.name] = decoder(v)
            continue

        # Generic handling based on type
        tp = f.type
        origin = get_origin(tp)
        args = get_args(tp)

        # Handle Optional[T] where T may use typing.Union or PEP 604 (T | None)
        union_types = {Union}
        if hasattr(types, "UnionType"):
            union_types.add(types.UnionType)  # type: ignore[attr-defined]
        if origin in union_types and type(None) in args:
            non_none_args = tuple(a for a in args if a is not type(None))
            # Prefer dataclass conversion when value is dict and any arg is a dataclass
            if isinstance(v, dict):
                dc_types = [a for a in non_none_args if is_dataclass(a)]
                if dc_types:
                    kwargs[f.name] = dict_to_dataclass(v, dc_types[0])
                    continue
            # Otherwise, use value as-is
            kwargs[f.name] = v
            continue

        # Handle Dict[K, V] where V is a dataclass
        if origin is dict and args and len(args) == 2:
            if is_dataclass(args[1]):
                kwargs[f.name] = {
                    k: dict_to_dataclass(val, args[1]) if isinstance(val, dict) else val
                    for k, val in v.items()
                }
            else:
                kwargs[f.name] = v
            continue

        # Handle List[T] where T is a dataclass
        if origin is list and args:
            if is_dataclass(args[0]):
                kwargs[f.name] = [
                    dict_to_dataclass(x, args[0]) if isinstance(x, dict) else x for x in v
                ]
            else:
                kwargs[f.name] = v
            continue

        # Handle direct dataclass fields
        if is_dataclass(tp) and isinstance(v, dict):
            kwargs[f.name] = dict_to_dataclass(v, tp)
            continue

        # Default: use value as-is
        kwargs[f.name] = v

    return cls(**kwargs)


# I/O functions
def to_json(obj: Any, indent: int = 2) -> str:
    """Convert a dataclass to JSON string."""
    return json.dumps(dataclass_to_dict(obj), indent=indent, ensure_ascii=False)


def from_json(json_str: str, cls: Type[T]) -> T:
    """Parse JSON string to dataclass."""
    return dict_to_dataclass(json.loads(json_str), cls)


def save_json_file(obj: Any, path: Union[str, Path]) -> None:
    """Save a dataclass to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclass_to_dict(obj), f, indent=2, ensure_ascii=False)


def load_json_file(path: Union[str, Path], cls: Type[T]) -> T:
    """Load a dataclass from a JSON file."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return dict_to_dataclass(json.load(f), cls)


def save_jsonl_file(objects: List[Any], path: Union[str, Path]) -> None:
    """Save a list of dataclasses to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(dataclass_to_dict(obj), ensure_ascii=False))
            f.write("\n")


def load_jsonl_file(path: Union[str, Path], cls: Type[T]) -> List[T]:
    """Load a list of dataclasses from a JSONL file."""
    out = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(dict_to_dataclass(json.loads(line), cls))
    return out


def append_jsonl_line(path: Union[str, Path], obj: Any) -> None:
    """Append a dataclass to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dataclass_to_dict(obj), ensure_ascii=False))
        f.write("\n")


def append_jsonl_lines(path: Union[str, Path], objs: List[Any]) -> None:
    """Append a list of dataclasses to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(dataclass_to_dict(obj), ensure_ascii=False))
            f.write("\n")
