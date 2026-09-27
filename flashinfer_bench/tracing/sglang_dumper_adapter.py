"""Adapt SGLang Dumper inputs to the shared tracing workload runtime."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flashinfer_bench.data import TraceSet

from .config import TracingConfig, TracingConfigRegistry
from .runtime import TracingRuntime
from .sglang_inventory import _module_path_order


@dataclass(frozen=True)
class SGLangCaptureSpec:
    """One reviewed definition and its exact SGLang capture point."""

    name: str
    modules: tuple[str, ...]
    callables: tuple[str, ...]
    input_sources: dict[str, tuple[str, str]]


def load_sglang_definition_files(definitions_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Return definitions with an exact SGLang module or callable capture tag."""
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(definitions_dir.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"definition must be a JSON object: {path}")
        spec = _capture_spec(value)
        if spec.modules or spec.callables:
            files.append(path)
        else:
            skipped.append({"path": str(path), "reason": "missing_sglang_capture_tag"})
    return files, skipped


def build_sglang_dumper_workload_filter(
    definition_files: list[Path], module_inventory: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Build an exact SGLang Dumper filter for reviewed module definitions."""
    module_classes: set[str] = set()
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        module_classes.update(_capture_spec(value).modules)
    if not module_classes:
        return "", {"selected_modules": {}, "missing_modules": []}

    inventory_modules = module_inventory.get("modules")
    if not isinstance(inventory_modules, list):
        raise ValueError("SGLang execution inventory has no modules list")
    paths_by_class: dict[str, list[str]] = {}
    for item in inventory_modules:
        if not isinstance(item, dict) or not isinstance(item.get("class_path"), str):
            continue
        module_paths = item.get("module_paths")
        if not isinstance(module_paths, list):
            continue
        paths_by_class[str(item["class_path"])] = sorted(
            {path for path in module_paths if isinstance(path, str) and path}
        )

    selected: dict[str, str] = {}
    missing: list[str] = []
    for class_path in sorted(module_classes):
        paths = paths_by_class.get(class_path, [])
        if not paths:
            missing.append(class_path)
            continue
        selected[class_path] = min(paths, key=_module_path_order)
    if missing:
        raise ValueError(
            "reviewed SGLang modules are absent from execution inventory: "
            + ", ".join(missing)
        )
    if not selected:
        return "", {"selected_modules": {}, "missing_modules": []}

    alternatives = "|".join(re.escape(path) for path in selected.values())
    pattern = (
        rf"^non_intrusive__(?:{alternatives})\."
        r"(?:inputs(?:\.|$)|output(?:\.|$))"
    )
    return (
        f"search({pattern!r}, name) is not None",
        {"selected_modules": selected, "missing_modules": []},
    )


def adapt_sglang_dumper_workloads(
    dump_root: Path,
    *,
    capture_root: Path,
    definition_files: list[Path],
) -> dict[str, Any]:
    """Map SGLang module input dumps to Definitions and reuse TracingRuntime.collect."""
    definitions: dict[str, tuple[Path, SGLangCaptureSpec, list[str]]] = {}
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        spec = _capture_spec(value)
        if spec.modules:
            inputs = value.get("inputs")
            definitions[spec.name] = (
                path,
                spec,
                list(inputs) if isinstance(inputs, dict) else [],
            )
    if not definitions:
        return {
            "summary": {
                "dump_files": 0,
                "module_calls": 0,
                "collect_attempts": 0,
                "collect_accepted": 0,
                "collect_rejected": 0,
            },
            "collect_results": {},
            "errors": [],
        }

    bindings = _load_module_bindings(capture_root)
    shard = capture_root / "shards" / "sglang_dumper"
    runtime = _create_shard_runtime(
        shard,
        definitions_dir=None,
        specs=[spec for _, spec, _ in definitions.values()],
        definition_files=[path for path, _, _ in definitions.values()],
    )

    records: list[tuple[tuple[str, int, int, int, str], str, str, str, Any]] = []
    errors: list[str] = []
    files = sorted(dump_root.rglob("*.pt")) if dump_root.exists() else []
    for path in files:
        try:
            item = _load_torch_payload(path)
        except Exception as exc:  # noqa: BLE001 - bad evidence must not hide all captures
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("meta"), dict):
            continue
        meta = item["meta"]
        parsed = _parse_sglang_dumper_name(meta.get("name"))
        if parsed is None:
            continue
        module_path, value_kind, value_name = parsed
        relative = path.relative_to(dump_root)
        experiment = relative.parts[0] if len(relative.parts) > 1 else ""
        rank = meta.get("rank", meta.get("world_rank"))
        step = meta.get("step")
        dump_index = meta.get("dump_index")
        order = (
            experiment,
            int(rank) if type(rank) is int else -1,
            int(step) if type(step) is int else -1,
            int(dump_index) if type(dump_index) is int else -1,
            str(path),
        )
        records.append((order, module_path, value_kind, value_name, item.get("value")))

    pending: dict[tuple[str, int, str], dict[str, Any]] = {}
    module_calls = 0
    collect_attempts = 0
    collect_accepted = 0
    collect_rejected = 0
    collect_results: dict[str, dict[str, Any]] = {}
    missing_bindings: set[str] = set()
    for order, module_path, value_kind, value_name, value in sorted(records):
        key = (order[0], order[1], module_path)
        if value_kind == "input":
            pending.setdefault(key, {})[value_name] = value
            continue
        call_inputs = pending.pop(key, {})
        if not call_inputs:
            continue
        module_calls += 1
        module_bindings = bindings.get(module_path, [])
        if not module_bindings:
            missing_bindings.add(module_path)
            continue
        for binding in module_bindings:
            definition_name = binding.get("definition")
            definition_entry = definitions.get(str(definition_name))
            if definition_entry is None:
                continue
            _, spec, input_names = definition_entry
            try:
                values = _definition_values_from_dumper(
                    spec, binding, call_inputs, input_names=input_names
                )
            except (KeyError, TypeError, ValueError) as exc:
                if len(errors) < 50:
                    errors.append(f"{spec.name}@{module_path}: {exc}")
                continue
            result = runtime.collect(spec.name, args=(), kwargs=values)
            collect_attempts += 1
            definition_result = collect_results.setdefault(
                spec.name, {"attempts": 0, "accepted": 0, "rejected": 0, "reasons": {}}
            )
            definition_result["attempts"] += 1
            if result.accepted:
                collect_accepted += 1
                definition_result["accepted"] += 1
            else:
                collect_rejected += 1
                definition_result["rejected"] += 1
                reason = result.reason or "unknown"
                reasons = definition_result["reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
    runtime.flush()
    return {
        "summary": {
            "dump_files": len(files),
            "module_calls": module_calls,
            "collect_attempts": collect_attempts,
            "collect_accepted": collect_accepted,
            "collect_rejected": collect_rejected,
            "missing_bindings": len(missing_bindings),
            "errors": len(errors),
        },
        "collect_results": collect_results,
        "missing_binding_paths": sorted(missing_bindings),
        "errors": errors[:50],
    }


def merge_sglang_workload_shards(
    capture_root: Path, *, dataset_dir: Path, definition_files: list[Path], max_new_workloads: int
) -> dict[str, int]:
    """Merge process-local workload shards and deduplicate them by definition axes."""
    definitions: dict[str, dict[str, Any]] = {}
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        definitions[str(value["name"])] = value

    for name, definition in definitions.items():
        op_type = str(definition["op_type"])
        (dataset_dir / "workloads" / op_type / f"{name}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(dataset_dir / "blob" / "workloads" / op_type / name, ignore_errors=True)

    selected: dict[str, list[tuple[dict[str, Any], Path]]] = {name: [] for name in definitions}
    seen_axes: dict[str, set[str]] = {name: set() for name in definitions}
    for shard in sorted((capture_root / "shards").glob("*")):
        for path in sorted((shard / "workloads").rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                name = item.get("definition")
                if name not in definitions or len(selected[name]) >= max_new_workloads:
                    continue
                axes = item.get("workload", {}).get("axes", {})
                axes_key = json.dumps(axes, sort_keys=True, separators=(",", ":"))
                if axes_key in seen_axes[name]:
                    continue
                _restore_scalar_inputs(item, definitions[name], blob_root=shard)
                seen_axes[name].add(axes_key)
                selected[name].append((item, shard))

    counts: dict[str, int] = {}
    for name, entries in selected.items():
        definition = definitions[name]
        op_type = str(definition["op_type"])
        output_path = dataset_dir / "workloads" / op_type / f"{name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if entries:
            with output_path.open("w", encoding="utf-8") as stream:
                for item, shard in entries:
                    _copy_trace_blobs(item, shard=shard, dataset_dir=dataset_dir)
                    stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        counts[name] = len(entries)
    return counts


def restore_dataset_scalar_inputs(
    dataset_dir: Path, *, definition_files: list[Path]
) -> int:
    """Repair scalar SGLang inputs in an already materialized dataset."""
    restored = 0
    for definition_path in definition_files:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        workload_path = (
            dataset_dir
            / "workloads"
            / str(definition["op_type"])
            / f"{definition['name']}.jsonl"
        )
        if not workload_path.exists():
            continue
        items = [
            json.loads(line)
            for line in workload_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        changed = 0
        for item in items:
            changed += _restore_scalar_inputs(item, definition, blob_root=dataset_dir)
        if changed:
            workload_path.write_text(
                "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in items),
                encoding="utf-8",
            )
            restored += changed
        referenced = {
            dataset_dir / str(input_spec["path"])
            for item in items
            for input_spec in item.get("workload", {}).get("inputs", {}).values()
            if isinstance(input_spec, dict)
            and input_spec.get("type") == "safetensors"
            and isinstance(input_spec.get("path"), str)
        }
        blob_dir = (
            dataset_dir
            / "blob"
            / "workloads"
            / str(definition["op_type"])
            / str(definition["name"])
        )
        for blob_path in blob_dir.glob("*.safetensors"):
            if blob_path not in referenced:
                blob_path.unlink()
    return restored


def _create_shard_runtime(
    shard: Path,
    *,
    definitions_dir: Path | None,
    specs: list[SGLangCaptureSpec],
    definition_files: list[Path] | None = None,
) -> TracingRuntime:
    definitions_output = shard / "definitions"
    names = {spec.name for spec in specs}
    sources = (
        sorted(definition_files)
        if definition_files is not None
        else sorted(definitions_dir.rglob("*.json")) if definitions_dir is not None else []
    )
    for source in sources:
        value = json.loads(source.read_text(encoding="utf-8"))
        name = value.get("name") if isinstance(value, dict) else None
        if not isinstance(name, str) or name not in names:
            continue
        if definitions_dir is not None:
            relative = source.relative_to(definitions_dir)
        else:
            relative = Path(str(value["op_type"])) / source.name
        destination = definitions_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    trace_set = TraceSet.from_path(str(shard))
    config = TracingConfig(
        input_dump_policy="dump_structural",
        filter_policy="keep_first_k_by_axes",
        filter_policy_kwargs={"k": 1},
    )
    registry = TracingConfigRegistry(per_definition={spec.name: config for spec in specs})
    return TracingRuntime(trace_set, registry)


def _capture_spec(definition: dict[str, Any]) -> SGLangCaptureSpec:
    tags = definition.get("tags") if isinstance(definition.get("tags"), list) else []
    modules = tuple(
        tag.removeprefix("sglang_module:")
        for tag in tags
        if isinstance(tag, str) and tag.startswith("sglang_module:")
    )
    callables = tuple(
        tag.removeprefix("sglang_callable:")
        for tag in tags
        if isinstance(tag, str) and tag.startswith("sglang_callable:")
    )
    mappings: dict[str, tuple[str, str]] = {}
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("sglang_input:"):
            continue
        payload = tag.removeprefix("sglang_input:")
        definition_input, separator, source = payload.partition("=")
        kind, kind_separator, source_name = source.partition(":")
        if separator and kind_separator and kind in {"arg", "attr"}:
            mappings[definition_input] = (kind, source_name)
    return SGLangCaptureSpec(
        name=str(definition.get("name") or ""),
        modules=modules,
        callables=callables,
        input_sources=mappings,
    )


def _parse_sglang_dumper_name(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str) or not value.startswith("non_intrusive__"):
        return None
    value = value.removeprefix("non_intrusive__")
    match = re.fullmatch(r"(.+)\.(inputs|output)(?:\.(.+))?", value)
    if match is None:
        return None
    module_path, kind, value_name = match.groups()
    if kind == "inputs" and not value_name:
        return None
    return module_path, "input" if kind == "inputs" else "output", value_name or "output"


def _restore_scalar_inputs(
    item: dict[str, Any], definition: dict[str, Any], *, blob_root: Path
) -> int:
    workload_inputs = item.get("workload", {}).get("inputs", {})
    definition_inputs = definition.get("inputs", {})
    if not isinstance(workload_inputs, dict) or not isinstance(definition_inputs, dict):
        return 0
    restored = 0
    loaded: dict[Path, dict[str, torch.Tensor]] = {}
    for name, definition_input in definition_inputs.items():
        workload_input = workload_inputs.get(name)
        if (
            not isinstance(definition_input, dict)
            or definition_input.get("shape") is not None
            or not isinstance(workload_input, dict)
            or workload_input.get("type") != "safetensors"
        ):
            continue
        raw_path = workload_input.get("path")
        tensor_key = workload_input.get("tensor_key")
        if not isinstance(raw_path, str) or not isinstance(tensor_key, str):
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SGLang workload blob path: {raw_path}")
        source = blob_root / relative
        tensors = loaded.get(source)
        if tensors is None:
            from safetensors.torch import load_file

            tensors = load_file(str(source), device="cpu")
            loaded[source] = tensors
        tensor = tensors.get(tensor_key)
        if tensor is None or tensor.numel() != 1:
            continue
        workload_inputs[name] = {"type": "scalar", "value": tensor.item()}
        restored += 1
    return restored


def _load_module_bindings(capture_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((capture_root / "module_bindings").rglob("*.pt")):
        value = _load_torch_payload(path)
        if not isinstance(value, dict) or not isinstance(value.get("module_path"), str):
            continue
        result.setdefault(str(value["module_path"]), []).append(value)
    return result


def _definition_values_from_dumper(
    spec: SGLangCaptureSpec,
    binding: dict[str, Any],
    call_inputs: dict[str, Any],
    *,
    input_names: list[str],
) -> dict[str, Any]:
    argument_positions = binding.get("argument_positions")
    attributes = binding.get("attributes")
    if not isinstance(argument_positions, dict) or not isinstance(attributes, dict):
        raise TypeError("invalid module binding sidecar")
    values: dict[str, Any] = {}
    for input_name in input_names:
        kind, source_name = spec.input_sources.get(input_name, ("arg", input_name))
        if kind == "attr":
            if input_name not in attributes:
                raise KeyError(f"attribute input {input_name!r} is missing")
            values[input_name] = attributes[input_name]
            continue
        dumper_name = source_name
        if not source_name.isdecimal() and source_name in argument_positions:
            dumper_name = str(argument_positions[source_name])
        values[input_name] = _dumper_argument(call_inputs, dumper_name)
    return values


def _dumper_argument(call_inputs: dict[str, Any], name: str) -> Any:
    if name in call_inputs:
        return call_inputs[name]
    prefix = f"{name}."
    indexed = [
        (int(key.removeprefix(prefix)), value)
        for key, value in call_inputs.items()
        if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
    ]
    if indexed:
        return tuple(value for _, value in sorted(indexed))
    raise KeyError(f"SGLang dumper input {name!r} is missing")


def _load_torch_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _copy_trace_blobs(item: dict[str, Any], *, shard: Path, dataset_dir: Path) -> None:
    inputs = item.get("workload", {}).get("inputs", {})
    if not isinstance(inputs, dict):
        return
    for spec in inputs.values():
        if not isinstance(spec, dict) or spec.get("type") != "safetensors":
            continue
        raw_path = spec.get("path")
        if not isinstance(raw_path, str):
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SGLang workload blob path: {raw_path}")
        source = shard / relative
        destination = dataset_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
