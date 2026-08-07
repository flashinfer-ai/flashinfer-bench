"""Summarize SGLang module observations and worker capture status."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def summarize_module_inventory(capture_root: Path) -> dict[str, Any]:
    """Combine worker module observations into one source-evidence report."""
    modules: dict[str, dict[str, Any]] = {}
    files = sorted((capture_root / "inventory").glob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            class_path = item.get("class_path")
            if not isinstance(class_path, str):
                continue
            existing = modules.setdefault(class_path, dict(item))
            processes = set(existing.pop("observed_processes", []))
            processes.add(int(item.get("pid") or 0))
            existing["observed_processes"] = sorted(process for process in processes if process)
            module_paths = set(existing.pop("module_paths", []))
            raw_paths = item.get("module_paths")
            if isinstance(raw_paths, list):
                module_paths.update(path for path in raw_paths if isinstance(path, str) and path)
            existing["module_paths"] = sorted(module_paths)
            layer_indices = set(existing.pop("layer_indices", []))
            raw_indices = item.get("layer_indices")
            if isinstance(raw_indices, list):
                layer_indices.update(index for index in raw_indices if type(index) is int)
            existing["layer_indices"] = sorted(layer_indices)
            observations: set[tuple[int, str, int | None]] = set()
            for observation in existing.pop("module_observations", []):
                if not isinstance(observation, dict):
                    continue
                pid = observation.get("pid")
                module_path = observation.get("module_path")
                layer_index = observation.get("layer_index")
                if type(pid) is int and isinstance(module_path, str):
                    observations.add(
                        (pid, module_path, layer_index if type(layer_index) is int else None)
                    )
            raw_observations = item.get("module_observations")
            if isinstance(raw_observations, list):
                for observation in raw_observations:
                    if not isinstance(observation, dict):
                        continue
                    pid = observation.get("pid")
                    module_path = observation.get("module_path")
                    layer_index = observation.get("layer_index")
                    if type(pid) is int and isinstance(module_path, str):
                        observations.add(
                            (
                                pid,
                                module_path,
                                layer_index if type(layer_index) is int else None,
                            )
                        )
            elif type(item.get("pid")) is int:
                for module_path in raw_paths if isinstance(raw_paths, list) else []:
                    if isinstance(module_path, str) and module_path:
                        observations.add(
                            (item["pid"], module_path, _layer_index_from_path(module_path))
                        )
            existing["module_observations"] = [
                {
                    "pid": pid,
                    "module_path": module_path,
                    "layer_index": layer_index,
                }
                for pid, module_path, layer_index in sorted(
                    observations,
                    key=lambda observation: (observation[0], observation[1]),
                )
            ]
    return {
        "summary": {"modules": len(modules), "worker_files": len(files)},
        "modules": [modules[name] for name in sorted(modules)],
    }


def summarize_sglang_capture_status(capture_root: Path) -> dict[str, Any]:
    """Combine process-local capture counters and bounded error samples."""
    captures: dict[str, int] = {}
    bindings: dict[str, int] = {}
    errors: list[str] = []
    files = sorted((capture_root / "status").glob("*.json"))
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        for name, count in item.get("captures", {}).items():
            captures[str(name)] = captures.get(str(name), 0) + int(count)
        for name, count in item.get("bindings", {}).items():
            bindings[str(name)] = bindings.get(str(name), 0) + int(count)
        errors.extend(str(error) for error in item.get("errors", []))
    return {
        "summary": {
            "processes": len(files),
            "capture_calls": sum(captures.values()),
            "module_bindings": sum(bindings.values()),
            "errors": len(errors),
        },
        "captures": captures,
        "bindings": bindings,
        "errors": errors[:50],
    }


def _layer_index_from_path(path: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", path)
    return int(match.group(1)) if match else None


def _module_path_order(path: str) -> tuple[int, str]:
    layer_index = _layer_index_from_path(path)
    return (layer_index if layer_index is not None else sys.maxsize, path)
