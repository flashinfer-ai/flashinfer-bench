"""Configure FlashInfer's native tensor-dump logger from reviewed definitions."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_INT_DTYPES = {"int32", "int64"}
_DUMP_ENV = {
    "FLASHINFER_TRACE_DUMP": "0",
    "FLASHINFER_LOGLEVEL": "10",
    "FLASHINFER_DUMP_SAFETENSORS": "1",
    "FLASHINFER_DUMP_EXCLUDE": "*.__init__",
    "FLASHINFER_DUMP_MAX_COUNT": "1000",
    "FLASHINFER_DUMP_MAX_SIZE_GB": "30",
    "FLASHINFER_USE_CUDA_NORM": "1",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
}


def fi_api_runtime_names(api: str) -> list[str]:
    """Translate an exact ``fi_api`` path into native logger function names."""
    parts = api.split(".")
    if not parts or not parts[-1]:
        return []
    name = parts[-1]
    if len(parts) >= 2 and parts[-2][:1].isupper():
        wrapper = parts[-2]
        names = [f"{wrapper}.{name}"]
        if name == "run" and "Ragged" in wrapper:
            names.extend((f"{wrapper}.forward", f"{wrapper}.forward_return_lse"))
        return list(dict.fromkeys(names))
    if name[:1].isupper():
        return []
    return [name]


@contextmanager
def flashinfer_definition_dump(definitions_dir: Path) -> Iterator[None]:
    """Enable FlashInfer definition tracing for one short model pass."""
    shutil.rmtree(definitions_dir, ignore_errors=True)
    definitions_dir.mkdir(parents=True, exist_ok=True)
    updates = {"FLASHINFER_TRACE_DUMP": "1", "FLASHINFER_TRACE_DUMP_DIR": str(definitions_dir)}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_fi_definition_files(definitions_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Return reviewed definitions backed by a FlashInfer API and skipped entries."""
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(definitions_dir.rglob("*.json")):
        try:
            definition = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid definition JSON {path}: {exc}") from exc
        if not isinstance(definition, dict):
            raise ValueError(f"definition must be a JSON object: {path}")
        tags = definition.get("tags")
        has_fi_api = isinstance(tags, list) and any(
            isinstance(tag, str) and tag.startswith("fi_api:") for tag in tags
        )
        if has_fi_api:
            files.append(path)
        else:
            skipped.append({"path": str(path), "reason": "missing_fi_api_tag"})
    return files, skipped


def build_fi_include_pattern(definition_files: list[Path]) -> str:
    """Build ``FLASHINFER_DUMP_INCLUDE`` from definition ``fi_api:`` tags."""
    api_to_definitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in definition_files:
        definition = json.loads(path.read_text(encoding="utf-8"))
        for tag in definition.get("tags", []):
            if isinstance(tag, str) and tag.startswith("fi_api:"):
                api_to_definitions[tag.removeprefix("fi_api:")].append(definition)

    patterns: list[str] = []
    for api, definitions in sorted(api_to_definitions.items()):
        runtime_names = fi_api_runtime_names(api)
        patterns.extend(runtime_names)
        parts = api.split(".")
        wrapper = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else None
        if wrapper:
            needs_plan = any(
                any(
                    isinstance(spec, dict)
                    and spec.get("shape") is not None
                    and spec.get("dtype") in _INT_DTYPES
                    for spec in definition.get("inputs", {}).values()
                )
                for definition in definitions
            )
            if needs_plan:
                patterns.append(f"{wrapper}.plan")

    return ",".join(dict.fromkeys(patterns))


def infer_sglang_pass_settings(definition_files: list[Path]) -> tuple[list[str], list[int]]:
    """Infer the minimal SGLang passes needed by reviewed definitions."""
    needs_paged_prefill = False
    needs_default = False
    page_sizes: set[int] = set()
    for path in definition_files:
        definition = json.loads(path.read_text(encoding="utf-8"))
        tags = definition.get("tags") if isinstance(definition.get("tags"), list) else []
        apis = [
            tag.removeprefix("fi_api:")
            for tag in tags
            if isinstance(tag, str) and tag.startswith("fi_api:")
        ]
        is_paged_prefill = any("BatchPrefillWithPagedKVCacheWrapper" in api for api in apis)
        needs_paged_prefill = needs_paged_prefill or is_paged_prefill
        needs_default = needs_default or not is_paged_prefill
        page_axis = (
            definition.get("axes", {}).get("page_size")
            if isinstance(definition.get("axes"), dict)
            else None
        )
        if (
            isinstance(page_axis, dict)
            and page_axis.get("type") == "const"
            and type(page_axis.get("value")) is int
        ):
            page_sizes.add(int(page_axis["value"]))
    if needs_paged_prefill and needs_default:
        modes = ["both"]
    elif needs_paged_prefill:
        modes = ["paged"]
    else:
        modes = ["default"]
    return modes, sorted(page_sizes)


@contextmanager
def flashinfer_workload_dump(definition_files: list[Path], dump_dir: Path) -> Iterator[str]:
    """Enable native FlashInfer dumps for one SGLang workload pass."""
    include_pattern = build_fi_include_pattern(definition_files)
    if not include_pattern:
        raise ValueError("reviewed definitions contain no usable fi_api tags")

    shutil.rmtree(dump_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    updates = {
        **_DUMP_ENV,
        "FLASHINFER_DUMP_DIR": str(dump_dir),
        "FLASHINFER_DUMP_INCLUDE": include_pattern,
        "FLASHINFER_LOGDEST": str(dump_dir.parent / "flashinfer_api.log"),
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield include_pattern
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
