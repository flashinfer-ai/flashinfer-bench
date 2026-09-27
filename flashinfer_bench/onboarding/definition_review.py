"""Deterministic checks for the reviewed definition directory."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

from flashinfer_bench.data.definition import Definition

_INTERNAL_TAG_PREFIXES = (
    "sglang_module:",
    "sglang_callable:",
    "sglang_input:",
)
_PUBLIC_STATUS_TAGS = {
    "status:verified",
    "status:unverified",
    "status:reference",
}


def publication_definition(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate publishable definition fields from capture-only metadata."""
    published = copy.deepcopy(value)
    tags = published.get("tags")
    tags = list(tags) if isinstance(tags, list) else []
    capture_tags = [
        tag
        for tag in tags
        if isinstance(tag, str) and tag.startswith(_INTERNAL_TAG_PREFIXES)
    ]
    source_statuses = [
        tag for tag in tags if isinstance(tag, str) and tag.startswith("status:")
    ]
    published_tags = [
        tag
        for tag in tags
        if isinstance(tag, str)
        and not tag.startswith(_INTERNAL_TAG_PREFIXES)
        and tag != "status:source_reviewed"
    ]
    if "status:source_reviewed" in source_statuses and not any(
        tag in _PUBLIC_STATUS_TAGS for tag in published_tags
    ):
        published_tags.append("status:unverified")
    published["tags"] = published_tags
    sidecar = {
        "name": published.get("name"),
        "op_type": published.get("op_type"),
        "capture_tags": capture_tags,
        "source_status_tags": source_statuses,
    }
    return published, sidecar


def organize_definition_directory(definitions_dir: Path) -> None:
    """Move flat native-trace files into the stable ``op_type/name.json`` layout."""
    for path in sorted(definitions_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        op_type = value.get("op_type")
        if not (
            isinstance(name, str)
            and re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name)
            and isinstance(op_type, str)
            and re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", op_type)
        ):
            continue
        destination = definitions_dir / op_type / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() == path.read_bytes():
                path.unlink()
            continue
        path.replace(destination)


def check_definition_directory(
    definitions_dir: Path,
    *,
    sglang_inventory: set[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Validate definition templates before workload collection."""
    checked: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not definitions_dir.exists():
        errors.append(
            {"path": str(definitions_dir), "reason": "definitions directory does not exist"}
        )
    else:
        for path in sorted(definitions_dir.rglob("*.json")):
            relative = path.relative_to(definitions_dir)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append({"path": str(relative), "reason": f"invalid JSON: {exc}"})
                continue
            if not isinstance(value, dict):
                errors.append({"path": str(relative), "reason": "definition must be a JSON object"})
                continue

            name = value.get("name")
            op_type = value.get("op_type")
            if not isinstance(name, str) or not re.fullmatch(
                r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name
            ):
                errors.append({"path": str(relative), "reason": "name must be lower snake_case"})
            if not isinstance(op_type, str) or not op_type:
                errors.append(
                    {"path": str(relative), "reason": "op_type must be a non-empty string"}
                )
            if isinstance(name, str) and path.stem != name:
                errors.append({"path": str(relative), "reason": f"filename must be {name}.json"})
            if isinstance(op_type, str) and path.parent.name != op_type:
                errors.append(
                    {"path": str(relative), "reason": f"parent directory must be {op_type}"}
                )

            try:
                Definition.model_validate(value)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - keep the complete schema boundary in one report
                errors.append({"path": str(relative), "reason": _format_schema_error(exc)})

            tags = value.get("tags")
            fi_apis = (
                [
                    tag.removeprefix("fi_api:")
                    for tag in tags
                    if isinstance(tag, str) and tag.startswith("fi_api:")
                ]
                if isinstance(tags, list)
                else []
            )
            sglang_modules = (
                [
                    tag.removeprefix("sglang_module:")
                    for tag in tags
                    if isinstance(tag, str) and tag.startswith("sglang_module:")
                ]
                if isinstance(tags, list)
                else []
            )
            sglang_callables = (
                [
                    tag.removeprefix("sglang_callable:")
                    for tag in tags
                    if isinstance(tag, str) and tag.startswith("sglang_callable:")
                ]
                if isinstance(tags, list)
                else []
            )
            for api in fi_apis:
                if api.rsplit(".", 1)[-1][:1].isupper():
                    errors.append(
                        {
                            "path": str(relative),
                            "reason": (
                                f"fi_api must name an exact callable, not wrapper class {api}"
                            ),
                        }
                    )
            capture_kinds = sum(
                bool(items) for items in (fi_apis, sglang_modules, sglang_callables)
            )
            if capture_kinds > 1:
                errors.append(
                    {
                        "path": str(relative),
                        "reason": (
                            "definition must use exactly one capture backend: fi_api, "
                            "sglang_module, or sglang_callable"
                        ),
                    }
                )
            if (sglang_modules or sglang_callables) and isinstance(name, str) and isinstance(
                op_type, str
            ):
                for model_token in _model_family_tokens(model_name):
                    components = [*name.split("_"), *op_type.split("_")]
                    if model_token in components:
                        errors.append(
                            {
                                "path": str(relative),
                                "reason": (
                                    "non-FI name/op_type must describe semantics, not model "
                                    f"provenance: found model token {model_token!r}; use a "
                                    "model:<slug> tag"
                                ),
                            }
                        )
                        break
            for capture_path in [*sglang_modules, *sglang_callables]:
                if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", capture_path):
                    errors.append(
                        {
                            "path": str(relative),
                            "reason": f"invalid exact SGLang capture path: {capture_path}",
                        }
                    )
            if sglang_inventory is not None:
                for module_path in sglang_modules:
                    if module_path not in sglang_inventory:
                        errors.append(
                            {
                                "path": str(relative),
                                "reason": (
                                    f"sglang_module was not observed in the short model pass: "
                                    f"{module_path}"
                                ),
                            }
                        )
            for reason in _sglang_input_tag_errors(value, sglang_modules):
                errors.append({"path": str(relative), "reason": reason})
            for reason in _definition_semantic_errors(value):
                errors.append({"path": str(relative), "reason": reason})
            for reason in _reference_output_errors(value):
                errors.append({"path": str(relative), "reason": reason})
            collectable = bool(fi_apis or sglang_modules or sglang_callables)
            if not collectable:
                warnings.append(
                    {
                        "path": str(relative),
                        "reason": (
                            "no fi_api, sglang_module, or sglang_callable tag; "
                            "dump-workload will skip it"
                        ),
                    }
                )
            checked.append(
                {
                    "path": str(relative),
                    "name": name,
                    "op_type": op_type,
                    "fi_apis": fi_apis,
                    "sglang_modules": sglang_modules,
                    "sglang_callables": sglang_callables,
                    "capture_backend": (
                        "flashinfer"
                        if fi_apis
                        else "sglang" if sglang_modules or sglang_callables else None
                    ),
                    "collectable": collectable,
                }
            )
        if not checked and not errors:
            errors.append(
                {"path": str(definitions_dir), "reason": "no definition JSON files found"}
            )

    return {
        "summary": {
            "ok": not errors and bool(checked),
            "definitions": len(checked),
            "collectable": sum(1 for item in checked if item["collectable"]),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "definitions": checked,
        "errors": errors,
        "warnings": warnings,
    }


def _model_family_tokens(model_name: str | None) -> set[str]:
    """Return conservative model-family tokens that must remain provenance-only."""
    if not model_name:
        return set()
    leaf = model_name.rsplit("/", 1)[-1].lower()
    parts = re.findall(r"[a-z]+|\d+", leaf)
    if not parts:
        return set()
    tokens = {parts[0]}
    if len(parts) > 1 and parts[1].isdigit():
        tokens.add(f"{parts[0]}{parts[1]}")
    return {token for token in tokens if len(token) >= 4}


def render_definition_review(report: dict[str, Any]) -> str:
    """Render the first human-review checkpoint."""
    summary = report["summary"]
    lines = [
        "# Definition Review",
        "",
        "Review and edit `definitions/` in place. `dump-workload` reads exactly these files.",
        "",
        "## Status",
        "",
        f"- schema ready: {'yes' if summary['ok'] else 'no'}",
        f"- definitions: {summary['definitions']}",
        f"- workload-collectable (`fi_api` / `sglang_*`): {summary['collectable']}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
    ]
    for heading, key in (("Errors", "errors"), ("Warnings", "warnings")):
        items = report.get(key, [])
        if items:
            lines.extend(["", f"## {heading}", ""])
            lines.extend(f"- `{item['path']}`: {item['reason']}" for item in items)
    lines.extend(
        [
            "",
            "## Approval",
            "",
            "Running `dump-workload` is the approval action for the current `definitions/` snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def render_workload_review(
    report: dict[str, Any],
    dataset_validation: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> str:
    """Render the second human-review checkpoint."""
    summary = report["summary"]
    validation_ok = bool(dataset_validation.get("ok"))
    lines = [
        "# Workload Review",
        "",
        "## Status",
        "",
        f"- workload collection ready: {'yes' if summary['ok'] else 'no'}",
        f"- dataset validation: {'pass' if validation_ok else 'fail'}",
        f"- reviewed definitions: {summary['definitions']}",
        f"- definitions with workloads: {summary['definitions_with_workloads']}",
        f"- workloads: {summary['workloads']}",
        f"- missing definitions: {summary['missing']}",
        f"- skipped definitions without a capture backend: {summary['skipped']}",
    ]
    for heading, key in (("Missing Workloads", "missing"), ("Skipped Definitions", "skipped")):
        items = report.get(key, [])
        if items:
            lines.extend(["", f"## {heading}", ""])
            for item in items:
                name = item.get("name") or item.get("path") or "unknown"
                detail = item.get("reason") or f"workloads={item.get('workloads', 0)}"
                lines.append(f"- `{name}`: {detail}")
    if analysis is not None:
        lines.extend(
            [
                "",
                "## Agent Definition Analysis",
                "",
                f"- agent return code: {analysis['returncode']}",
                f"- definitions changed: {'yes' if analysis['definitions_changed'] else 'no'}",
            ]
        )
        if analysis["requires_review_and_rerun"]:
            lines.append("- next: review `definitions/`, then rerun `dump-workload`")
    lines.extend(
        [
            "",
            "## Output",
            "",
            "- definitions: `output/definitions/`",
            "- workloads: `output/workloads/`",
            "- tensor blobs: `output/blob/`",
            "- machine report: `reports/run_report.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _format_schema_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            items = errors()
        except Exception:  # noqa: BLE001
            items = None
        if isinstance(items, list) and items:
            parts = []
            for item in items[:3]:
                if not isinstance(item, dict):
                    continue
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = str(item.get("msg") or "invalid")
                parts.append(f"{location}: {message}" if location else message)
            if parts:
                suffix = f"; +{len(items) - len(parts)} more" if len(items) > len(parts) else ""
                return "formal Definition schema: " + "; ".join(parts) + suffix
    return "formal Definition schema: " + str(exc).splitlines()[0]


def _definition_semantic_errors(definition: dict[str, Any]) -> list[str]:
    """Check source-independent invariants not expressed by the formal schema."""
    op_type = definition.get("op_type")
    name = definition.get("name")
    axes = definition.get("axes")
    if not (
        isinstance(op_type, str)
        and op_type.startswith("gqa_")
        and isinstance(name, str)
        and isinstance(axes, dict)
    ):
        return []

    values = {
        axis: _const_axis_value(axes, axis)
        for axis in ("num_qo_heads", "num_kv_heads", "head_dim", "page_size")
    }
    errors = []
    qo_heads = values["num_qo_heads"]
    kv_heads = values["num_kv_heads"]
    if (
        qo_heads is not None
        and kv_heads is not None
        and (kv_heads <= 0 or qo_heads < kv_heads or qo_heads % kv_heads != 0)
    ):
        errors.append(
            "GQA head axes must satisfy num_qo_heads >= num_kv_heads and "
            "num_qo_heads % num_kv_heads == 0"
        )

    encoded_axes = {"num_qo_heads": "h", "num_kv_heads": "kv", "head_dim": "d", "page_size": "ps"}
    tokens = set(name.split("_"))
    missing = [
        f"{prefix}{values[axis]}"
        for axis, prefix in encoded_axes.items()
        if values[axis] is not None and f"{prefix}{values[axis]}" not in tokens
    ]
    if missing:
        errors.append(f"definition name must encode const axes: {', '.join(missing)}")
    return errors


def _sglang_input_tag_errors(definition: dict[str, Any], sglang_modules: list[str]) -> list[str]:
    tags = definition.get("tags") if isinstance(definition.get("tags"), list) else []
    inputs = definition.get("inputs") if isinstance(definition.get("inputs"), dict) else {}
    errors: list[str] = []
    mapped: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("sglang_input:"):
            continue
        payload = tag.removeprefix("sglang_input:")
        input_name, separator, source = payload.partition("=")
        kind, kind_separator, source_name = source.partition(":")
        if (
            not separator
            or not kind_separator
            or kind not in {"arg", "attr"}
            or not input_name
            or not source_name
        ):
            errors.append(
                f"invalid {tag!r}; expected sglang_input:<definition_input>=arg:<name-or-index> "
                "or attr:<path>"
            )
            continue
        if input_name not in inputs:
            errors.append(f"sglang_input maps unknown definition input: {input_name}")
        if input_name in mapped:
            errors.append(f"sglang_input maps {input_name} more than once")
        if kind == "attr" and not sglang_modules:
            errors.append("sglang_input attr sources require sglang_module capture")
        mapped.add(input_name)
    return errors


def _const_axis_value(axes: dict[str, Any], name: str) -> int | None:
    spec = axes.get(name)
    if not isinstance(spec, dict) or spec.get("type") != "const":
        return None
    value = spec.get("value")
    return value if type(value) is int else None


def _reference_output_errors(definition: dict[str, Any]) -> list[str]:
    reference = definition.get("reference")
    outputs = definition.get("outputs")
    if not isinstance(reference, str) or not isinstance(outputs, dict):
        return []
    try:
        module = ast.parse(reference)
    except SyntaxError:
        return []
    run = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    if run is None:
        return []
    arities = set()
    for node in ast.walk(run):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            arities.add(0)
        elif isinstance(node.value, (ast.Tuple, ast.List)):
            arities.add(len(node.value.elts))
        else:
            arities.add(1)
    expected = len(outputs)
    if not arities:
        arities.add(0)
    if arities != {expected}:
        actual = ", ".join(str(value) for value in sorted(arities))
        return [f"reference run returns {actual} values but outputs declares {expected}"]
    return []
