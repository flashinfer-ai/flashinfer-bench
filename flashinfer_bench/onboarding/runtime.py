"""Shared run configuration and execution helpers for onboarding stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.definition_review import render_definition_review
from flashinfer_bench.onboarding.planning import build_stage_plan
from flashinfer_bench.onboarding.runners.stage_runner import run_stage
from flashinfer_bench.serve.inferencex_requests import request_manifest_digest
from flashinfer_bench.tracing.flashinfer_logging import (
    infer_sglang_pass_settings,
    load_fi_definition_files,
)

RUN_CONFIG_KEYS = {
    "model_name",
    "tp_size",
    "batch_sizes",
    "max_new_tokens",
    "supplemental_runs",
    "disable_cuda_graph",
    "enable_piecewise_cuda_graph",
    "force_flashinfer_backends",
    "mem_fraction_static",
    "cuda_graph_max_bs",
    "engine_kwargs",
    "max_new_workloads",
    "isl",
    "osl",
    "random_range_ratio",
    "requests_per_concurrency",
    "seed",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower() or "run"


def run_dir(run: str) -> Path:
    path = Path(run)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"ERROR: invalid run name: {run}")
    return Path("runs").joinpath(*(_slug(part) for part in path.parts))


def load_config(run_path: Path) -> dict[str, Any]:
    path = run_path / "config" / "run_config.json"
    if not path.exists():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid run config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"ERROR: run config must be a JSON object: {path}")
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise SystemExit(f"ERROR: unsupported run_config.json fields: {unknown}")
    return config


def _runtime_value(
    args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None
) -> Any:
    value = getattr(args, key, None)
    return value if value is not None else config.get(key, default)


def _required_runtime_value(
    args: argparse.Namespace, config: dict[str, Any], key: str
) -> Any:
    value = _runtime_value(args, config, key)
    if value is None or value == "":
        raise SystemExit(
            f"ERROR: --{key.replace('_', '-')} is required or must be set in config/run_config.json"
        )
    return value


def resolve_config(args: argparse.Namespace, run_path: Path) -> dict[str, Any]:
    config = load_config(run_path)
    resolved = dict(config)
    resolved.update(
        {
            "model_name": _required_runtime_value(args, config, "model_name"),
            "tp_size": int(_runtime_value(args, config, "tp_size", 1)),
            "batch_sizes": config.get("batch_sizes", [1, 2, 4, 8]),
            "max_new_tokens": int(config.get("max_new_tokens", 16)),
            "supplemental_runs": config.get("supplemental_runs", []),
            "max_new_workloads": int(config.get("max_new_workloads", 20)),
            "isl": int(_runtime_value(args, config, "isl", 1024)),
            "osl": int(_runtime_value(args, config, "osl", 8)),
            "random_range_ratio": float(
                _runtime_value(args, config, "random_range_ratio", 1.0)
            ),
            "requests_per_concurrency": int(
                config.get("requests_per_concurrency", 10)
            ),
            "seed": int(_runtime_value(args, config, "seed", 0)),
        }
    )
    if resolved["tp_size"] < 1:
        raise SystemExit("ERROR: tp_size must be at least 1")
    if resolved["requests_per_concurrency"] < 1:
        raise SystemExit("ERROR: requests_per_concurrency must be at least 1")
    path = run_path / "config" / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return resolved


def plan_for_stage(
    *,
    stage: str,
    run_path: Path,
    config: dict[str, Any],
    reviewed_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pass_modes = None
    page_sizes = None
    if reviewed_definitions is not None:
        files, _ = load_fi_definition_files(run_path / "definitions")
        pass_modes, page_sizes = infer_sglang_pass_settings(files)
    plan = build_stage_plan(
        stage=stage,
        config=config,
        reviewed_definitions=reviewed_definitions,
        pass_modes=pass_modes,
        page_sizes=page_sizes,
    )
    if reviewed_definitions is not None:
        plan["definitions_sha256"] = definitions_digest(run_path / "definitions")
    plan["run_id"] = run_path.as_posix()
    return plan


def load_request_manifest(run_path: Path) -> dict[str, Any] | None:
    path = run_path / "reports" / "evidence" / "request_manifest.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: request manifest must be a JSON object: {path}")
    expected = value.get("manifest_sha256")
    if not isinstance(expected, str) or request_manifest_digest(value) != expected:
        raise SystemExit(f"ERROR: request manifest digest is invalid: {path}")
    return value


def load_definition_artifacts(definitions_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(definitions_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        artifacts.append({"path": str(path.relative_to(definitions_dir)), "data": data})
    return artifacts


def definitions_digest(definitions_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(definitions_dir.rglob("*.json")):
        digest.update(str(path.relative_to(definitions_dir)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_definition_review(run_path: Path, report: dict[str, Any]) -> None:
    reports_dir = run_path / "reports"
    write_json(reports_dir / "definition_report.json", report)
    (reports_dir / "definition_review.md").write_text(
        render_definition_review(report), encoding="utf-8"
    )


def load_sglang_execution_inventory(run_path: Path) -> dict[str, Any] | None:
    path = run_path / "reports" / "evidence" / "sglang_execution_inventory.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def sglang_inventory(run_path: Path) -> set[str] | None:
    value = load_sglang_execution_inventory(run_path)
    if value is None:
        return None
    modules = value.get("modules") if isinstance(value, dict) else None
    if not isinstance(modules, list):
        return None
    return {
        str(item["class_path"])
        for item in modules
        if isinstance(item, dict) and isinstance(item.get("class_path"), str)
    }
