"""GPU execution for the two-stage onboarding pipeline."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tarfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.definition_review import publication_definition
from flashinfer_bench.onboarding.runners.sglang_runner import run_sglang_model
from flashinfer_bench.tracing.flashinfer_logging import (
    flashinfer_definition_dump,
    flashinfer_workload_dump,
    load_fi_definition_files,
)
from flashinfer_bench.tracing.sanitize import sanitize_dumps
from flashinfer_bench.tracing.sglang_dumper_adapter import (
    adapt_sglang_dumper_workloads,
    build_sglang_dumper_workload_filter,
    load_sglang_definition_files,
    merge_sglang_workload_shards,
)
from flashinfer_bench.tracing.sglang_inventory import (
    summarize_module_inventory,
    summarize_sglang_capture_status,
)
from flashinfer_bench.tracing.sglang_worker_capture import sglang_capture_environment

DEFAULT_REMOTE_OUTPUT_DIR = "/tmp/flashinfer-bench-onboarding"
def run_remote_stage(
    plan: dict[str, Any], remote_output_dir: str = DEFAULT_REMOTE_OUTPUT_DIR
) -> dict[str, Any]:
    """Run exactly one GPU stage and return its archived artifacts."""
    stage = plan.get("stage")
    if stage not in {"definitions", "workloads"}:
        raise ValueError("plan.stage must be 'definitions' or 'workloads'")
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    output_dir = Path(remote_output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FLASHINFER_TRACE_OUTPUT_DIR"] = str(output_dir)
    print(f"[flashinfer_bench.onboarding] stage started: {stage}", flush=True)

    if stage == "definitions":
        return _dump_definitions(plan, output_dir)
    return _dump_workloads(plan, output_dir)


def _dump_definitions(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    definitions_dir = output_dir / "definitions"
    capture_root = output_dir / "sglang_capture"
    with (
        flashinfer_definition_dump(definitions_dir),
        sglang_capture_environment(capture_root, mode="inventory"),
    ):
        request_manifest = run_sglang_model(plan)

    files = sorted(definitions_dir.rglob("*.json"))
    module_inventory = summarize_module_inventory(capture_root)
    module_inventory["request_provenance"] = _request_provenance(request_manifest)
    shutil.rmtree(capture_root, ignore_errors=True)
    return {
        "stage": "definitions",
        "definitions_archive_b64": _archive_dir_b64(definitions_dir, arcname="definitions"),
        "request_manifest": request_manifest,
        "module_inventory": module_inventory,
        "summary": {
            "definitions": len(files),
            "sglang_modules": module_inventory["summary"]["modules"],
        },
    }


def _dump_workloads(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    definitions_dir = output_dir / "definitions"
    _write_reviewed_definitions(definitions_dir, plan.get("reviewed_definitions"))
    fi_definition_files, _ = load_fi_definition_files(definitions_dir)
    sglang_definition_files, _ = load_sglang_definition_files(definitions_dir)
    definition_files = list(dict.fromkeys([*fi_definition_files, *sglang_definition_files]))
    skipped = _uncaptured_definitions(definitions_dir, definition_files)
    if not definition_files:
        raise RuntimeError("no reviewed definition contains a supported capture tag")

    dump_dir = output_dir / "native_dumps"
    capture_root = output_dir / "sglang_capture"
    dumper_root = output_dir / "sglang_dumper"
    dataset_dir = output_dir / "output"
    capture_metadata = []
    for source in definition_files:
        destination = dataset_dir / "definitions" / source.relative_to(definitions_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_value = json.loads(source.read_text(encoding="utf-8"))
        published_value, metadata = publication_definition(source_value)
        destination.write_text(
            json.dumps(published_value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata["path"] = str(source.relative_to(definitions_dir))
        capture_metadata.append(metadata)

    include_pattern = ""
    execution_plan = dict(plan)
    dumper_filter = ""
    dumper_selection: dict[str, Any] = {
        "selected_modules": {},
        "missing_modules": [],
    }
    if sglang_definition_files:
        inventory = plan.get("sglang_execution_inventory")
        if not isinstance(inventory, dict):
            inventory = {}
        dumper_filter, dumper_selection = build_sglang_dumper_workload_filter(
            sglang_definition_files, inventory
        )
        if dumper_filter:
            sglang_config = dict(execution_plan.get("sglang") or {})
            sglang_config["dumper_output_dir"] = str(dumper_root)
            sglang_config["dumper_filter"] = dumper_filter
            execution_plan["sglang"] = sglang_config
    with ExitStack() as stack:
        if fi_definition_files:
            include_pattern = stack.enter_context(
                flashinfer_workload_dump(fi_definition_files, dump_dir)
            )
        if sglang_definition_files:
            stack.enter_context(
                sglang_capture_environment(
                    capture_root, mode="workloads", definitions_dir=definitions_dir
                )
            )
        request_manifest = run_sglang_model(execution_plan)

    max_new_workloads = int(plan.get("max_new_workloads") or 20)
    results: dict[str, list[dict[str, Any]]] = {}
    if fi_definition_files:
        results = sanitize_dumps(
            dump_dir=dump_dir,
            definition_files=fi_definition_files,
            flashinfer_trace_dir=dataset_dir,
            replace=True,
            max_new_workloads=max_new_workloads,
        )
    sglang_counts: dict[str, int] = {}
    sglang_diagnostics: dict[str, Any] = {
        "summary": {
            "processes": 0,
            "capture_calls": 0,
            "module_bindings": 0,
            "errors": 0,
        },
        "captures": {},
        "bindings": {},
        "errors": [],
    }
    sglang_dumper: dict[str, Any] = {
        "summary": {
            "dump_files": 0,
            "module_calls": 0,
            "collect_attempts": 0,
            "collect_accepted": 0,
            "collect_rejected": 0,
        },
        "collect_results": {},
        "selection": dumper_selection,
        "errors": [],
    }
    if sglang_definition_files:
        if dumper_filter:
            sglang_dumper = adapt_sglang_dumper_workloads(
                dumper_root,
                capture_root=capture_root,
                definition_files=sglang_definition_files,
            )
            sglang_dumper["selection"] = dumper_selection
        sglang_diagnostics = summarize_sglang_capture_status(capture_root)
        sglang_counts = merge_sglang_workload_shards(
            capture_root,
            dataset_dir=dataset_dir,
            definition_files=sglang_definition_files,
            max_new_workloads=max_new_workloads,
        )
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(capture_root, ignore_errors=True)
    shutil.rmtree(dumper_root, ignore_errors=True)

    fi_names = {path.stem for path in fi_definition_files}
    sglang_names = {path.stem for path in sglang_definition_files}
    collected = [
        {
            "name": path.stem,
            "path": str(path.relative_to(definitions_dir)),
            "backend": "flashinfer" if path.stem in fi_names else "sglang",
            "workloads": (
                len(results.get(path.stem, []))
                if path.stem in fi_names
                else sglang_counts.get(path.stem, 0)
            ),
        }
        for path in definition_files
    ]
    missing = [item for item in collected if item["workloads"] == 0]
    dumper_collect_results = sglang_dumper.get("collect_results", {})
    if isinstance(dumper_collect_results, dict):
        for item in missing:
            result = dumper_collect_results.get(item["name"])
            if not isinstance(result, dict):
                if item["backend"] == "sglang":
                    item["reason"] = "no SGLang dumper call matched this definition"
                continue
            item["collect_result"] = result
            reasons = result.get("reasons")
            if isinstance(reasons, dict) and reasons:
                item["reason"] = max(reasons, key=lambda reason: int(reasons[reason]))
            elif int(result.get("accepted") or 0) > 0:
                item["reason"] = "runtime accepted captures but emitted no workload"
            elif int(result.get("attempts") or 0) == 0:
                item["reason"] = "no SGLang dumper call matched this definition"
    report = {
        "summary": {
            "definitions": len(definition_files),
            "definitions_with_workloads": len(collected) - len(missing),
            "workloads": sum(item["workloads"] for item in collected),
            "missing": len(missing),
            "skipped": len(skipped),
            "ok": not missing,
        },
        "include_pattern": include_pattern,
        "definitions": collected,
        "missing": missing,
        "skipped": skipped,
        "sglang_capture": sglang_diagnostics,
        "sglang_dumper": sglang_dumper,
        "request_profiles": _request_profile_summary(plan),
        "request_provenance": _request_provenance(request_manifest),
    }
    return {
        "stage": "workloads",
        "definitions_sha256": plan.get("definitions_sha256"),
        "output_archive_b64": _archive_dir_b64(dataset_dir, arcname="output"),
        "workload_report": report,
        "capture_metadata": capture_metadata,
        "request_provenance": _request_provenance(request_manifest),
        "summary": report["summary"],
    }


def _request_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest.get("run_id"),
        "model_name": manifest.get("model_name"),
        "generator": manifest.get("generator"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "summary": manifest.get("summary"),
    }


def _write_reviewed_definitions(root: Path, artifacts: Any) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("workload stage requires reviewed_definitions")
    shutil.rmtree(root, ignore_errors=True)
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"reviewed definition #{index} must be an object")
        raw_path = artifact.get("path")
        data = artifact.get("data")
        if not isinstance(raw_path, str) or not isinstance(data, dict):
            raise ValueError(f"reviewed definition #{index} must contain path and data")
        relative = Path(raw_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe reviewed definition path: {raw_path}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _uncaptured_definitions(root: Path, captured: list[Path]) -> list[dict[str, str]]:
    captured_paths = {path.resolve() for path in captured}
    return [
        {
            "path": str(path.relative_to(root)),
            "reason": "missing fi_api, sglang_module, or sglang_callable tag",
        }
        for path in sorted(root.rglob("*.json"))
        if path.resolve() not in captured_paths
    ]


def _request_profile_summary(plan: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for scenario in plan.get("request_scenarios") or []:
        if not isinstance(scenario, dict):
            continue
        if scenario.get("source") == "inferencex_fixed_seq":
            summaries.append(
                {
                    key: scenario[key]
                    for key in (
                        "name",
                        "source",
                        "random_input_len",
                        "random_output_len",
                        "random_range_ratio",
                        "random_prefix_len",
                        "num_prompts",
                        "max_concurrency",
                        "seed",
                        "context_fraction",
                    )
                    if key in scenario
                }
            )
    return summaries


def _archive_dir_b64(root: Path, *, arcname: str) -> str:
    archive_path = root.parent / f"{arcname}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root, arcname=arcname)
    encoded = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    archive_path.unlink(missing_ok=True)
    return encoded
