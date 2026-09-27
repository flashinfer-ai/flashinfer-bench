"""Definition capture and Agent review stages."""

from __future__ import annotations

import argparse
import json

from flashinfer_bench.onboarding.agent_analysis import run_definition_analysis
from flashinfer_bench.onboarding.definition_review import (
    check_definition_directory,
    organize_definition_directory,
)
from flashinfer_bench.onboarding.runtime import (
    definitions_digest,
    load_config,
    plan_for_stage,
    resolve_config,
    run_dir,
    run_stage,
    sglang_inventory,
    write_definition_review,
    write_json,
)


def run_analyze_definition(args: argparse.Namespace) -> int:
    run_path = run_dir(args.run)
    config = load_config(run_path)
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise SystemExit("ERROR: config/run_config.json must contain model_name")
    definitions_dir = run_path / "definitions"
    inventory = sglang_inventory(run_path)
    if inventory is None:
        raise SystemExit(
            "ERROR: reports/evidence/sglang_execution_inventory.json "
            "is required for local analysis"
        )
    report = check_definition_directory(
        definitions_dir, sglang_inventory=inventory, model_name=model_name
    )
    report_path = run_path / "reports" / "definition_report.json"
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(existing_report, dict) and isinstance(
            existing_report.get("remote"), dict
        ):
            report["remote"] = existing_report["remote"]
    before = definitions_digest(definitions_dir)
    returncode = run_definition_analysis(
        run_dir=run_path,
        model_name=model_name,
        report=report,
        source="definition_reanalysis",
    )
    if returncode == 0:
        organize_definition_directory(definitions_dir)
        remote = report.get("remote")
        report = check_definition_directory(
            definitions_dir, sglang_inventory=inventory, model_name=model_name
        )
        if isinstance(remote, dict):
            report["remote"] = remote
    report["agent_analysis"] = {
        "returncode": returncode,
        "definitions_changed": definitions_digest(definitions_dir) != before,
    }
    write_definition_review(run_path, report)
    print(f"definitions: {definitions_dir}")
    print(f"definition review: {run_path / 'reports' / 'definition_review.md'}")
    print(f"schema ready: {report['summary']['ok']}")
    return 0 if returncode == 0 and report["summary"]["ok"] else 1


def run_dump_definition(args: argparse.Namespace) -> int:
    run_path = run_dir(args.run)
    definitions_dir = run_path / "definitions"
    if definitions_dir.exists() and any(definitions_dir.rglob("*.json")) and not args.overwrite:
        raise SystemExit(
            f"ERROR: {definitions_dir} already contains reviewed files; pass --overwrite to replace them"
        )
    config = resolve_config(args, run_path)
    plan = plan_for_stage(stage="definitions", run_path=run_path, config=config)
    result = run_stage(plan, run_path, config)
    reports_dir = run_path / "reports"
    evidence_dir = reports_dir / "evidence"
    module_inventory = result.get("module_inventory")
    if isinstance(module_inventory, dict):
        write_json(evidence_dir / "sglang_execution_inventory.json", module_inventory)
    request_manifest = result.get("request_manifest")
    if not isinstance(request_manifest, dict):
        raise SystemExit("ERROR: remote definition stage returned no request manifest")
    write_json(evidence_dir / "request_manifest.json", request_manifest)
    organize_definition_directory(definitions_dir)
    inventory = sglang_inventory(run_path)
    report = check_definition_directory(
        definitions_dir,
        sglang_inventory=inventory,
        model_name=str(config["model_name"]),
    )
    report["remote"] = result.get("summary", {})
    if args.agent:
        before = definitions_digest(definitions_dir)
        returncode = run_definition_analysis(
            run_dir=run_path,
            model_name=str(config["model_name"]),
            report=report,
            source="definition_authoring",
        )
        if returncode == 0:
            organize_definition_directory(definitions_dir)
            report = check_definition_directory(
                definitions_dir,
                sglang_inventory=inventory,
                model_name=str(config["model_name"]),
            )
            report["remote"] = result.get("summary", {})
        report["agent_analysis"] = {
            "returncode": returncode,
            "definitions_changed": definitions_digest(definitions_dir) != before,
        }
    write_definition_review(run_path, report)
    print(f"definitions: {definitions_dir}")
    print(f"definition review: {run_path / 'reports' / 'definition_review.md'}")
    print(f"schema ready: {report['summary']['ok']}")
    return 0
