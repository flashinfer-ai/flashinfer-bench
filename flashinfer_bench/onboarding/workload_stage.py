"""Workload capture and canonical dataset validation stage."""

from __future__ import annotations

import argparse

from flashinfer_bench.onboarding.agent_analysis import run_definition_analysis
from flashinfer_bench.onboarding.definition_review import (
    check_definition_directory,
    organize_definition_directory,
    render_workload_review,
)
from flashinfer_bench.onboarding.runtime import (
    definitions_digest,
    load_definition_artifacts,
    load_request_manifest,
    load_sglang_execution_inventory,
    plan_for_stage,
    resolve_config,
    run_dir,
    run_stage,
    write_definition_review,
    write_json,
)


def run_dump_workload(args: argparse.Namespace) -> int:
    run_path = run_dir(args.run)
    config = resolve_config(args, run_path)
    if args.max_new_workloads is not None:
        if args.max_new_workloads < 1:
            raise SystemExit("ERROR: --max-new-workloads must be at least 1")
        config["max_new_workloads"] = args.max_new_workloads
        write_json(run_path / "config" / "run_config.json", config)

    definitions_dir = run_path / "definitions"
    inventory_report = load_sglang_execution_inventory(run_path)
    inventory_modules = (
        inventory_report.get("modules") if isinstance(inventory_report, dict) else None
    )
    inventory = (
        {
            str(item["class_path"])
            for item in inventory_modules
            if isinstance(item, dict) and isinstance(item.get("class_path"), str)
        }
        if isinstance(inventory_modules, list)
        else None
    )
    definition_report = check_definition_directory(
        definitions_dir, sglang_inventory=inventory, model_name=str(config["model_name"])
    )
    if not definition_report["summary"]["ok"] and args.agent:
        returncode = run_definition_analysis(
            run_dir=run_path,
            model_name=str(config["model_name"]),
            report=definition_report,
            source="pre_workload_definition_analysis",
        )
        if returncode == 0:
            organize_definition_directory(definitions_dir)
            definition_report = check_definition_directory(
                definitions_dir, sglang_inventory=inventory, model_name=str(config["model_name"])
            )
    write_definition_review(run_path, definition_report)
    if not definition_report["summary"]["ok"]:
        raise SystemExit(
            "ERROR: definitions failed review; fix reports/definition_review.md before dump-workload"
        )
    if definition_report["summary"]["collectable"] == 0:
        raise SystemExit("ERROR: no reviewed definition contains a supported capture tag")

    artifacts = load_definition_artifacts(definitions_dir)
    plan = plan_for_stage(
        stage="workloads", run_path=run_path, config=config, reviewed_definitions=artifacts
    )
    request_manifest = load_request_manifest(run_path)
    if request_manifest is None:
        raise SystemExit(
            "ERROR: reports/evidence/request_manifest.json is required; "
            "run dump-definition before dump-workload"
        )
    plan["request_manifest"] = request_manifest
    if isinstance(inventory_modules, list):
        plan["sglang_execution_inventory"] = {
            "modules": [
                {
                    "class_path": item["class_path"],
                    "module_paths": item.get("module_paths", []),
                }
                for item in inventory_modules
                if isinstance(item, dict) and isinstance(item.get("class_path"), str)
            ]
        }
    result = run_stage(plan, run_path, config)
    workload_report = result.get("workload_report")
    if not isinstance(workload_report, dict):
        raise SystemExit("ERROR: workload stage returned no workload report")
    request_provenance = result.get("request_provenance")
    if (
        not isinstance(request_provenance, dict)
        or request_provenance.get("manifest_sha256")
        != request_manifest.get("manifest_sha256")
    ):
        raise SystemExit(
            "ERROR: workload stage did not use the reviewed request manifest"
        )
    current_digest = definitions_digest(definitions_dir)
    if (
        result.get("definitions_sha256") != plan["definitions_sha256"]
        or current_digest != plan["definitions_sha256"]
    ):
        raise SystemExit(
            "ERROR: definitions changed while dump-workload was running; rerun with the reviewed snapshot"
        )

    names = [
        str(item["name"])
        for item in workload_report.get("definitions", [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and int(item.get("workloads") or 0) > 0
    ]
    from flashinfer_bench.data.validate import validate_dataset

    validation_report = validate_dataset(
        dataset=str(run_path / "output"),
        checks=["layout", "definition", "workload"],
        outputs=["stdout"],
        disable_gpu=True,
        definitions=names,
    )
    validation_errors = sorted(
        name for name, report in validation_report.definitions.items() if report.status == "error"
    )
    dataset_validation = {
        "ok": not validation_errors,
        "error_definitions": validation_errors,
        "report": validation_report.model_dump(mode="json"),
    }
    accepted = bool(workload_report["summary"]["ok"] and dataset_validation["ok"])
    report = {
        "schema_version": 1,
        "model_name": config["model_name"],
        "definitions_sha256": current_digest,
        "workload": workload_report,
        "dataset_validation": dataset_validation,
        "request_provenance": request_provenance,
        "accepted": accepted,
    }
    reports_dir = run_path / "reports"
    capture_metadata = result.get("capture_metadata")
    if isinstance(capture_metadata, list):
        write_json(
            reports_dir / "evidence" / "capture_metadata.json",
            {"schema_version": 1, "definitions": capture_metadata},
        )
    analysis = None
    if not accepted and args.agent:
        returncode = run_definition_analysis(
            run_dir=run_path,
            model_name=str(config["model_name"]),
            report=report,
            source="workload_result_analysis",
        )
        analyzed_digest = definitions_digest(definitions_dir)
        refreshed = check_definition_directory(
            definitions_dir, sglang_inventory=inventory, model_name=str(config["model_name"])
        )
        write_definition_review(run_path, refreshed)
        analysis = {
            "returncode": returncode,
            "definitions_changed": analyzed_digest != current_digest,
            "definitions_sha256": analyzed_digest,
            "requires_review_and_rerun": analyzed_digest != current_digest,
        }
        report["agent_analysis"] = analysis
        report["current_definitions_sha256"] = analyzed_digest
    else:
        report["current_definitions_sha256"] = current_digest
    write_json(reports_dir / "run_report.json", report)
    (reports_dir / "review.md").write_text(
        render_workload_review(workload_report, dataset_validation, analysis), encoding="utf-8"
    )
    print(f"workloads: {workload_report['summary']['workloads']}")
    print(f"run report: {reports_dir / 'run_report.json'}")
    print(f"review: {reports_dir / 'review.md'}")
    print(f"run accepted: {accepted}")
    if analysis and analysis["requires_review_and_rerun"]:
        print("definitions changed: review definitions/ and rerun dump-workload")
    return 0 if accepted else 1
