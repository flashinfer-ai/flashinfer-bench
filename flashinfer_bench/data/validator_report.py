"""Pydantic models for dataset validation report, text rendering, and file loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class CheckMessage(BaseModel):
    level: Literal["info", "warning", "error"]
    message: str


class CheckResult(BaseModel):
    status: Literal["ok", "warning", "error"]
    messages: list[CheckMessage] = []


class DefinitionReport(BaseModel):
    status: Literal["ok", "warning", "error"] = "ok"
    layout: Optional[CheckResult] = None
    definition: Optional[CheckResult] = None
    workload: Optional[CheckResult] = None
    solution: Optional[CheckResult] = None
    trace: Optional[CheckResult] = None
    baseline: Optional[CheckResult] = None
    benchmark: Optional[CheckResult] = None


class ReportConfig(BaseModel):
    dataset: str
    checks: list[str]
    disable_gpu: bool
    op_types: list[str]
    definitions: list[str]


class DatasetReport(BaseModel):
    config: ReportConfig
    definitions: dict[str, DefinitionReport]


CHECK_FIELDS = ["layout", "definition", "workload", "solution", "trace", "baseline", "benchmark"]


def compute_definition_status(report: DefinitionReport) -> Literal["ok", "warning", "error"]:
    """Derive overall status from the worst individual check result.

    Parameters
    ----------
    report : DefinitionReport
        Report for a single definition.

    Returns
    -------
    Literal["ok", "warning", "error"]
        The worst status across all non-None check results.
    """
    worst = "ok"
    for field_name in CHECK_FIELDS:
        result = getattr(report, field_name)
        if result is None:
            continue
        if result.status == "error":
            return "error"
        if result.status == "warning":
            worst = "warning"
    return worst


def render_text_report(report: DatasetReport) -> str:
    """Render a DatasetReport into human-readable text.

    Parameters
    ----------
    report : DatasetReport
        The validation report to render.

    Returns
    -------
    str
        Multi-line text representation of the report.
    """
    lines: list[str] = []
    lines.append("=== Dataset Validation Report ===")
    lines.append(f"dataset:       {report.config.dataset}")
    lines.append(f"checks:        {', '.join(report.config.checks)}")
    lines.append(f"disable_gpu:   {report.config.disable_gpu}")
    lines.append(f"op_types:      {', '.join(report.config.op_types) or '(all)'}")
    lines.append(f"definitions:   {', '.join(report.config.definitions) or '(all)'}")
    lines.append("")

    counts = {"ok": 0, "warning": 0, "error": 0}
    for def_report in report.definitions.values():
        counts[def_report.status] += 1
    total = sum(counts.values())
    lines.append(
        f"{total} definitions: {counts['ok']} ok, "
        f"{counts['warning']} warning, {counts['error']} error"
    )
    lines.append("")

    for definition_name, def_report in sorted(report.definitions.items()):
        lines.append(f"--- {definition_name} [{def_report.status.upper()}] ---")
        for field_name in CHECK_FIELDS:
            result: Optional[CheckResult] = getattr(def_report, field_name)
            if result is None:
                lines.append(f"  {field_name + ':':12s} None (skipped)")
            else:
                lines.append(f"  {field_name + ':':12s} {result.status}")
                for message in result.messages:
                    lines.append(f"    - {message.message}")
        lines.append("")

    return "\n".join(lines)


def render_report(report_json: str, output: Optional[str] = None) -> str:
    """Load a report JSON file and render it as human-readable text.

    Parameters
    ----------
    report_json : str
        Path to the report JSON file.
    output : Optional[str]
        Path to write the text output to. If None, prints to stdout.

    Returns
    -------
    str
        The rendered text report.
    """
    report_path = Path(report_json)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")

    report = DatasetReport.model_validate_json(report_path.read_text())
    text = render_text_report(report)

    if output:
        Path(output).write_text(text)
    else:
        print(text)

    return text
