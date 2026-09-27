"""Execute one onboarding stage in the current GPU environment."""

from __future__ import annotations

import base64
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.runners.remote_pipeline import run_remote_stage


def run_stage(
    plan: dict[str, Any],
    run_path: Path,
    _config: dict[str, Any],
) -> dict[str, Any]:
    """Run a stage locally and materialize its returned dataset archive."""
    run_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onboarding-stage-", dir=run_path) as work_dir:
        result = run_remote_stage(plan, remote_output_dir=work_dir)
        materialize_stage_result(result, run_path)
    return result


def materialize_stage_result(result: dict[str, Any], run_path: Path) -> None:
    """Extract one stage result into its stable run directory."""
    stage = result.get("stage")
    if stage == "definitions":
        root = run_path / "definitions"
        archive_b64 = result.get("definitions_archive_b64")
        expected_root = "definitions"
    elif stage == "workloads":
        root = run_path / "output"
        archive_b64 = result.get("output_archive_b64")
        expected_root = "output"
    else:
        raise ValueError(f"unknown onboarding stage result: {stage!r}")
    _extract_result_archive(
        root=root,
        archive_b64=archive_b64,
        expected_root=expected_root,
    )


def _extract_result_archive(*, root: Path, archive_b64: Any, expected_root: str) -> None:
    if not isinstance(archive_b64, str) or not archive_b64:
        raise ValueError(f"{expected_root} stage returned no archive")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    archive_path = parent / f".{expected_root}.tar.gz"
    archive_path.write_bytes(base64.b64decode(archive_b64.encode("ascii")))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            parent_resolved = parent.resolve()
            for member in archive.getmembers():
                if not (parent / member.name).resolve().is_relative_to(parent_resolved):
                    raise ValueError(f"unsafe archive member: {member.name}")
            try:
                archive.extractall(parent, filter="data")
            except TypeError:
                archive.extractall(parent)
    finally:
        archive_path.unlink(missing_ok=True)
