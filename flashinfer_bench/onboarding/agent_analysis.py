"""Optional agent analysis for writing reviewed Definition JSON files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "definition_analysis.md"


def run_definition_analysis(
    *, run_dir: Path, model_name: str, report: dict[str, Any], source: str
) -> int:
    """Ask Codex to analyze evidence and write definitions directly."""
    codex = _find_codex_binary()
    prompt = _analysis_prompt(run_dir=run_dir, model_name=model_name, report=report, source=source)
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join([str(codex.parent), environment.get("PATH", "")])
    completed = subprocess.run(
        [str(codex), "exec", "-C", str(Path.cwd()), "-s", "workspace-write", "--ephemeral", "-"],
        input=prompt,
        text=True,
        env=environment,
        check=False,
    )
    return completed.returncode


def _find_codex_binary() -> Path:
    resolved = shutil.which("codex")
    if resolved:
        return Path(resolved)
    candidates = sorted(Path.home().glob(".vscode-server/extensions/*/bin/linux-x86_64/codex"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("codex binary not found on PATH or under ~/.vscode-server/extensions")


def _analysis_prompt(*, run_dir: Path, model_name: str, report: dict[str, Any], source: str) -> str:
    instructions = _PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    context = "\n".join(
        [
            "## Task Context",
            f"Model: {model_name}",
            f"Run directory: {run_dir}",
            f"Definitions directory: {run_dir / 'definitions'}",
            f"Analysis source: {source}",
            "",
            "## Evidence Paths",
            "- executed SGLang inventory: "
            f"{run_dir / 'reports' / 'evidence' / 'sglang_execution_inventory.json'}",
            "- native FlashInfer definitions already present under definitions/",
            "",
            "## Machine Report",
            "```json",
            json.dumps(report, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return f"{instructions}\n\n{context}\n"
