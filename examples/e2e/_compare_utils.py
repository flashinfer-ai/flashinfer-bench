"""Shared helpers for E2E compare launchers."""

from __future__ import annotations

import csv
import json
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "run"


def render_template(value: str, mapping: Mapping[str, str]) -> str:
    return value.format_map(_SafeFormatDict(mapping))


def render_templates(values: Iterable[str], mapping: Mapping[str, str]) -> List[str]:
    return [render_template(value, mapping) for value in values]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(
    command: List[str],
    *,
    cwd: Optional[Path],
    env: Optional[Dict[str, str]],
    stdout_path: Path,
    stderr_path: Path,
) -> Dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
            )
    duration_s = time.perf_counter() - started
    return {
        "exit_code": completed.returncode,
        "duration_s": duration_s,
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def launch_command(
    command: List[str],
    *,
    cwd: Optional[Path],
    env: Optional[Dict[str, str]],
    stdout_path: Path,
    stderr_path: Path,
) -> Dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
    )
    return {
        "process": process,
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
        "started_at": time.perf_counter(),
    }


def stop_process(
    state: Dict[str, Any],
    *,
    interrupt_timeout_s: float = 5.0,
    terminate_timeout_s: float = 20.0,
) -> Dict[str, Any]:
    process = state["process"]
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=interrupt_timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=terminate_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=terminate_timeout_s)

    state["stdout_file"].close()
    state["stderr_file"].close()
    duration_s = time.perf_counter() - state["started_at"]
    return {
        "exit_code": process.returncode,
        "duration_s": duration_s,
        "status": "passed" if process.returncode == 0 else "failed",
    }


def wait_for_http_ready(
    url: str,
    *,
    timeout_s: float = 300.0,
    interval_s: float = 2.0,
    required_substring: str = "",
    process: Optional[subprocess.Popen[str]] = None,
) -> Dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    attempts = 0
    last_error = ""
    while time.monotonic() < deadline:
        attempts += 1
        if process is not None and process.poll() is not None:
            return {
                "ok": False,
                "attempts": attempts,
                "error": f"server process exited before readiness with code {process.returncode}",
            }
        try:
            with opener.open(url, timeout=min(10.0, interval_s + 1.0)) as response:
                body = response.read().decode("utf-8", errors="replace")
                if required_substring and required_substring not in body:
                    last_error = (
                        f"HTTP {response.status} received from {url}, but substring "
                        f"{required_substring!r} was not present"
                    )
                else:
                    return {
                        "ok": True,
                        "attempts": attempts,
                        "http_status": getattr(response, "status", 200),
                    }
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(interval_s)

    return {
        "ok": False,
        "attempts": attempts,
        "error": last_error or f"Timed out waiting for {url}",
    }


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_artifact(path: Path, fmt: str = "auto") -> Optional[Any]:
    if not path.exists():
        return None

    format_name = fmt
    if format_name == "auto":
        format_name = "jsonl" if path.suffix.lower() == ".jsonl" else "json"

    if format_name == "json":
        return load_json_file(path)

    if format_name == "jsonl":
        records = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if len(records) == 1:
            return records[0]
        return {"jsonl_record_count": len(records)}

    raise ValueError(f"Unsupported artifact format: {fmt}")


def flatten_scalars(payload: Any, *, prefix: str = "", max_depth: int = 2) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    def _walk(value: Any, path: str, depth: int) -> None:
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[path] = value
            return

        if isinstance(value, list):
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                flattened[path] = ",".join("" if item is None else str(item) for item in value)
            return

        if isinstance(value, dict) and depth < max_depth:
            for key, child in value.items():
                child_path = f"{path}_{key}" if path else str(key)
                _walk(child, child_path, depth + 1)

    _walk(payload, prefix.strip("_"), 0)
    return flattened


def summarize_named_counts(items: Iterable[Mapping[str, Any]], *, key_name: str) -> str:
    parts = []
    for item in items:
        key = str(item.get(key_name, "")).strip()
        if not key:
            continue
        calls = item.get("calls")
        if calls is None:
            parts.append(key)
        else:
            parts.append(f"{key}:{calls}")
    return ";".join(parts)
