"""Compute-sanitizer tool"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Literal, Optional, Union

from flashinfer_bench.data import Definition, Solution, TraceSet, Workload

logger = logging.getLogger(__name__)

SanitizerType = Literal["memcheck", "racecheck", "initcheck", "synccheck"]
VALID_SANITIZER_TYPES: set[SanitizerType] = {"memcheck", "racecheck", "initcheck", "synccheck"}


def _build_sanitizer_command(
    sanitizer_type: SanitizerType,
    data_dir: Path,
    device: str,
    trace_set_path: Optional[Path],
    sanitizer_path: str,
) -> List[str]:
    cmd = [sanitizer_path, "--tool", sanitizer_type]

    runner_cmd = [
        sys.executable,
        "-u",
        "-m",
        "flashinfer_bench.agents._solution_runner",
        "--data-dir",
        str(data_dir),
        "--device",
        device,
    ]
    if trace_set_path:
        runner_cmd.extend(["--trace-set-path", str(trace_set_path)])

    cmd.extend(runner_cmd)
    return cmd


def _truncate_output(output: str, max_lines: int) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output

    truncated = lines[:max_lines]
    remaining = len(lines) - max_lines
    truncated.append(f"\n[Output truncated: {remaining} more lines, use max_lines=None to see all]")
    return "\n".join(truncated)


def flashinfer_bench_run_sanitizer(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    # Runtime environment
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    # Sanitizer configuration
    sanitizer_types: Optional[List[SanitizerType]] = None,
    sanitizer_path: str = "compute-sanitizer",
    # Execution control
    timeout: int = 300,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Run compute-sanitizer checks on a solution with a specific workload:
    memcheck, racecheck, initcheck, synccheck.

    Parameters
    ----------
    solution : Solution or str
        The solution to check. Can be a Solution object or a path to a JSON file.
    workload : Workload or str
        The workload configuration specifying input dimensions and data. Can be a
        Workload object or a path to a JSON file.
    device : str, optional
        CUDA device to run on. Default is "cuda:0".
    trace_set_path : str, optional
        Path to the trace set. If not provided, uses FIB_DATASET_PATH environment variable.
    sanitizer_types : List[SanitizerType], optional
        List of sanitizer tools to run. Default runs all: memcheck, racecheck,
        initcheck, synccheck.
    sanitizer_path : str, optional
        Path to the compute-sanitizer executable. Default is "compute-sanitizer".
    timeout : int, optional
        Timeout in seconds for each sanitizer check. Default is 300.
    tmpdir : str, optional
        Temporary directory. If not provided, uses system default.
    max_lines : int, optional
        Maximum number of lines in output. If None, returns full output.

    Returns
    -------
    str
        Sanitizer results as text, or error message starting with "ERROR:".
    """
    if sanitizer_types is None:
        sanitizer_types = list(VALID_SANITIZER_TYPES)

    for st in sanitizer_types:
        if st not in VALID_SANITIZER_TYPES:
            return f"ERROR: Invalid sanitizer type '{st}'. Must be one of: {VALID_SANITIZER_TYPES}"

    if isinstance(solution, str):
        path = Path(solution)
        if not path.exists():
            return f"ERROR: Solution file not found: {solution}"
        try:
            solution = Solution.model_validate_json(path.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse solution file: {e}"

    if isinstance(workload, str):
        path = Path(workload)
        if not path.exists():
            return f"ERROR: Workload file not found: {workload}"
        try:
            workload = Workload.model_validate_json(path.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse workload file: {e}"

    try:
        trace_set = TraceSet.from_path(trace_set_path)
    except Exception as e:
        return f"ERROR: Failed to load trace set: {e}"

    if solution.definition not in trace_set.definitions:
        return (
            f"ERROR: Definition '{solution.definition}' not found in trace database. "
            f"Available definitions: {list(trace_set.definitions.keys())}"
        )
    definition = trace_set.definitions[solution.definition]

    if shutil.which(sanitizer_path) is None:
        return (
            f"ERROR: compute-sanitizer executable not found at '{sanitizer_path}'. "
            "Please install NVIDIA CUDA toolkit."
        )

    with tempfile.TemporaryDirectory(prefix="fib_sanitizer_", dir=tmpdir) as build_dir:
        build_path = Path(build_dir)

        (build_path / "definition.json").write_text(definition.model_dump_json())
        (build_path / "solution.json").write_text(solution.model_dump_json())
        (build_path / "workload.json").write_text(workload.model_dump_json())

        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        output = ""

        for sanitizer_type in sanitizer_types:
            output += f"\n{'=' * 60}\n"
            output += f"Running {sanitizer_type.upper()}\n"
            output += f"{'=' * 60}\n\n"

            cmd = _build_sanitizer_command(
                sanitizer_type,
                build_path,
                device,
                Path(trace_set_path) if trace_set_path else None,
                sanitizer_path,
            )

            logger.info("FlashInfer Bench Run Sanitizer: Running Command: %s", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, timeout=timeout
                )

                output += f"STDOUT:\n{result.stdout}\n\n"

                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n\n"

                output += f"Return code: {result.returncode}\n"

                if result.returncode != 0:
                    output += f"\nWARNING: {sanitizer_type} detected issues!\n"
                else:
                    output += f"\n{sanitizer_type} passed successfully.\n"

            except subprocess.TimeoutExpired:
                output += f"ERROR: {sanitizer_type} timed out after {timeout} seconds.\n"
            except Exception as e:
                output += f"ERROR: Failed to run {sanitizer_type}: {e}\n"

        output += f"\n{'=' * 60}\n"
        output += "Sanitizer checks complete\n"
        output += f"{'=' * 60}\n"

        if max_lines:
            output = _truncate_output(output, max_lines)

        return output
