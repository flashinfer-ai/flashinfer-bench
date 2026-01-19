"""NCU profiling tool for LLM agents."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from flashinfer_bench.data import Solution, TraceSet, Workload

logger = logging.getLogger(__name__)


# Valid NCU pages
VALID_PAGES = {"raw", "details", "source"}


def flashinfer_bench_list_ncu_options(ncu_path: str = "ncu") -> str:
    """List available NCU sets and sections for profiling configuration.

    This function queries the NCU executable to get the available profiling
    sets and sections. Use this to discover valid options for the `set` and
    `sections` parameters of `flashinfer_bench_run_ncu`.

    Parameters
    ----------
    ncu_path : str, optional
        Path to the NCU executable. Default is "ncu".

    Returns
    -------
    str
        Combined output of `ncu --list-sets` and `ncu --list-sections`.

    Examples
    --------
    >>> from flashinfer_bench.agents import flashinfer_bench_list_ncu_options
    >>> print(flashinfer_bench_list_ncu_options())
    """
    if shutil.which(ncu_path) is None:
        return (
            f"ERROR: NCU executable not found at '{ncu_path}'. "
            "Please install NVIDIA Nsight Compute."
        )

    try:
        sets_result = subprocess.run(
            [ncu_path, "--list-sets"], capture_output=True, text=True, timeout=10
        )
        sections_result = subprocess.run(
            [ncu_path, "--list-sections"], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        return "ERROR: NCU command timed out."

    set_result_str = sets_result.stdout + sets_result.stderr
    sections_result_str = sections_result.stdout + sections_result.stderr

    if sets_result.returncode != 0:
        return (
            f"ERROR: ncu --list-sets failed with code {sets_result.returncode}:\n"
            f"{set_result_str}"
        )
    if sections_result.returncode != 0:
        return (
            f"ERROR: ncu --list-sections failed with code {sections_result.returncode}:\n"
            f"{sections_result_str}"
        )

    return f"=== NCU Sets ===\n{set_result_str}\n=== NCU Sections ===\n{sections_result_str}"


def _build_ncu_command(
    data_dir: Path,
    set: str,
    sections: Optional[List[str]],
    kernel_name: Optional[str],
    page: str,
    device: str,
    trace_set_path: Optional[Path],
    ncu_path: str,
) -> List[str]:
    """Build the NCU command line."""

    cmd = [
        ncu_path,
        "--page",
        page,
        "--set",
        set,
        "--nvtx",
        "--nvtx-include",
        "flashinfer_bench_ncu_profile",
    ]

    # Add extra sections
    if sections:
        for section in sections:
            cmd.extend(["--section", section])

    # Kernel filter
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])

    # Force overwrite
    cmd.append("-f")

    # Run the runner module with data directory
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


def flashinfer_bench_run_ncu(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    # Runtime environment
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    # NCU configuration
    set: str = "detailed",
    sections: Optional[List[str]] = None,
    kernel_name: Optional[str] = None,
    page: str = "details",
    ncu_path: str = "ncu",
    # Execution control
    timeout: int = 60,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Run NCU profiling on a solution with a specific workload.

    This function analyzes the performance of a solution using NVIDIA Nsight Compute.

    All inputs and outputs are JSON-serializable, making it suitable as a target
    for LLM agent tool calling.

    Uses FIB_DATASET_PATH environment variable when trace_set_path is not provided.

    Parameters
    ----------
    solution : Solution or str
        The solution to profile. Can be a Solution object or a path to a JSON file.
    workload : Workload or str
        The workload configuration specifying input dimensions and data. Can be a
        Workload object or a path to a JSON file.
    device : str, optional
        CUDA device to run on. Default is "cuda:0".
    trace_set_path : str, optional
        Path to the trace set. If not provided, uses FIB_DATASET_PATH environment variable.
    set : str, optional
        NCU section set to collect. Use `flashinfer_bench_list_ncu_options` to see
        available sets. Default is "detailed".
    sections : List[str], optional
        Additional sections to collect beyond the set. Use `flashinfer_bench_list_ncu_options`
        to see available sections.
    kernel_name : str, optional
        Filter to profile only kernels matching this name (supports regex).
    page : str, optional
        NCU output page format. One of: "raw", "details", "source". Default is "details".
    ncu_path : str, optional
        Path to the NCU executable. Default is "ncu".
    timeout : int, optional
        Timeout in seconds for NCU profiling. Default is 60.
    tmpdir : str, optional
        Temporary directory for NCU. If not provided, uses system default.
    max_lines : int, optional
        Maximum number of lines in output. If None, returns full output.

    Returns
    -------
    str
        NCU profiling results as text, or error message starting with "ERROR:".

    Raises
    ------
    None
        All errors are returned as strings.

    Examples
    --------
    >>> from flashinfer_bench.agents import flashinfer_bench_run_ncu
    >>> result = flashinfer_bench_run_ncu(solution, workload, set="detailed")
    >>> print(result)
    """
    # Parse solution
    if isinstance(solution, str):
        path = Path(solution)
        if not path.exists():
            return f"ERROR: Solution file not found: {solution}"
        try:
            solution = Solution.model_validate_json(path.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse solution file: {e}"

    # Parse workload
    if isinstance(workload, str):
        path = Path(workload)
        if not path.exists():
            return f"ERROR: Workload file not found: {workload}"
        try:
            workload = Workload.model_validate_json(path.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse workload file: {e}"

    # Validate page
    if page not in VALID_PAGES:
        return f"ERROR: Invalid page '{page}'. Must be one of: {VALID_PAGES}"

    # Get trace set and definition
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

    # Check NCU exists
    if shutil.which(ncu_path) is None:
        return (
            f"ERROR: NCU executable not found at '{ncu_path}'. "
            "Please install NVIDIA Nsight Compute."
        )

    # Create temporary directory for build artifacts
    with tempfile.TemporaryDirectory(prefix="fib_ncu_", dir=tmpdir) as build_dir:
        build_path = Path(build_dir)

        # Write data files for the runner
        (build_path / "definition.json").write_text(definition.model_dump_json())
        (build_path / "solution.json").write_text(solution.model_dump_json())
        (build_path / "workload.json").write_text(workload.model_dump_json())

        # Build NCU command
        cmd = _build_ncu_command(
            build_path,
            set,
            sections,
            kernel_name,
            page,
            device,
            Path(trace_set_path) if trace_set_path else None,
            ncu_path,
        )

        # Set up environment
        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        # Run NCU
        logger.info("FlashInfer Bench Run NCU: Running Command: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: NCU profiling timed out after {timeout} seconds."

        # Combine stdout and stderr
        output = result.stdout + result.stderr

        # Check for errors
        if result.returncode != 0:
            return f"ERROR: NCU exited with non-zero return code {result.returncode}:\n{output}"

        # Truncate if requested
        if max_lines:
            output = _truncate_output(output, max_lines)

        return output
