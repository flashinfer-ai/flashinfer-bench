"""NCU profiling tool for LLM agents."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from flashinfer_bench.data import Solution, TraceSet, Workload
from flashinfer_bench.env import get_fib_ncu_path

# Valid NCU sets
VALID_SETS = {"basic", "detailed", "full", "nvlink", "pmsampling", "roofline"}

# Valid NCU pages
VALID_PAGES = {"raw", "details", "source"}

# Valid NCU sections
VALID_SECTIONS = {
    "C2CLink",
    "ComputeWorkloadAnalysis",
    "InstructionStats",
    "LaunchStats",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Chart",
    "MemoryWorkloadAnalysis_Tables",
    "NumaAffinity",
    "Nvlink",
    "Nvlink_Tables",
    "Nvlink_Topology",
    "Occupancy",
    "PmSampling",
    "PmSampling_WarpStates",
    "SchedulerStats",
    "SourceCounters",
    "SpeedOfLight",
    "SpeedOfLight_HierarchicalDoubleRooflineChart",
    "SpeedOfLight_HierarchicalHalfRooflineChart",
    "SpeedOfLight_HierarchicalSingleRooflineChart",
    "SpeedOfLight_HierarchicalTensorRooflineChart",
    "SpeedOfLight_RooflineChart",
    "Tile",
    "WarpStateStats",
    "WorkloadDistribution",
}


def _build_ncu_command(
    data_path: Path,
    set: str,
    sections: Optional[List[str]],
    kernel_name: Optional[str],
    report_path: Path,
    page: str,
) -> List[str]:
    """Build the NCU command line."""
    ncu_path = get_fib_ncu_path()

    cmd = [ncu_path, "--export", str(report_path), "--page", page, "--set", set]

    # Add extra sections
    if sections:
        for section in sections:
            cmd.extend(["--section", section])

    # Kernel filter
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])

    # Force overwrite
    cmd.append("-f")

    # Run the runner module with data file
    cmd.extend(
        ["python", "-u", "-m", "flashinfer_bench.agents._ncu_runner", "--data", str(data_path)]
    )

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


def run_ncu(
    solution: Solution,
    workload: Workload,
    *,
    page: str = "details",
    set: str = "detailed",
    sections: Optional[List[str]] = None,
    kernel_name: Optional[str] = None,
    device: str = "cuda:0",
    timeout: int = 60,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Run NCU profiling on a solution with a specific workload.

    This function analyzes the performance of a solution using NVIDIA Nsight Compute.

    All inputs and outputs are JSON-serializable, making it suitable as a target
    for LLM agent tool calling.

    Environment Variables
    ---------------------
    FIB_DATASET_PATH : str
        Path to the trace dataset. The solution's definition must exist in this dataset.
    FIB_NCU_PATH : str, optional
        Path to the NCU executable. Defaults to "ncu".

    Parameters
    ----------
    solution : Solution
        The solution to profile.
    workload : Workload
        The workload configuration specifying input dimensions and data.
    page : str, optional
        NCU output page format. One of: "raw", "details", "source". Default is "details".
    set : str, optional
        NCU section set to collect. One of: "basic", "detailed", "full",
        "nvlink", "pmsampling", "roofline". Default is "detailed".
    sections : List[str], optional
        Additional sections to collect beyond the set. See NCU documentation
        for available sections.
    kernel_name : str, optional
        Filter to profile only kernels matching this name (supports regex).
    device : str, optional
        CUDA device to run on. Default is "cuda:0".
    timeout : int, optional
        Timeout in seconds for NCU profiling. Default is 60.
    tmpdir : str, optional
        Temporary directory for NCU. If not provided, uses system default.
    max_lines : int, optional
        Maximum number of lines in output. If None, returns full output.

    Returns
    -------
    str
        NCU profiling results as text.

    Raises
    ------
    ValueError
        If the solution's definition is not found in the trace database,
        or if invalid set/sections are specified.
    RuntimeError
        If NCU profiling fails.

    Examples
    --------
    >>> from flashinfer_bench.agents import run_ncu
    >>> result = run_ncu(solution, workload, set="detailed")
    >>> print(result)
    """
    # Validate set
    if set not in VALID_SETS:
        raise ValueError(f"Invalid set '{set}'. Must be one of: {VALID_SETS}")

    # Validate page
    if page not in VALID_PAGES:
        raise ValueError(f"Invalid page '{page}'. Must be one of: {VALID_PAGES}")

    # Validate sections
    if sections:
        invalid = [s for s in sections if s not in VALID_SECTIONS]
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}. Valid sections: {VALID_SECTIONS}")

    # Get trace set and definition
    trace_set = TraceSet.from_path()
    if solution.definition not in trace_set.definitions:
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace database. "
            f"Available definitions: {list(trace_set.definitions.keys())}"
        )
    definition = trace_set.definitions[solution.definition]

    # Create temporary directory for build artifacts
    with tempfile.TemporaryDirectory(prefix="fib_ncu_", dir=tmpdir) as build_dir:
        build_path = Path(build_dir)

        # Write data file for the runner
        data_path = build_path / "data.json"
        data = {
            "definition": definition.model_dump(mode="json"),
            "solution": solution.model_dump(mode="json"),
            "workload": workload.model_dump(mode="json"),
            "trace_set_root": str(trace_set.root) if trace_set.root else None,
            "device": device,
        }
        data_path.write_text(json.dumps(data))

        # Build NCU command
        report_path = build_path / "report"
        cmd = _build_ncu_command(data_path, set, sections, kernel_name, report_path, page)

        # Set up environment
        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        # Run NCU
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: NCU profiling timed out after {timeout} seconds."
        except FileNotFoundError:
            return (
                f"ERROR: NCU executable not found at '{get_fib_ncu_path()}'. "
                "Please install NVIDIA Nsight Compute."
            )

        # Combine stdout and stderr
        output = result.stdout + result.stderr

        # Check for errors
        if result.returncode != 0:
            # On error, return the full output without truncation to aid debugging.
            return output

        # Truncate if requested
        if max_lines:
            output = _truncate_output(output, max_lines)

        return output
