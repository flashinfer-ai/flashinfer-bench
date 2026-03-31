"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import statistics
import subprocess
import sys
import tempfile
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import Definition, Solution, Workload

logger = logging.getLogger(__name__)

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()

# NCU metrics to collect for profiling
NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "launch__shared_mem_per_block",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
]


def _device_lock(device: str) -> LockType:
    """Get or create a multiprocessing lock for the specified device.

    This function maintains a registry of locks per device to ensure that
    benchmarking operations on the same device are serialized, preventing
    interference between concurrent measurements.

    Parameters
    ----------
    device : str
        The device identifier (e.g., "cuda:0", "cuda:1").

    Returns
    -------
    LockType
        A lock object specific to the given device.
    """
    with _registry_lock:
        lock = _device_locks.get(device)
        if lock is None:
            lock = Lock()
            _device_locks[device] = lock
        return lock


def time_runnable(fn: Runnable, args: List[Any], warmup: int, iters: int, device: str) -> float:
    """Time the execution of a value-returning style Runnable kernel.

    Uses CUPTI activity tracing for precise hardware-level kernel timing,
    with automatic fallback to CUDA events if CUPTI is unavailable.

    Parameters
    ----------
    fn : Runnable
        The kernel function to benchmark (must be value-returning style).
    args : List[Any]
        List of arguments in definition order.
    warmup : int
        Number of warmup iterations before timing.
    iters : int
        Number of timing iterations to average over.
    device : str
        The CUDA device to run the benchmark on.

    Returns
    -------
    float
        The median execution time in milliseconds.
    """
    lock = _device_lock(device)
    with lock:
        with torch.cuda.device(device):
            times = bench_gpu_time_with_cupti(
                fn=fn,
                dry_run_iters=warmup,
                repeat_iters=iters,
                input_args=tuple(args),
                cold_l2_cache=True,
                use_cuda_graph=False,
            )
            return statistics.median(times)


def _parse_dim_tuple(s: str) -> List[int]:
    """Parse an NCU dimension string like '(256, 1, 1)' into [256, 1, 1]."""
    match = re.findall(r"\d+", s)
    return [int(x) for x in match] if match else [0, 0, 0]


def _safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    """Safely extract a float from a CSV row dict."""
    val = row.get(key, "")
    if not val or val == "":
        return default
    try:
        # NCU may format with commas for thousands
        return float(val.replace(",", ""))
    except (ValueError, TypeError):
        return default


def _safe_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    """Safely extract an int from a CSV row dict."""
    val = row.get(key, "")
    if not val or val == "":
        return default
    try:
        return int(float(val.replace(",", "")))
    except (ValueError, TypeError):
        return default


def _parse_ncu_csv(csv_output: str) -> List[Dict[str, Any]]:
    """Parse NCU --csv --page raw output into kernel profile dicts.

    Parameters
    ----------
    csv_output : str
        Raw stdout from ncu --csv --page raw.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts, each suitable for KernelProfile(**d).
    """
    # Filter out NCU preamble/warning lines (start with ==)
    lines = csv_output.split("\n")
    csv_lines = [line for line in lines if line and not line.startswith("==")]

    if len(csv_lines) < 2:
        return []

    # First non-preamble line is header, second is units, rest are data
    # With --page raw, the format is: header, units, data rows
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))

    kernels: List[Dict[str, Any]] = []
    for i, row in enumerate(reader):
        if i == 0:
            # First row after header is the units row — skip it
            # Check if this looks like a units row (contains "ns", "byte", "%", etc.)
            id_val = row.get("ID", "")
            if id_val == "" or not id_val.strip().isdigit():
                continue

        # Skip unit rows that slipped through
        id_val = row.get("ID", "")
        if not id_val or not id_val.strip().isdigit():
            continue

        kernel_name = row.get("Kernel Name", "unknown")
        grid = _parse_dim_tuple(row.get("Grid Size", "(0,0,0)"))
        block = _parse_dim_tuple(row.get("Block Size", "(0,0,0)"))

        kernels.append(
            {
                "name": kernel_name,
                "gpu__time_duration.sum": _safe_float(
                    row, "gpu__time_duration.sum"
                ),
                "grid": grid,
                "block": block,
                "launch__registers_per_thread": _safe_int(
                    row, "launch__registers_per_thread"
                ),
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": _safe_float(
                    row, "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                ),
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": _safe_float(
                    row,
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                ),
                "dram__bytes_read.sum": _safe_float(
                    row, "dram__bytes_read.sum"
                ),
                "dram__bytes_write.sum": _safe_float(
                    row, "dram__bytes_write.sum"
                ),
                "l1tex__t_sector_hit_rate.pct": _safe_float(
                    row, "l1tex__t_sector_hit_rate.pct"
                ),
                "lts__t_sector_hit_rate.pct": _safe_float(
                    row, "lts__t_sector_hit_rate.pct"
                ),
                "launch__shared_mem_per_block": _safe_float(
                    row, "launch__shared_mem_per_block"
                ),
                "sm__warps_active.avg.pct_of_peak_sustained_active": _safe_float(
                    row,
                    "sm__warps_active.avg.pct_of_peak_sustained_active",
                ),
                "sm__maximum_warps_per_active_cycle_pct": _safe_float(
                    row, "sm__maximum_warps_per_active_cycle_pct"
                ),
            }
        )

    return kernels


def profile_runnable(
    definition: Definition,
    solution: Solution,
    workload: Workload,
    device: str,
    trace_set_root: Optional[Path] = None,
    ncu_path: str = "ncu",
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """Profile kernel execution with NCU (Nsight Compute).

    Runs the solution in a subprocess under NCU to collect per-kernel
    hardware profiling data (SM throughput, cache hit rates, DRAM bandwidth,
    occupancy, etc.).

    Parameters
    ----------
    definition : Definition
        The kernel definition.
    solution : Solution
        The solution to profile.
    workload : Workload
        The workload to run.
    device : str
        The CUDA device to run on.
    trace_set_root : Path, optional
        Root path of the trace set (for safetensors loading).
    ncu_path : str
        Path to the NCU executable.
    timeout : int
        Timeout in seconds for NCU profiling.

    Returns
    -------
    List[Dict[str, Any]]
        List of kernel profile dicts, each suitable for KernelProfile(**d).

    Raises
    ------
    RuntimeError
        If NCU fails or times out.
    """
    with tempfile.TemporaryDirectory(prefix="fib_ncu_profile_") as build_dir:
        build_path = Path(build_dir)

        # Write data files for _solution_runner
        (build_path / "definition.json").write_text(definition.model_dump_json())
        (build_path / "solution.json").write_text(solution.model_dump_json())
        (build_path / "workload.json").write_text(workload.model_dump_json())

        # Build NCU command
        metrics_str = ",".join(NCU_METRICS)
        cmd = [
            ncu_path,
            "--csv",
            "--page", "raw",
            "--metrics", metrics_str,
            "--replay-mode", "application",
            "--nvtx",
            "--nvtx-include", "flashinfer_bench_ncu_profile]",
            "-f",
            "--",
            sys.executable,
            "-u",
            "-m", "flashinfer_bench.agents._solution_runner",
            "--data-dir", str(build_path),
            "--device", device,
        ]
        if trace_set_root is not None:
            cmd.extend(["--trace-set-path", str(trace_set_root)])

        # Remap the target device to cuda:0 via CUDA_VISIBLE_DEVICES so NCU
        # profiles on device 0.  NCU + TMA (cp.async.bulk.tensor) can fail on
        # non-zero devices during replay.
        env = os.environ.copy()
        device_idx = 0
        if device.startswith("cuda:"):
            try:
                device_idx = int(device.split(":")[1])
            except (IndexError, ValueError):
                pass
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        # The subprocess now sees only one GPU, so tell the runner to use cuda:0
        cmd[cmd.index("--device") + 1] = "cuda:0"

        logger.info("NCU profile command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"NCU profiling timed out after {timeout} seconds")

        if result.returncode != 0:
            stderr_snippet = result.stderr[:500] if result.stderr else ""
            stdout_snippet = result.stdout[:500] if result.stdout else ""
            raise RuntimeError(
                f"NCU exited with code {result.returncode}.\n"
                f"stderr: {stderr_snippet}\nstdout: {stdout_snippet}"
            )

        kernels = _parse_ncu_csv(result.stdout)
        if not kernels:
            logger.warning("NCU produced no kernel data. stderr: %s", result.stderr[:300])

        return kernels
