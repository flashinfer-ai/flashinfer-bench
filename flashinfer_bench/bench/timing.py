"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

import bisect
import statistics
from functools import partial
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Dict, List, Tuple

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.testing.utils import get_l2_cache_size

from flashinfer_bench.compile import Runnable

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()


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


def profile_runnable(
    fn: Runnable, args: List[Any], warmup: int, iters: int, device: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """Profile kernel execution with CUPTI Activity API.

    Collects per-kernel hardware profiling data (launch config, register usage,
    shared memory) alongside timing measurements.

    Parameters
    ----------
    fn : Runnable
        The kernel function to profile.
    args : List[Any]
        List of arguments in definition order.
    warmup : int
        Number of warmup iterations before profiling.
    iters : int
        Number of profiling iterations.
    device : str
        The CUDA device to run on.

    Returns
    -------
    Tuple[float, List[Dict[str, Any]]]
        (median_latency_ms, kernel_profiles) where kernel_profiles is a list
        of dicts with CUPTI ActivityKernel11 fields from the median iteration.
    """
    from cupti import cupti

    def _buffer_requested():
        return 8 * 1024 * 1024, 0

    def _collect_kernel_info(activity):
        return {
            "name": activity.name,
            "start": activity.start,
            "end": activity.end,
            "correlation_id": activity.correlation_id,
            "grid": [activity.grid_x, activity.grid_y, activity.grid_z],
            "block": [activity.block_x, activity.block_y, activity.block_z],
            "registers_per_thread": int(activity.registers_per_thread),
            "static_shared_memory": int(activity.static_shared_memory),
            "dynamic_shared_memory": int(activity.dynamic_shared_memory),
            "shared_memory_executed": int(activity.shared_memory_executed),
            "local_memory_per_thread": int(activity.local_memory_per_thread),
            "local_memory_total": int(activity.local_memory_total),
        }

    def _buffer_completed(launches, kernels, activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels.append(_collect_kernel_info(activity))
            elif activity.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append(
                    (activity.start, activity.end, activity.correlation_id)
                )

    def call_fn():
        fn(*args)

    lock = _device_lock(device)
    with lock:
        with torch.cuda.device(device):
            # L2 cache flush buffer
            l2_size = get_l2_cache_size(device)
            l2_flush_size = (l2_size * 2) // (1024 * 1024) * 1024 * 1024
            buffer = torch.empty(l2_flush_size, device=device, dtype=torch.int8)

            # Warmup
            torch.cuda.synchronize()
            for _ in range(warmup):
                buffer.zero_()
                call_fn()
            torch.cuda.synchronize()

            # CUPTI measurement
            launches: List[Tuple] = []
            kernels: List[Dict] = []
            iter_timestamps = []

            cupti.activity_enable(cupti.ActivityKind.RUNTIME)
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_enable(cupti.ActivityKind.DRIVER)
            cupti.activity_register_callbacks(
                _buffer_requested, partial(_buffer_completed, launches, kernels)
            )

            for _ in range(iters):
                buffer.zero_()
                start_cpu = cupti.get_timestamp()
                call_fn()
                end_cpu = cupti.get_timestamp()
                torch.cuda.synchronize()
                iter_timestamps.append((start_cpu, end_cpu))

            cupti.activity_flush_all(0)
            cupti.activity_disable(cupti.ActivityKind.RUNTIME)
            cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_disable(cupti.ActivityKind.DRIVER)
            cupti.finalize()

            del buffer

            # Correlate launches to kernels per iteration
            sorted_launches = sorted(launches, key=lambda l: l[0])
            launch_starts = [l[0] for l in sorted_launches]

            corr_id_to_kernels: Dict[int, List[Dict]] = {}
            for k in kernels:
                cid = k["correlation_id"]
                if cid not in corr_id_to_kernels:
                    corr_id_to_kernels[cid] = []
                corr_id_to_kernels[cid].append(k)

            measured_times = []
            iter_kernel_data: List[List[Dict]] = []

            for start_cpu, end_cpu in iter_timestamps:
                left = bisect.bisect_left(launch_starts, start_cpu)
                right = bisect.bisect_right(launch_starts, end_cpu)
                corr_ids = set(sorted_launches[i][2] for i in range(left, right))

                iter_kernels = []
                for cid in corr_ids:
                    if cid in corr_id_to_kernels:
                        iter_kernels.extend(corr_id_to_kernels[cid])

                if not iter_kernels:
                    measured_times.append(0.0)
                    iter_kernel_data.append([])
                    continue

                min_start = min(k["start"] for k in iter_kernels)
                max_end = max(k["end"] for k in iter_kernels)
                span_ms = (max_end - min_start) / 1e6
                measured_times.append(span_ms)
                iter_kernel_data.append(iter_kernels)

            # Find median iteration
            median_ms = statistics.median(measured_times)
            median_idx = min(
                range(len(measured_times)),
                key=lambda i: abs(measured_times[i] - median_ms),
            )

            # Build profile dicts from median iteration kernels
            profile_kernels = []
            for k in iter_kernel_data[median_idx]:
                profile_kernels.append(
                    {
                        "name": k["name"],
                        "duration_ns": int(k["end"] - k["start"]),
                        "grid": k["grid"],
                        "block": k["block"],
                        "registers_per_thread": k["registers_per_thread"],
                        "static_shared_memory": k["static_shared_memory"],
                        "dynamic_shared_memory": k["dynamic_shared_memory"],
                        "shared_memory_executed": k["shared_memory_executed"],
                        "local_memory_per_thread": k["local_memory_per_thread"],
                        "local_memory_total": k["local_memory_total"],
                    }
                )

            return median_ms, profile_kernels
