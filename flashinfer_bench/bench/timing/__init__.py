"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, List

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from ._common import _device_lock
from .split_timing import SplitTimingMetrics, time_runnable_split_timing

__all__ = ["time_runnable", "time_runnable_split_timing", "SplitTimingMetrics"]


def time_runnable(
    fn: Callable[..., Any], args: List[Any], warmup: int, iters: int, device: str
) -> float:
    """Time the execution of a value-returning style kernel callable.

    Uses CUPTI activity tracing for precise hardware-level kernel timing,
    with automatic fallback to CUDA events if CUPTI is unavailable.

    Parameters
    ----------
    fn : Callable
        The kernel callable to benchmark — a ``Runnable`` or any wrapper with
        the same positional call convention (e.g. the evaluator's
        setup-plus-run closure for setup-hook solutions).
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
