from __future__ import annotations

from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType

import torch
from triton.testing import do_bench

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


def time_runnable(fn: Runnable, inputs: dict, warmup: int, iters: int, device: str) -> float:
    """Time the execution of a Runnable kernel with proper synchronization.

    It will lock the device to prevent concurrent measurements, and use Triton's do_bench for
    accurate timing.

    Parameters
    ----------
    fn : Runnable
        The kernel function to benchmark.
    inputs : dict
        Input arguments to pass to the kernel function.
    warmup : int
        Number of warmup iterations before timing.
    iters : int
        Number of timing iterations to average over.
    device : str
        The CUDA device to run the benchmark on.

    Returns
    -------
    float
        The average execution time in milliseconds.
    """
    lock = _device_lock(device)
    with lock:
        with torch.no_grad():
            fn(**inputs)
        torch.cuda.synchronize(device=torch.device(device))

        return do_bench(lambda: fn(**inputs), warmup=warmup, rep=iters)
