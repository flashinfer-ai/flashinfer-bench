from __future__ import annotations

import threading

import torch
from triton.testing import do_bench

from flashinfer_bench.compile import Runnable

# device -> lock registry
_device_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _device_lock(device: str) -> threading.Lock:
    with _registry_lock:
        lock = _device_locks.get(device)
        if lock is None:
            lock = threading.Lock()
            _device_locks[device] = lock
        return lock


def time_runnable(fn: Runnable, inputs: dict, warmup: int, iters: int, device: str) -> float:
    lock = _device_lock(device)
    with lock:
        with torch.no_grad():
            fn(**inputs)
        torch.cuda.synchronize(device=torch.device(device))

        return do_bench(lambda: fn(**inputs), warmup=warmup, rep=iters)
