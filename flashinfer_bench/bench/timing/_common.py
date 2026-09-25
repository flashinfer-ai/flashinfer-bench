"""Shared helpers for the timing subpackage (private)."""

from __future__ import annotations

from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Dict

_device_locks: Dict[str, LockType] = {}
_registry_lock = Lock()


def _device_lock(device: str) -> LockType:
    """Get or create a multiprocessing lock for the specified device.

    Maintains a registry of locks per device to serialize benchmarking
    operations on the same device, preventing interference between concurrent
    measurements (e.g. CUPTI activity buffers from two timers racing).

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
