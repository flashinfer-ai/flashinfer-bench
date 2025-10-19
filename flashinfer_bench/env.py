"""Defines the environment variables used in FlashInfer-Bench."""

import os
from pathlib import Path


def get_fib_enable_apply() -> bool:
    """Get the value of the FIB_ENABLE_APPLY environment variable. It controls whether the apply
    functionality is enabled.

    Returns
    -------
    bool
        The value of the FIB_ENABLE_APPLY environment variable.
    """
    return os.environ.get("FIB_ENABLE_APPLY", "0") == "1"


def get_fib_enable_tracing() -> bool:
    """Get the value of the FIB_ENABLE_TRACING environment variable. It controls whether the tracing
    functionality is enabled.

    Returns
    -------
    bool
        The value of the FIB_ENABLE_TRACING environment variable.
    """
    return os.environ.get("FIB_ENABLE_TRACING", "0") == "1"


def get_fib_dataset_path() -> Path:
    """Get the value of the FIB_DATASET_PATH environment variable. It controls the path to the
    dataset to dump or to load.

    Returns
    -------
    Path
        The value of the FIB_DATASET_PATH environment variable.
    """
    value = os.environ.get("FIB_DATASET_PATH")
    if value:
        return Path(value).expanduser()
    return Path(Path.home() / ".cache" / "flashinfer_bench" / "dataset")


def get_fib_cache_path() -> Path:
    """Get the value of the FIB_CACHE_PATH environment variable. It controls the path to the cache.

    Returns
    -------
    Path
        The value of the FIB_CACHE_PATH environment variable.
    """
    value = os.environ.get("FIB_CACHE_PATH")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / "flashinfer_bench" / "cache"
