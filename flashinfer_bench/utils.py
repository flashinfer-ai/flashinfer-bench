"""Utility functions for FlashInfer-Bench."""

from __future__ import annotations

import os
import platform
import sys
from functools import cache
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch

    from flashinfer_bench.data import Environment


@cache
def _get_dtype_str_to_python_dtype() -> Dict[str, type]:
    """Get dtype string to Python type mapping (cached)."""
    return {
        "float32": float,
        "float16": float,
        "bfloat16": float,
        "float8_e4m3fn": float,
        "float8_e5m2": float,
        "float4_e2m1": float,
        "int64": int,
        "int32": int,
        "int16": int,
        "int8": int,
        "bool": bool,
    }


def dtype_str_to_python_dtype(dtype_str: str) -> type:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_python_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_dtype_str_to_torch_dtype() -> Dict[str, torch.dtype]:
    """Lazily build dtype string to torch dtype mapping."""
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_torch_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_integer_dtypes() -> frozenset:
    """Get frozenset of integer and boolean dtypes (cached)."""
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: torch.dtype) -> bool:
    """Check if dtype is an integer or boolean type."""
    return dtype in _get_integer_dtypes()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    import torch

    return torch.cuda.is_available()


def list_cuda_devices() -> List[str]:
    import torch

    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


def env_snapshot(device: str) -> Environment:
    import torch

    from flashinfer_bench.data import Environment

    libs: Dict[str, str] = {"torch": torch.__version__}
    try:
        import triton as _tr

        libs["triton"] = getattr(_tr, "__version__", "unknown")
    except Exception:
        pass

    try:
        import tilelang as _tl

        libs["tilelang"] = getattr(_tl, "__version__", "unknown")
    except Exception:
        pass

    try:
        import torch.version as tv

        if getattr(tv, "cuda", None):
            libs["cuda"] = tv.cuda
    except Exception:
        pass
    return Environment(hardware=hardware_from_device(device), libs=libs)


def hardware_from_device(device: str) -> str:
    import torch

    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.get_device_name(d.index)
    if d.type == "cpu":
        # Best-effort CPU model
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or platform.machine() or "CPU"
    if d.type == "mps":
        return "Apple GPU (MPS)"
    if d.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return torch.xpu.get_device_name(d.index)
        except Exception:
            return "Intel XPU"
    return d.type


def redirect_stdio_to_file(log_path: str) -> tuple[int, int]:
    """Redirect stdout/stderr to log file.

    Returns original stdout and stderr file descriptors for printing to terminal.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    # Redirect stdout/stderr to log file
    os.dup2(fd, 1)  # stdout -> fd
    os.dup2(fd, 2)  # stderr -> fd
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return original_stdout_fd, original_stderr_fd


def flush_stdio_streams() -> None:
    """Best-effort flush of redirected stdout/stderr streams."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
