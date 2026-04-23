"""Utility functions for FlashInfer-Bench."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import weakref
from functools import cache
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch

    from flashinfer_bench.data import Environment

_logger = logging.getLogger(__name__)


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


# ── Subprocess lifecycle helpers ──
#
# These helpers ensure that external subprocesses (compute-sanitizer, ncu, ...)
# and their descendants (_solution_runner) don't leak GPU memory when the
# outer process is killed or when the subprocess times out. subprocess.run's
# default timeout behaviour only SIGKILLs the immediate child; for tools that
# re-exec a Python runner, that leaves the runner as an orphan holding the
# GPU. We wrap every such call in a new session so the whole process group
# can be reaped with killpg, and track active Popens so graceful server
# shutdown can interrupt worker threads blocked in communicate().


_active_procs: "weakref.WeakSet[subprocess.Popen]" = weakref.WeakSet()
_active_procs_lock = threading.Lock()


def set_parent_death_signal(sig: int = signal.SIGKILL) -> None:
    """Install PR_SET_PDEATHSIG so the calling process receives ``sig`` when its
    immediate parent dies. Linux-only; no-op on other platforms or if prctl is
    unavailable.

    Call this from long-running worker scripts launched as grandchildren of a
    managed parent (e.g. ``_solution_runner`` under compute-sanitizer/ncu) so
    they are reaped automatically if the parent is SIGKILLed.
    """
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            _logger.warning("prctl(PR_SET_PDEATHSIG) failed: errno=%d", ctypes.get_errno())
    except OSError as e:
        _logger.warning("Failed to set parent death signal: %s", e)


def _kill_process_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    """Best-effort SIGKILL of the subprocess's whole process group."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError) as e:
        _logger.debug("killpg(%d, %d) failed: %s", pgid, sig, e)


def kill_all_tracked_subprocesses() -> int:
    """Kill the process group of every subprocess currently tracked by
    :func:`run_managed_subprocess`.

    Returns the number of groups signaled. Safe to call multiple times; safe
    to call from a signal handler or during interpreter shutdown.
    """
    with _active_procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        _kill_process_group(proc)
    return len(procs)


def run_managed_subprocess(
    cmd: List[str], *, timeout: float, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    """Run a subprocess that owns a fresh session/process group, guaranteeing
    that descendants (e.g. ``_solution_runner`` under compute-sanitizer) are
    reaped on timeout, exception, or server shutdown.

    Semantics mirror ``subprocess.run(cmd, capture_output=True, text=True,
    env=env, timeout=timeout, start_new_session=True)`` with these additions:

      * on ``TimeoutExpired``, the whole process group is SIGKILLed before
        raising, so grandchildren don't leak
      * the child installs ``PR_SET_PDEATHSIG(SIGKILL)`` via ``preexec_fn`` so
        the direct child (compute-sanitizer / ncu) is reaped automatically if
        the current process is killed before graceful shutdown runs
      * while running, the Popen is registered in a module-level weakset so
        :func:`kill_all_tracked_subprocesses` (called from the server's
        graceful shutdown path) can reach it
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
        preexec_fn=set_parent_death_signal if sys.platform == "linux" else None,
    )
    with _active_procs_lock:
        _active_procs.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            # typeshed types TimeoutExpired.std{out,err} as bytes; in text mode
            # CPython stores the captured str exactly like CompletedProcess.
            exc.stdout = stdout  # type: ignore[assignment]
            exc.stderr = stderr  # type: ignore[assignment]
            raise
        except BaseException:
            _kill_process_group(proc)
            raise
        return subprocess.CompletedProcess(
            args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr
        )
    finally:
        with _active_procs_lock:
            _active_procs.discard(proc)
        if proc.poll() is None:
            _kill_process_group(proc)


def redirect_stdio_to_tempfile(path: str | None = None) -> str:
    """Redirect stdout/stderr to a temporary file.

    Parameters
    ----------
    path : str or None
        If provided, use this path for the log file instead of creating a new
        temporary file. This allows a parent process to pre-allocate the path
        so it can read the log even if the child process crashes.

    Returns the path to the temporary file.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    if path is not None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    else:
        fd, path = tempfile.mkstemp(suffix=".log", prefix="fib_")
    os.dup2(fd, 1)  # stdout -> fd
    os.dup2(fd, 2)  # stderr -> fd
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return path
