import os
import platform
import sys
from typing import Dict, List

import torch

from flashinfer_bench.data import Environment


def torch_dtype_from_def(dtype_str: str):
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    table = {
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
    dtype = table.get(dtype_str.lower(), None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


def list_cuda_devices() -> List[str]:
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


def env_snapshot(device: str) -> Environment:
    libs: Dict[str, str] = {"torch": torch.__version__}
    try:
        import triton as _tr

        libs["triton"] = getattr(_tr, "__version__", "unknown")
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


def redirect_stdio_to_file(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    # Redirect stdout/stderr to log file
    os.dup2(fd, 1)  # stdout -> fd
    os.dup2(fd, 2)  # stderr -> fd
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
