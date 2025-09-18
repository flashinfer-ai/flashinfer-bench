from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import env_snapshot, torch_dtype_from_def


def rand_tensor(shape: List[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate a random tensor with specified shape, dtype, and device."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)

    # low-precision floats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)

    # booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Normalize various output types to a consistent dict format."""
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        # Python scalar -> 0-D tensor for comparison
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError("Single Tensor returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def load_safetensors(
    defn: Definition, wl: Workload, traceset_root: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """Load tensors from safetensors files specified in workload."""
    try:
        import safetensors.torch as st
    except Exception:
        raise RuntimeError("safetensors is not available in the current environment")

    expected = defn.get_input_shapes(wl.axes)
    stensors: Dict[str, torch.Tensor] = {}
    for name, desc in wl.inputs.items():
        if desc.type != "safetensors":
            continue

        path = desc.path
        if traceset_root is not None and not Path(path).is_absolute():
            path = str(traceset_root / path)

        tensors = st.load_file(path)
        if desc.tensor_key not in tensors:
            raise ValueError(f"Missing key '{desc.tensor_key}' in '{path}'")
        t = tensors[desc.tensor_key]
        # shape check
        if list(t.shape) != expected[name]:
            raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
        # dtype check
        expect_dtype = torch_dtype_from_def(defn.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        stensors[name] = t
    return stensors


def gen_inputs(
    defn: Definition,
    wl: Workload,
    device: str,
    stensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Generate input tensors for a workload."""
    shapes = defn.get_input_shapes(wl.axes)
    dev = torch.device(device)
    out: Dict[str, Any] = {}

    for name, spec in defn.inputs.items():
        dtype = torch_dtype_from_def(spec.dtype)

        if name in wl.inputs and wl.inputs[name].type == "safetensors":
            if stensors is None or name not in stensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = stensors[name]
            out[name] = t_cpu.to(device=dev, non_blocking=True)
        elif name in wl.inputs and wl.inputs[name].type == "scalar":
            out[name] = wl.inputs[name].value
        else:  # random
            if spec.shape is None:
                out[name] = rand_tensor([], dtype, dev).item()
            else:
                shape = shapes[name]
                out[name] = rand_tensor(shape, dtype, dev)
    return out


def make_eval(
    status: EvaluationStatus,
    device: str,
    log_file: str,
    correctness: Optional[Correctness] = None,
    performance: Optional[Performance] = None,
    error: Optional[str] = None,
) -> Evaluation:
    """Create an Evaluation object with common fields."""
    return Evaluation(
        status=status,
        log_file=log_file,
        environment=env_snapshot(device),
        timestamp=datetime.now().isoformat(),
        correctness=correctness,
        performance=performance,
        error=error,
    )
