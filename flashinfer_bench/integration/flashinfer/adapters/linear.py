from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _resolve_def_name(weight: torch.Tensor) -> str:
    out_features, in_features = weight.shape
    return f"gemm_n{out_features}_k{in_features}"


def _extract_arg(
    name: str, position: int, args: Sequence[Any], kwargs: Dict[str, Any]
) -> Optional[Any]:
    if name in kwargs:
        return kwargs[name]
    if position < len(args):
        return args[position]
    return None


class LinearAdapter:
    """Adapter for torch.nn.functional.linear using GEMM traces."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="torch.nn.functional.linear",
                kind="function",
                name="linear",
                ctx_key="gemm_linear",
            )
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_tensor = _extract_arg("input", 0, args, kwargs)
            weight = _extract_arg("weight", 1, args, kwargs)
            bias = _extract_arg("bias", 2, args, kwargs)

            if not isinstance(input_tensor, torch.Tensor) or not isinstance(weight, torch.Tensor):
                return orig(*args, **kwargs)

            # Only handle 2D GEMM style activations for now.
            if weight.dim() != 2 or input_tensor.shape[-1] != weight.shape[1]:
                return orig(*args, **kwargs)

            dtype = input_tensor.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                return orig(*args, **kwargs)
            if weight.dtype != dtype:
                return orig(*args, **kwargs)
            if bias is not None:
                return orig(*args, **kwargs)
            if input_tensor.device != weight.device or not input_tensor.is_cuda:
                return orig(*args, **kwargs)

            leading_shape: Sequence[int] = input_tensor.shape[:-1]
            A = input_tensor.reshape(-1, weight.shape[1])
            rk: Dict[str, Any] = {"A": A, "B": weight}

            def _fb(**_rk):
                return orig(*args, **kwargs)

            try:
                out_2d = apply(_resolve_def_name(weight), runtime_kwargs=rk, fallback=_fb)
            except Exception:
                return orig(*args, **kwargs)

            if not isinstance(out_2d, torch.Tensor):
                return out_2d

            out = out_2d.reshape(*leading_shape, weight.shape[0])
            return out

        return wrapper
