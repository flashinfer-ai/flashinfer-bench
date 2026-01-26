"""Activation adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _infer_silu_def_name(input_tensor: torch.Tensor) -> str:
    """Infer definition name for SiLU activation."""
    hidden_size = input_tensor.shape[-1]
    return f"silu_h{hidden_size}"


def _infer_gelu_def_name(input_tensor: torch.Tensor, approximate: str = "none") -> str:
    """Infer definition name for GELU activation."""
    hidden_size = input_tensor.shape[-1]
    suffix = "_tanh" if approximate == "tanh" else ""
    return f"gelu{suffix}_h{hidden_size}"


class ActivationAdapter:
    """Adapter for activation functions.

    Traces common activation functions used in LLMs:
    - torch.nn.functional.silu (SwiGLU activation used in LLaMA, Qwen)
    - torch.nn.functional.gelu (used in GPT models)
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="torch.nn.functional.silu",
                kind="function",
                name="silu",
                ctx_key="torch_silu",
            ),
            PatchSpec(
                path="torch.nn.functional.gelu",
                kind="function",
                name="gelu",
                ctx_key="torch_gelu",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces activation calls."""

        if spec.name == "silu":
            return self._make_silu_wrapper(orig)
        elif spec.name == "gelu":
            return self._make_gelu_wrapper(orig)
        else:
            return orig

    def _make_silu_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for SiLU activation."""

        def wrapper(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, inplace)

            # Only trace on CUDA with supported dtypes
            if not input.is_cuda:
                return orig(input, inplace)

            if input.dtype not in (torch.float16, torch.bfloat16):
                return orig(input, inplace)

            def_name = _infer_silu_def_name(input)

            # Reshape to 2D for tracing
            original_shape = input.shape
            hidden_size = input.shape[-1]
            input_2d = input.reshape(-1, hidden_size)

            rk: Dict[str, Any] = {
                "input": input_2d,
            }

            def _fallback(**_rk):
                return orig(input, inplace)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            if isinstance(ret, torch.Tensor):
                return ret.reshape(original_shape)

            return ret

        return wrapper

    def _make_gelu_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for GELU activation."""

        def wrapper(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, approximate)

            # Only trace on CUDA with supported dtypes
            if not input.is_cuda:
                return orig(input, approximate)

            if input.dtype not in (torch.float16, torch.bfloat16):
                return orig(input, approximate)

            def_name = _infer_gelu_def_name(input, approximate)

            # Reshape to 2D for tracing
            original_shape = input.shape
            hidden_size = input.shape[-1]
            input_2d = input.reshape(-1, hidden_size)

            rk: Dict[str, Any] = {
                "input": input_2d,
            }

            def _fallback(**_rk):
                return orig(input, approximate)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            if isinstance(ret, torch.Tensor):
                return ret.reshape(original_shape)

            return ret

        return wrapper
