"""Sampling adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec


def _infer_sampling_def_name(vocab_size: int, method: str = "multinomial") -> str:
    """Infer definition name for sampling operation."""
    return f"sampling_{method}_v{vocab_size}"


def _infer_softmax_def_name(dim_size: int) -> str:
    """Infer definition name for softmax operation."""
    return f"softmax_d{dim_size}"


def _infer_topk_def_name(dim_size: int, k: int) -> str:
    """Infer definition name for top-k operation."""
    return f"topk_d{dim_size}_k{k}"


class SamplingAdapter:
    """Adapter for sampling operations.

    Traces common sampling operations used in LLM generation:
    - torch.multinomial (token sampling)
    - torch.nn.functional.softmax (probability computation)
    - torch.topk (top-k filtering)
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="torch.multinomial",
                kind="function",
                name="multinomial",
                ctx_key="torch_multinomial",
            ),
            PatchSpec(
                path="torch.nn.functional.softmax",
                kind="function",
                name="softmax",
                ctx_key="torch_softmax",
            ),
            PatchSpec(
                path="torch.topk",
                kind="function",
                name="topk",
                ctx_key="torch_topk",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces sampling calls."""

        if spec.name == "multinomial":
            return self._make_multinomial_wrapper(orig)
        elif spec.name == "softmax":
            return self._make_softmax_wrapper(orig)
        elif spec.name == "topk":
            return self._make_topk_wrapper(orig)
        else:
            return orig

    def _make_multinomial_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for multinomial sampling."""

        def wrapper(
            input: torch.Tensor,
            num_samples: int,
            replacement: bool = False,
            *,
            generator: Any = None,
            out: Any = None,
        ) -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, num_samples, replacement, generator=generator, out=out)

            # Only trace on CUDA with supported dtypes
            if not input.is_cuda:
                return orig(input, num_samples, replacement, generator=generator, out=out)

            if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return orig(input, num_samples, replacement, generator=generator, out=out)

            vocab_size = input.shape[-1]
            def_name = _infer_sampling_def_name(vocab_size, "multinomial")

            # Reshape to 2D if needed
            original_shape = input.shape
            if input.dim() > 2:
                input_2d = input.reshape(-1, vocab_size)
            else:
                input_2d = input

            rk: Dict[str, Any] = {
                "probs": input_2d,
                "num_samples": num_samples,
            }

            def _fallback(**_rk):
                return orig(input, num_samples, replacement, generator=generator, out=out)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            return ret

        return wrapper

    def _make_softmax_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for softmax."""

        def wrapper(
            input: torch.Tensor,
            dim: int | None = None,
            _stacklevel: int = 3,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, dim, _stacklevel, dtype)

            # Only trace on CUDA with supported dtypes
            if not input.is_cuda:
                return orig(input, dim, _stacklevel, dtype)

            if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return orig(input, dim, _stacklevel, dtype)

            # Only trace softmax over the last dimension (vocabulary)
            if dim is None:
                dim = -1
            if dim != -1 and dim != input.dim() - 1:
                return orig(input, dim, _stacklevel, dtype)

            dim_size = input.shape[dim]
            def_name = _infer_softmax_def_name(dim_size)

            # Reshape to 2D for tracing
            original_shape = input.shape
            input_2d = input.reshape(-1, dim_size)

            rk: Dict[str, Any] = {
                "input": input_2d,
            }

            def _fallback(**_rk):
                return orig(input, dim, _stacklevel, dtype)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            if isinstance(ret, torch.Tensor):
                return ret.reshape(original_shape)

            return ret

        return wrapper

    def _make_topk_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for top-k."""

        def wrapper(
            input: torch.Tensor,
            k: int,
            dim: int = -1,
            largest: bool = True,
            sorted: bool = True,
            *,
            out: Any = None,
        ) -> Any:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, k, dim, largest, sorted, out=out)

            # Only trace on CUDA with supported dtypes
            if not input.is_cuda:
                return orig(input, k, dim, largest, sorted, out=out)

            if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return orig(input, k, dim, largest, sorted, out=out)

            # Only trace top-k over the last dimension
            if dim != -1 and dim != input.dim() - 1:
                return orig(input, k, dim, largest, sorted, out=out)

            dim_size = input.shape[dim]
            def_name = _infer_topk_def_name(dim_size, k)

            # Reshape to 2D for tracing
            original_shape = input.shape
            input_2d = input.reshape(-1, dim_size)

            rk: Dict[str, Any] = {
                "input": input_2d,
                "k": k,
            }

            def _fallback(**_rk):
                return orig(input, k, dim, largest, sorted, out=out)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            return ret

        return wrapper
