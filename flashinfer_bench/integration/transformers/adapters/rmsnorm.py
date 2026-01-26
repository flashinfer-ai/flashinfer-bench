"""RMSNorm adapter for transformers integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.transformers.common import (
    SUPPORTED_ACTIVATION_DTYPES,
    infer_rmsnorm_def_name,
)


class RMSNormAdapter:
    """Adapter for RMSNorm operations.

    Patches torch.nn.functional.rms_norm (available in PyTorch 2.4+).
    This captures RMSNorm calls from various transformers models.
    """

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="torch.nn.functional.rms_norm",
                kind="function",
                name="rms_norm",
                ctx_key="torch_rmsnorm",
            ),
        ]

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces RMSNorm calls."""

        def wrapper(
            input: torch.Tensor,
            normalized_shape: List[int],
            weight: torch.Tensor | None = None,
            eps: float | None = None,
        ) -> torch.Tensor:
            # Validate inputs
            if not isinstance(input, torch.Tensor):
                return orig(input, normalized_shape, weight, eps)

            if weight is None:
                return orig(input, normalized_shape, weight, eps)

            # Check for supported dtypes
            if input.dtype not in SUPPORTED_ACTIVATION_DTYPES:
                return orig(input, normalized_shape, weight, eps)

            # Check weight shape compatibility
            if weight.dim() != 1:
                return orig(input, normalized_shape, weight, eps)

            hidden_size = weight.shape[0]

            # Validate input shape matches weight
            if input.shape[-1] != hidden_size:
                return orig(input, normalized_shape, weight, eps)

            def_name = infer_rmsnorm_def_name(weight, input.dtype)

            # Reshape input to 2D for the trace: [batch * seq_len, hidden_size]
            original_shape = input.shape
            input_2d = input.reshape(-1, hidden_size)

            rk: Dict[str, Any] = {
                "hidden_states": input_2d,
                "weight": weight,
            }

            def _fallback(**_rk):
                return orig(input, normalized_shape, weight, eps)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # Reshape output back to original shape if needed
            if isinstance(ret, torch.Tensor):
                return ret.reshape(original_shape)

            return ret

        return wrapper
