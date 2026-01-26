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

    Patches RMSNorm forward methods in various transformers models.
    Transformers models implement their own RMSNorm classes that don't use
    torch.nn.functional.rms_norm, so we need to patch the model-specific classes.
    """

    def targets(self) -> List[PatchSpec]:
        # Patch RMSNorm forward methods in various model implementations
        model_paths = [
            "transformers.models.llama.modeling_llama.LlamaRMSNorm",
            "transformers.models.llama4.modeling_llama4.Llama4RMSNorm",
            "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm",
            "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm",
            "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
            "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm",
            "transformers.models.mistral.modeling_mistral.MistralRMSNorm",
            "transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm",
            "transformers.models.gpt_oss.modeling_gpt_oss.GptOssRMSNorm",
            "transformers.models.gemma.modeling_gemma.GemmaRMSNorm",
            "transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm",
            "transformers.models.phi3.modeling_phi3.Phi3RMSNorm",
            "transformers.models.cohere.modeling_cohere.CohereLayerNorm",
            "transformers.models.cohere2.modeling_cohere2.Cohere2LayerNorm",
        ]
        
        specs = [
            # Also patch torch.nn.functional.rms_norm for models that use it
            PatchSpec(
                path="torch.nn.functional.rms_norm",
                kind="function",
                name="rms_norm",
                ctx_key="torch_rmsnorm",
            ),
        ]
        
        for path in model_paths:
            model_name = path.split(".")[-1]
            specs.append(
                PatchSpec(
                    path=f"{path}.forward",
                    kind="method",
                    name="forward",
                    ctx_key=f"rmsnorm_{model_name}",
                )
            )
        
        return specs

    def make_wrapper(
        self, spec: PatchSpec, orig: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper function that traces RMSNorm calls."""
        
        if spec.kind == "function":
            # Wrapper for torch.nn.functional.rms_norm
            return self._make_functional_wrapper(orig)
        else:
            # Wrapper for RMSNorm class forward methods
            return self._make_class_wrapper(orig)
    
    def _make_functional_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for torch.nn.functional.rms_norm."""

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
    
    def _make_class_wrapper(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for RMSNorm class forward methods."""

        def wrapper(self_module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            # Validate inputs
            if not isinstance(hidden_states, torch.Tensor):
                return orig(self_module, hidden_states)

            # Only trace on CUDA with supported dtypes
            if not hidden_states.is_cuda:
                return orig(self_module, hidden_states)

            if hidden_states.dtype not in SUPPORTED_ACTIVATION_DTYPES:
                return orig(self_module, hidden_states)

            # Get weight from the module
            weight = getattr(self_module, "weight", None)
            if weight is None:
                return orig(self_module, hidden_states)

            # Check weight shape compatibility
            if weight.dim() != 1:
                return orig(self_module, hidden_states)

            hidden_size = weight.shape[0]

            # Validate input shape matches weight
            if hidden_states.shape[-1] != hidden_size:
                return orig(self_module, hidden_states)

            def_name = infer_rmsnorm_def_name(weight, hidden_states.dtype)

            # Reshape input to 2D for the trace: [batch * seq_len, hidden_size]
            original_shape = hidden_states.shape
            input_2d = hidden_states.reshape(-1, hidden_size)

            rk: Dict[str, Any] = {
                "hidden_states": input_2d,
                "weight": weight,
            }

            def _fallback(**_rk):
                return orig(self_module, hidden_states)

            ret = apply(def_name, kwargs=rk, fallback=_fallback)

            # Reshape output back to original shape if needed
            if isinstance(ret, torch.Tensor):
                return ret.reshape(original_shape)

            return ret

        return wrapper
