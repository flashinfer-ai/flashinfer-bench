"""Common utilities for Transformers integration adapters."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# Supported dtypes for tracing
# Standard floating point types
STANDARD_FLOAT_DTYPES: Tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# FP8 types (available in PyTorch 2.1+)
FP8_DTYPES: Tuple[torch.dtype, ...] = ()
try:
    FP8_DTYPES = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
except AttributeError:
    pass  # FP8 not available in this PyTorch version

# MXFP4 types (microscaling FP4, available in PyTorch 2.4+ or via custom extensions)
MXFP4_DTYPES: Tuple[torch.dtype, ...] = ()
try:
    # PyTorch native FP4 support (if available)
    MXFP4_DTYPES = (torch.float4_e2m1fn,) if hasattr(torch, "float4_e2m1fn") else ()
except AttributeError:
    pass

# All supported dtypes for tracing activation tensors
# Most operations work with fp16/bf16 activations even when weights are quantized
SUPPORTED_ACTIVATION_DTYPES: Tuple[torch.dtype, ...] = (
    STANDARD_FLOAT_DTYPES + FP8_DTYPES + MXFP4_DTYPES
)

# Supported dtypes for weight tensors (includes quantized types)
SUPPORTED_WEIGHT_DTYPES: Tuple[torch.dtype, ...] = (
    STANDARD_FLOAT_DTYPES + FP8_DTYPES + MXFP4_DTYPES
)

# For operations that can work with any floating type
SUPPORTED_FLOAT_DTYPES: Tuple[torch.dtype, ...] = (
    STANDARD_FLOAT_DTYPES + FP8_DTYPES + MXFP4_DTYPES
)


def get_dtype_suffix(dtype: torch.dtype) -> str:
    """Get a suffix string for the dtype to append to definition names.
    
    Returns empty string for standard dtypes (fp16/bf16/fp32) since they use
    the default definitions. Returns a suffix like "_fp8e4m3" for FP8 types.
    
    Parameters
    ----------
    dtype : torch.dtype
        The tensor dtype
        
    Returns
    -------
    str
        Suffix string to append to definition name, or empty string
    """
    # Standard types don't need a suffix
    if dtype in STANDARD_FLOAT_DTYPES:
        return ""
    
    # FP8 types
    try:
        if dtype == torch.float8_e4m3fn:
            return "_fp8e4m3"
        if dtype == torch.float8_e5m2:
            return "_fp8e5m2"
    except AttributeError:
        pass
    
    # MXFP4 types
    if hasattr(torch, "float4_e2m1fn") and dtype == torch.float4_e2m1fn:
        return "_mxfp4"
    
    return ""


def infer_attention_def_name(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = True,
) -> str:
    """
    Infer the definition name for a GQA ragged attention operation.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor of shape [batch, num_q_heads, seq_len, head_dim]
    key : torch.Tensor
        Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    is_causal : bool
        Whether the attention is causal

    Returns
    -------
    str
        Definition name like "gqa_ragged_prefill_causal_h32_kv8_d128"
    """
    # Query shape: [batch, num_q_heads, seq_len, head_dim]
    # Key shape: [batch, num_kv_heads, seq_len, head_dim]
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = query.shape[-1]
    dtype_suffix = get_dtype_suffix(query.dtype)

    causal_str = "causal" if is_causal else "full"
    return f"gqa_ragged_prefill_{causal_str}_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}{dtype_suffix}"


def infer_rmsnorm_def_name(weight: torch.Tensor, input_dtype: Optional[torch.dtype] = None) -> str:
    """
    Infer the definition name for an RMSNorm operation.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor of shape [hidden_size]
    input_dtype : Optional[torch.dtype]
        The dtype of the input tensor (for dtype suffix)

    Returns
    -------
    str
        Definition name like "rmsnorm_h4096"
    """
    hidden_size = weight.shape[0]
    dtype_suffix = get_dtype_suffix(input_dtype) if input_dtype else ""
    return f"rmsnorm_h{hidden_size}{dtype_suffix}"


def infer_embedding_def_name(weight: torch.Tensor) -> str:
    """
    Infer the definition name for an embedding operation.

    Parameters
    ----------
    weight : torch.Tensor
        Embedding weight tensor of shape [vocab_size, embedding_dim]

    Returns
    -------
    str
        Definition name like "embedding_v32000_d4096"
    """
    vocab_size, embedding_dim = weight.shape
    dtype_suffix = get_dtype_suffix(weight.dtype)
    return f"embedding_v{vocab_size}_d{embedding_dim}{dtype_suffix}"


def infer_activation_def_name(
    input_tensor: torch.Tensor,
    activation_type: str,
    approximate: str = "none",
) -> str:
    """
    Infer the definition name for an activation operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    activation_type : str
        Type of activation ("silu" or "gelu")
    approximate : str
        For GELU, the approximation method ("none" or "tanh")

    Returns
    -------
    str
        Definition name like "silu_h4096" or "gelu_tanh_h4096"
    """
    hidden_size = input_tensor.shape[-1]
    dtype_suffix = get_dtype_suffix(input_tensor.dtype)
    if activation_type == "gelu" and approximate == "tanh":
        return f"gelu_tanh_h{hidden_size}{dtype_suffix}"
    return f"{activation_type}_h{hidden_size}{dtype_suffix}"


def infer_moe_def_name(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    implementation: str = "batched",
    dtype: Optional[torch.dtype] = None,
) -> str:
    """
    Infer the definition name for a MoE operation.

    Parameters
    ----------
    num_experts : int
        Number of experts
    hidden_dim : int
        Hidden dimension
    intermediate_dim : int
        Intermediate dimension of expert MLP
    top_k : int
        Number of experts selected per token
    implementation : str
        MoE implementation type ("batched" or "grouped")

    Returns
    -------
    str
        Definition name like "moe_batched_e8_h4096_i14336_topk2"
    """
    dtype_suffix = get_dtype_suffix(dtype) if dtype else ""
    return f"moe_{implementation}_e{num_experts}_h{hidden_dim}_i{intermediate_dim}_topk{top_k}{dtype_suffix}"


def infer_sampling_def_name(
    vocab_size: int, method: str = "multinomial", dtype: Optional[torch.dtype] = None
) -> str:
    """
    Infer the definition name for a sampling operation.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    method : str
        Sampling method ("multinomial", "softmax", "topk")
    dtype : Optional[torch.dtype]
        The dtype of the input tensor

    Returns
    -------
    str
        Definition name like "sampling_multinomial_v32000"
    """
    dtype_suffix = get_dtype_suffix(dtype) if dtype else ""
    return f"sampling_{method}_v{vocab_size}{dtype_suffix}"


def infer_rope_def_name(q: "torch.Tensor") -> str:
    """
    Infer the definition name for a RoPE (Rotary Position Embedding) operation.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch, num_heads, seq_len, head_dim]

    Returns
    -------
    str
        Definition name like "rope_h32_d128"
    """
    num_heads = q.shape[1]
    head_dim = q.shape[-1]
    dtype_suffix = get_dtype_suffix(q.dtype)
    return f"rope_h{num_heads}_d{head_dim}{dtype_suffix}"


def get_dtype_str(dtype: torch.dtype) -> Optional[str]:
    """Convert torch dtype to string representation."""
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    # Add FP8 types if available
    try:
        dtype_map[torch.float8_e4m3fn] = "float8_e4m3fn"
        dtype_map[torch.float8_e5m2] = "float8_e5m2"
    except AttributeError:
        pass
    # Add MXFP4 types if available
    if hasattr(torch, "float4_e2m1fn"):
        dtype_map[torch.float4_e2m1fn] = "float4_e2m1"
    return dtype_map.get(dtype)
