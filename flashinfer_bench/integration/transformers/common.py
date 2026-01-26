"""Common utilities for Transformers integration adapters."""

from __future__ import annotations

from typing import Optional

import torch


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

    causal_str = "causal" if is_causal else "full"
    return f"gqa_ragged_prefill_{causal_str}_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}"


def infer_rmsnorm_def_name(weight: torch.Tensor) -> str:
    """
    Infer the definition name for an RMSNorm operation.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor of shape [hidden_size]

    Returns
    -------
    str
        Definition name like "rmsnorm_h4096"
    """
    hidden_size = weight.shape[0]
    return f"rmsnorm_h{hidden_size}"


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
    return f"embedding_v{vocab_size}_d{embedding_dim}"


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
    if activation_type == "gelu" and approximate == "tanh":
        return f"gelu_tanh_h{hidden_size}"
    return f"{activation_type}_h{hidden_size}"


def infer_moe_def_name(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    implementation: str = "batched",
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
    return f"moe_{implementation}_e{num_experts}_h{hidden_dim}_i{intermediate_dim}_topk{top_k}"


def infer_sampling_def_name(vocab_size: int, method: str = "multinomial") -> str:
    """
    Infer the definition name for a sampling operation.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    method : str
        Sampling method ("multinomial", "softmax", "topk")

    Returns
    -------
    str
        Definition name like "sampling_multinomial_v32000"
    """
    return f"sampling_{method}_v{vocab_size}"


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
    return f"rope_h{num_heads}_d{head_dim}"


def get_dtype_str(dtype: torch.dtype) -> Optional[str]:
    """Convert torch dtype to string representation."""
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    return dtype_map.get(dtype)
