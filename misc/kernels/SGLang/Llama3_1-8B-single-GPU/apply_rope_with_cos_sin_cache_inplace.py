"""
The implementation of rope application using flashinfer and native PyTorch.

Shapes:
- q: (B, hidden_size)
- k: (B, hidden_size / num_q_heads * num_kv_heads)
- pos_ids: (B,)
- cos_sin_cache: (max_position, rotary_dim)
"""

import os
import flashinfer
import torch
from typing import Optional, Union, Tuple
from sglang.srt.layers.rotary_embedding import get_rope, RotaryEmbedding


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


################### FlashInfer implementation ##################

def forward_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
        q = q,
        k = k,
        q_rope = q_rope,
        k_rope = k_rope,
        cos_sin_cache = cos_sin_cache,
        pos_ids = pos_ids,
        interleave = interleave,
    )
    return q, k

def forward_pytorch(
    rotary_emb: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/sgl-project/sglang/blob/f20f70003d9546ec6e6d5b068b3c5f7ab0521109/sgl-kernel/tests/test_rotary_embedding.py"""
    if offsets is not None:
        positions = positions + offsets

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = rotary_emb.cos_sin_cache.index_select(0, positions)

    # Modification: float32 is required for the rotary embedding to work correctly
    query = query.to(torch.float32)
    key = key.to(torch.float32)

    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, rotary_emb.head_size)
    query_rot = query[..., : rotary_emb.rotary_dim]
    query_pass = query[..., rotary_emb.rotary_dim :]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, rotary_emb.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, rotary_emb.head_size)
    key_rot = key[..., : rotary_emb.rotary_dim]
    key_pass = key[..., rotary_emb.rotary_dim :]
    key_rot = _apply_rotary_emb(key_rot, cos, sin, rotary_emb.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    # Modification: convert to the correct dtype
    query = query.to(rotary_emb.dtype)
    key = key.to(rotary_emb.dtype)
    return query, key

def kernel_descriptions():
    return """
apply_rope_with_cos_sin_cache_inplace(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, interleave)

Shape Variables:
- B: symbolic shape, batch_size
- H: static constant, per-model's hidden_size
- rotary_dim: static constant, per-model
- max_position: static constant, per-model
- num_q_heads: static constant, per-model
- num_kv_heads: static constant, per-model

Input shpes:
- q: (B, H)
- k: (B, H / num_q_heads * num_kv_heads)
- q_rope: (B, H)
- k_rope: (B, H / num_q_heads * num_kv_heads)
- cos_sin_cache: (max_position, rotary_dim)
- pos_ids: (B,)
- interleave: bool

Computation:
- Refer to forward_pytorch().
"""

if __name__ == "__main__":
    # Check with `nvidia-smi --query-gpu=compute_cap --format=csv`
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

    # dynamic size
    B = 16
    
    # Llama3.1-8B constants
    HEAD_SIZE = 128
    HIDDEN_SIZE = 4096
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 8

    ROTARY_DIM = 128
    MAX_POSITION_EMBEDDINGS = 2048
    ROTARY_THETA = 10000.0
    IS_NEOX_STYLE = True
    ROPE_SCALING = None
    DTYPE = torch.bfloat16 # for query and key; cos_sin_cache is fp32
    DEVICE = "cuda"

    # prep inputs
    rotary_emb: RotaryEmbedding = get_rope(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position=MAX_POSITION_EMBEDDINGS,
        base=ROTARY_THETA,
        rope_scaling=ROPE_SCALING,
        is_neox_style=IS_NEOX_STYLE,
        dtype=DTYPE,
    ).to(DEVICE)
    positions = torch.arange(B, device=DEVICE)
    q = torch.randn(B, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, HIDDEN_SIZE // NUM_Q_HEADS * NUM_KV_HEADS, dtype=DTYPE, device=DEVICE)

    # Test
    print("flashinfer")
    # Adapted from `sgl_kernel`
    q_flashinfer_clone = q.clone()
    k_flashinfer_clone = k.clone()
    q_flashinfer, k_flashinfer = forward_flashinfer(
        q_flashinfer_clone.view(q.shape[0], -1, HEAD_SIZE),
        k_flashinfer_clone.view(k.shape[0], -1, HEAD_SIZE),
        q_flashinfer_clone.view(q.shape[0], -1, HEAD_SIZE),
        k_flashinfer_clone.view(k.shape[0], -1, HEAD_SIZE),
        rotary_emb.cos_sin_cache,
        positions.long(),
        (not IS_NEOX_STYLE),
    )

    print("native")
    q_native_clone = q.clone()
    k_native_clone = k.clone()
    q_native, k_native = forward_pytorch(
        rotary_emb,
        positions,
        q_native_clone,
        k_native_clone,
    )

    print("assert")
    torch.testing.assert_close(q_flashinfer.view(q.shape), q_native)
    torch.testing.assert_close(k_flashinfer.view(k.shape), k_native)

