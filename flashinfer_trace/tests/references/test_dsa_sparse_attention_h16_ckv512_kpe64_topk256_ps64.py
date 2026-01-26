"""
Tests for DSA (DeepSeek Sparse Attention) sparse attention reference implementation.
Page size 64 variant.

Ground truth comparison tests are in test_dsa_vs_definition_reference.py
which tests against FlashInfer's trtllm_batch_decode_with_kv_cache_mla.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Module-level constants (DeepSeek V3/R1 with TP=8)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 256

TRACE_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Reference implementation for DSA sparse attention with page_size=64."""
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert head_dim_ckv == HEAD_DIM_CKV
    assert head_dim_kpe == HEAD_DIM_KPE
    assert page_size == PAGE_SIZE
    assert topk == TOPK

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]

        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]
        Kp = Kp_all[tok_idx]
        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)

        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[t] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}


def generate_random_inputs(
    num_tokens,
    num_qo_heads=NUM_QO_HEADS,
    head_dim_ckv=HEAD_DIM_CKV,
    head_dim_kpe=HEAD_DIM_KPE,
    topk=TOPK,
    device="cuda",
):
    """Generate random inputs for DSA sparse attention testing with page_size=64."""
    total_kv_tokens = max(num_tokens * 4, 2048)
    num_pages = (total_kv_tokens + PAGE_SIZE - 1) // PAGE_SIZE

    total_tokens_in_cache = num_pages * PAGE_SIZE
    sparse_indices = torch.randint(
        0, total_tokens_in_cache, (num_tokens, topk), dtype=torch.int32, device=device
    )

    q_nope = torch.randn(
        num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    ckv_cache = torch.randn(num_pages, PAGE_SIZE, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, head_dim_kpe, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": torch.tensor(sm_scale, dtype=torch.float32, device=device),
        "num_pages": num_pages,
    }


def test_output_shape(num_tokens=64, topk=TOPK):
    """Test that reference produces correct output shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = generate_random_inputs(num_tokens, topk=topk, device=device)

    result = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["sparse_indices"],
        inputs["sm_scale"],
    )

    output = result["output"]
    lse = result["lse"]

    assert output.shape == (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV)
    assert lse.shape == (num_tokens, NUM_QO_HEADS)


def test_padding_handling(num_tokens=64, topk=TOPK):
    """Test that padding (-1 indices) are handled correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_pages = 64

    q_nope = torch.randn(
        num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + HEAD_DIM_KPE), dtype=torch.float32, device=device)

    sparse_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
    total_tokens_in_cache = num_pages * PAGE_SIZE

    for t in range(num_tokens):
        valid_count = (t % 4 + 1) * (topk // 4)
        valid_count = min(valid_count, topk)
        sparse_indices[t, :valid_count] = torch.randint(
            0, total_tokens_in_cache, (valid_count,), dtype=torch.int32, device=device
        )

    result = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output = result["output"]
    lse = result["lse"]

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert not torch.isnan(lse).any()


if __name__ == "__main__":
    print("Testing DSA Sparse Attention Reference (page_size=64)")
    print(
        f"Constants: h={NUM_QO_HEADS}, ckv={HEAD_DIM_CKV}, kpe={HEAD_DIM_KPE}, ps={PAGE_SIZE}, topk={TOPK}"
    )
    print("=" * 70)

    test_output_shape()
    print("test_output_shape: PASSED")

    test_padding_handling()
    print("test_padding_handling: PASSED")

    print("\nAll tests passed!")
