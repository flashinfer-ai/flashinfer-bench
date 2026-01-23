"""
Test DSA sparse attention h16_ckv512_kpe64_topk256_ps64 reference implementation.

This test validates that the reference implementation from the definition
produces correct output shapes and handles padding correctly.
"""

import numpy as np
import pytest
import torch
from test_utils import get_reference_run

# Load reference implementation from definition
run = get_reference_run("dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64")

# Module-level constants (DeepSeek V3/R1 with TP=8)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 256


def generate_random_inputs(num_tokens, topk=TOPK, device="cuda"):
    """Generate random inputs for DSA sparse attention testing."""
    num_pages = max(num_tokens * 2, 1024)

    sparse_indices = torch.randint(
        0, num_pages, (num_tokens, topk), dtype=torch.int32, device=device
    )

    q_nope = torch.randn(
        num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)

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

    output, lse = result

    assert output.shape == (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV)
    assert lse.shape == (num_tokens, NUM_QO_HEADS)


def test_padding_handling(num_tokens=64, topk=TOPK):
    """Test that padding (-1 indices) are handled correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_pages = 1000

    q_nope = torch.randn(
        num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + HEAD_DIM_KPE), dtype=torch.float32, device=device)

    # Create sparse indices with varying amounts of padding per token
    sparse_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)

    total_tokens_in_cache = num_pages * PAGE_SIZE

    for t in range(num_tokens):
        valid_count = (t % 4 + 1) * (topk // 4)
        valid_count = min(valid_count, topk)
        sparse_indices[t, :valid_count] = torch.randint(
            0, total_tokens_in_cache, (valid_count,), dtype=torch.int32, device=device
        )

    result = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output, lse = result

    # Verify outputs are valid
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert not torch.isnan(lse).any()


if __name__ == "__main__":
    print("Testing DSA Sparse Attention Reference (page_size=64, from definition)")
    print(
        f"Constants: h={NUM_QO_HEADS}, ckv={HEAD_DIM_CKV}, kpe={HEAD_DIM_KPE}, ps={PAGE_SIZE}, topk={TOPK}"
    )
    print("=" * 70)

    test_output_shape()
    print("test_output_shape: PASSED")

    test_padding_handling()
    print("test_padding_handling: PASSED")

    print("\nAll tests passed!")
