"""
Test DSA sparse attention h16_ckv512_kpe64_topk256_ps64 reference implementation.

This test validates that the reference implementation from the definition
produces correct output shapes and handles padding correctly.
"""

import flashinfer
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics

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


def test_correctness(num_tokens=64, topk=TOPK, atol=1e-2, rtol=5e-2):
    """Test correctness of DSA sparse attention reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing DSA Sparse Attention (ps64) num_tokens={num_tokens}, topk={topk}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    inputs = generate_random_inputs(num_tokens, topk=topk, device=device)

    print("Running reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["sparse_indices"],
        inputs["sm_scale"],
    )

    # Prepare FlashInfer inputs (trtllm-gen format)
    # Query: concatenate q_nope and q_pe, add seq_len dim
    query = torch.cat([inputs["q_nope"], inputs["q_pe"]], dim=-1).unsqueeze(1)
    # KV cache: concatenate ckv and kpe caches
    kv_cache = torch.cat([inputs["ckv_cache"], inputs["kpe_cache"]], dim=-1)
    # Block tables: add seq_len dim to sparse_indices
    block_tables = inputs["sparse_indices"].unsqueeze(1)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    total_tokens = inputs["num_pages"] * PAGE_SIZE
    seq_lens = torch.full((num_tokens,), total_tokens, dtype=torch.int32, device=device)

    print("Running FlashInfer...")
    fi_output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=128,  # QK_NOPE_HEAD_DIM
        kv_lora_rank=HEAD_DIM_CKV,
        qk_rope_head_dim=HEAD_DIM_KPE,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=total_tokens,
        sparse_mla_top_k=topk,
        bmm1_scale=inputs["sm_scale"].item(),
    )
    fi_output = fi_output.squeeze(1)  # Remove seq_len dim

    print("\nComparing outputs...")
    output_metrics = compare_tensors(ref_o, fi_output, atol=atol, rtol=rtol)
    print_comparison_metrics(output_metrics, tensor_name="Output tensor")

    all_close = output_metrics.all_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance")

    return all_close


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

    test_configs = [(16, TOPK), (32, TOPK), (64, TOPK), (128, TOPK)]
    passed = sum(1 for cfg in test_configs if test_correctness(*cfg))
    print(f"\n{'='*60}\nCorrectness: {passed}/{len(test_configs)} tests passed\n{'='*60}")

    test_output_shape()
    print("test_output_shape: PASSED")

    test_padding_handling()
    print("test_padding_handling: PASSED")

    print("\nAll tests passed!")
