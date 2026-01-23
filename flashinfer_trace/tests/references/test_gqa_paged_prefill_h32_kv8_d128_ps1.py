"""
Test GQA paged prefill h32_kv8_d128_ps1 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import math

import flashinfer
import torch

from test_utils import get_reference_run

# Load reference implementation from definition
run = get_reference_run("gqa_paged_prefill_causal_h32_kv8_d128_ps1")

# Constants from definition
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 1


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    max_pages,
    causal=True,
    device="cuda",
):
    """Generate random inputs for paged prefill testing."""

    # Generate random query lengths for each batch element
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)

    # Generate random KV lengths for each batch element
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        if causal:
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()
        else:
            kv_lens[i] = torch.randint(1, max_kv_len + 1, (1,)).item()

    # Create indptr arrays
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    # Get total tokens
    total_q = qo_indptr[-1].item()
    num_kv_indices = kv_indptr[-1].item()

    # Generate page indices
    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]

    # Create kv_indices by assigning pages to each sequence
    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        kv_indices[idx : idx + seq_len] = all_page_ids[idx : idx + seq_len]
        idx += seq_len

    # Generate KV cache
    k_cache = torch.randn(
        max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # Generate query tensor
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # For page_size=1, last_page_len is always all ones
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "num_kv_indices": num_kv_indices,
        "sm_scale": sm_scale,
        "causal": causal,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of paged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Prefill batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}, causal={causal}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Maximum number of pages
    max_pages = max_kv_len * batch_size * 2

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        max_pages,
        causal,
        device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Total query tokens: {inputs['total_q']}")
    print(f"Total KV indices: {inputs['num_kv_indices']}")

    # Run reference implementation from definition
    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=inputs["causal"],
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    # Compare outputs
    print("\nComparing outputs...")

    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    rel_diff = abs_diff / (torch.abs(fi_output_f32) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\nOutput tensor comparison:")
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_o_f32.flatten(), fi_output_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_o_f32 - fi_output_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    lse_rel_diff = lse_abs_diff / (torch.abs(fi_lse) + 1e-8)

    lse_max_abs_diff = lse_abs_diff.max().item()
    lse_max_rel_diff = lse_rel_diff.max().item()
    lse_mean_abs_diff = lse_abs_diff.mean().item()
    lse_mean_rel_diff = lse_rel_diff.mean().item()

    print(f"\nLSE comparison:")
    print(f"Max absolute difference: {lse_max_abs_diff:.6e}")
    print(f"Max relative difference: {lse_max_rel_diff:.6e}")
    print(f"Mean absolute difference: {lse_mean_abs_diff:.6e}")
    print(f"Mean relative difference: {lse_mean_rel_diff:.6e}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and LSE match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch GQA Paged Prefill Reference Implementation (from definition)")

    test_configs = [
        (1, 8, 16, True),
        (4, 16, 32, True),
        (8, 32, 64, True),
        (16, 64, 128, True),
    ]

    passed = 0
    total = len(test_configs)

    for batch_size, max_q_len, max_kv_len, causal in test_configs:
        try:
            if test_correctness(batch_size, max_q_len, max_kv_len, causal):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
