"""
Test GQA paged decode h32_kv4_d128_ps64 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import flashinfer
import numpy as np
import torch
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics

# Load reference implementation from definition
run = get_reference_run("gqa_paged_decode_h32_kv4_d128_ps64")

# Constants from definition
NUM_QO_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for testing."""
    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate pages needed for each sequence
    pages_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE  # Ceiling division
    total_pages_needed = pages_per_seq.sum().item()

    # Generate kv_indptr based on pages per sequence
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    # Generate kv_indices (page indices for each sequence)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Calculate last_page_len for each sequence
    kv_last_page_len = ((seq_lens - 1) % PAGE_SIZE) + 1

    # Generate query tensor
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # Generate K and V caches
    num_pages = total_pages_needed + 100
    k_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # Generate attention parameters
    sm_scale = 1.0 / np.sqrt(HEAD_DIM)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale,
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=256, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Generate inputs
    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['kv_last_page_len'].cpu().numpy()}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation from definition (page_size=64 includes kv_last_page_len)
    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # Plan the attention computation
    decode_wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = decode_wrapper.run(
        inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True
    )

    # Compare outputs
    print("\nComparing outputs...")
    output_metrics = compare_tensors(ref_o, fi_output, atol=atol, rtol=rtol)
    print_comparison_metrics(output_metrics, tensor_name="Output tensor")

    lse_metrics = compare_tensors(ref_lse, fi_lse, atol=atol, rtol=rtol)
    print_comparison_metrics(lse_metrics, tensor_name="LSE tensor")

    all_close = output_metrics.all_close and lse_metrics.all_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and LSE match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch GQA Paged Decode Reference Implementation (page_size=64, from definition)")

    test_configs = [(1, 64), (4, 128), (8, 256), (16, 512)]

    passed = 0
    total = len(test_configs)

    for batch_size, max_seq_len in test_configs:
        try:
            if test_correctness(batch_size, max_seq_len):
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
