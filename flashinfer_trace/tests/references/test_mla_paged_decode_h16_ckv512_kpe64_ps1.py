"""
Test MLA paged decode h16_ckv512_kpe64_ps1 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import flashinfer
import numpy as np
import torch
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics

# Load reference implementation from definition
run = get_reference_run("mla_paged_decode_h16_ckv512_kpe64_ps1")

# Constants from definition
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for MLA testing."""
    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate total pages needed (page_size=1)
    total_pages_needed = seq_lens.sum().item()

    # Generate kv_indptr based on sequence lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

    # Generate kv_indices
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # kv_len_arr stores the actual sequence lengths
    kv_len_arr = seq_lens.clone()

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    # Generate compressed KV and positional caches
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    # MLA uses head dimension before matrix absorption (128 + 64 = 192)
    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # For decode, qo_indptr is just [0, 1, 2, ..., batch_size]
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr,
        "sm_scale": sm_scale,
        "qo_indptr": qo_indptr,
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing MLA batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Generate inputs
    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation from definition
    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")

    # Plan the attention computation
    mla_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"],
        kv_len_arr=inputs["kv_len_arr"],
        num_heads=NUM_QO_HEADS,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=PAGE_SIZE,
        causal=False,
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = mla_wrapper.run(
        inputs["q_nope"], inputs["q_pe"], inputs["ckv_cache"], inputs["kpe_cache"], return_lse=True
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
    print("Testing Batch MLA Paged Decode Reference Implementation (from definition)")

    test_configs = [(1, 16), (4, 32), (8, 64), (16, 128), (32, 256)]

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
