"""
Test GQA ragged prefill h32_kv4_d128 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import math

import flashinfer
import torch
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics

# Load reference implementation from definition
run = get_reference_run("gqa_ragged_prefill_causal_h32_kv4_d128")

# Constants from definition
NUM_QO_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128


def generate_random_inputs(batch_size, max_q_len, max_kv_len, causal=True, device="cuda"):
    """Generate random inputs for ragged prefill testing."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        if causal:
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()
        else:
            kv_lens[i] = torch.randint(1, max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "total_kv": total_kv,
        "sm_scale": sm_scale,
        "causal": causal,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of ragged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Ragged Prefill batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, causal, device)

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")

    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["sm_scale"],
    )

    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=inputs["causal"],
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

    print("\nComparing outputs...")
    output_metrics = compare_tensors(ref_o, fi_output, atol=atol, rtol=rtol)
    print_comparison_metrics(output_metrics, tensor_name="Output tensor")

    lse_metrics = compare_tensors(ref_lse, fi_lse, atol=atol, rtol=rtol)
    print_comparison_metrics(lse_metrics, tensor_name="LSE tensor")

    all_close = output_metrics.all_close and lse_metrics.all_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance")

    return all_close


def main():
    print("Testing Batch GQA Ragged Prefill Reference Implementation (from definition)")
    test_configs = [(1, 8, 16, True), (4, 16, 32, True), (8, 32, 64, True), (16, 64, 128, True)]
    passed = sum(1 for cfg in test_configs if test_correctness(*cfg))
    print(f"\n{'='*60}\nSummary: {passed}/{len(test_configs)} tests passed\n{'='*60}")


if __name__ == "__main__":
    main()
