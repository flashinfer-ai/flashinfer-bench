"""
Test MLA paged prefill h16_ckv512_kpe64_ps64 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import flashinfer
import numpy as np
import torch
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics

# Load reference implementation from definition
run = get_reference_run("mla_paged_prefill_causal_h16_ckv512_kpe64_ps64")

# Constants from definition
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64


def generate_random_inputs(batch_size, max_q_len, max_kv_len, causal=True, device="cuda"):
    """Generate random inputs for MLA prefill testing with page_size=64."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        if causal:
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()
        else:
            kv_lens[i] = torch.randint(1, max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    pages_per_seq = (kv_lens + PAGE_SIZE - 1) // PAGE_SIZE
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq.to(device), dim=0)

    total_q = qo_indptr[-1].item()
    num_kv_pages = kv_indptr[-1].item()
    kv_len_arr = kv_lens.to(device)

    max_pages = num_kv_pages + 100
    kv_indices = torch.arange(num_kv_pages, dtype=torch.int32, device=device)
    last_page_len = ((kv_lens - 1) % PAGE_SIZE) + 1
    last_page_len = last_page_len.to(torch.int32).to(device)

    q_nope = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(max_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(max_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr,
        "last_page_len": last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "sm_scale": sm_scale,
        "causal": causal,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=256, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing MLA Paged Prefill (ps64) batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, causal, device)

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['last_page_len'].cpu().numpy()}")

    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["last_page_len"],
        inputs["sm_scale"],
    )

    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")

    mla_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"],
        kv_len_arr=inputs["kv_len_arr"],
        num_heads=NUM_QO_HEADS,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=PAGE_SIZE,
        causal=inputs["causal"],
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = mla_wrapper.run(
        inputs["q_nope"], inputs["q_pe"], inputs["ckv_cache"], inputs["kpe_cache"], return_lse=True
    )

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
    print(
        "Testing Batch MLA Paged Prefill Reference Implementation (page_size=64, from definition)"
    )
    test_configs = [(1, 16, 64, True), (4, 32, 128, True), (8, 64, 256, True)]
    passed = sum(1 for cfg in test_configs if test_correctness(*cfg))
    print(f"\n{'='*60}\nSummary: {passed}/{len(test_configs)} tests passed\n{'='*60}")


if __name__ == "__main__":
    main()
