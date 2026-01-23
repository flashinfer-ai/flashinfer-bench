"""
Test GQA paged prefill h32_kv4_d128_ps64 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import math

import flashinfer
import torch

from test_utils import get_reference_run

# Load reference implementation from definition
run = get_reference_run("gqa_paged_prefill_causal_h32_kv4_d128_ps64")

# Constants from definition
NUM_QO_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64


def generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, causal=True, device="cuda"):
    """Generate random inputs for paged prefill testing with page_size=64."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        if causal:
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()
        else:
            kv_lens[i] = torch.randint(1, max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    # Calculate pages needed for each sequence
    pages_per_seq = (kv_lens + PAGE_SIZE - 1) // PAGE_SIZE
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq.to(device), dim=0)

    total_q = qo_indptr[-1].item()
    num_kv_pages = kv_indptr[-1].item()

    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_pages]
    kv_indices = torch.zeros(num_kv_pages, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        num_pages = pages_per_seq[i].item()
        kv_indices[idx : idx + num_pages] = all_page_ids[idx : idx + num_pages]
        idx += num_pages

    # Calculate last_page_len for each sequence
    last_page_len = ((kv_lens - 1) % PAGE_SIZE) + 1
    last_page_len = last_page_len.to(torch.int32).to(device)

    k_cache = torch.randn(max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q, "k_cache": k_cache, "v_cache": v_cache,
        "qo_indptr": qo_indptr, "kv_indptr": kv_indptr, "kv_indices": kv_indices,
        "last_page_len": last_page_len, "q_lens": q_lens, "kv_lens": kv_lens,
        "total_q": total_q, "sm_scale": sm_scale, "causal": causal,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=256, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of paged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing GQA Paged Prefill (ps64) batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    max_pages = (max_kv_len * batch_size * 2) // PAGE_SIZE + 100
    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, causal, device)

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['last_page_len'].cpu().numpy()}")

    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q"], inputs["k_cache"], inputs["v_cache"],
        inputs["qo_indptr"], inputs["kv_indptr"], inputs["kv_indices"],
        inputs["last_page_len"], inputs["sm_scale"],
    )

    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"], paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"], paged_kv_last_page_len=inputs["last_page_len"],
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim_qk=HEAD_DIM, head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE, causal=inputs["causal"], sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    print("\nComparing outputs...")
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(ref_o_f32.flatten(), fi_output_f32.flatten(), dim=0).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance")

    return all_close


def main():
    print("Testing Batch GQA Paged Prefill Reference Implementation (page_size=64, from definition)")
    test_configs = [(1, 16, 64, True), (4, 32, 128, True), (8, 64, 256, True)]
    passed = sum(1 for cfg in test_configs if test_correctness(*cfg))
    print(f"\n{'='*60}\nSummary: {passed}/{len(test_configs)} tests passed\n{'='*60}")


if __name__ == "__main__":
    main()
