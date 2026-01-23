"""
Test MLA paged decode h16_ckv512_kpe64_ps64 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import numpy as np
import flashinfer
import torch

from test_utils import get_reference_run

# Load reference implementation from definition
run = get_reference_run("mla_paged_decode_h16_ckv512_kpe64_ps64")

# Constants from definition
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for MLA testing with page_size=64."""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    pages_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages_needed = pages_per_seq.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)
    kv_len_arr = seq_lens.clone()
    kv_last_page_len = ((seq_lens - 1) % PAGE_SIZE) + 1

    q_nope = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)

    return {
        "q_nope": q_nope, "q_pe": q_pe,
        "ckv_cache": ckv_cache, "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr, "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr, "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale, "qo_indptr": qo_indptr, "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=256, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing MLA (ps64) batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['kv_last_page_len'].cpu().numpy()}")

    print("\nRunning reference implementation from definition...")
    ref_o, ref_lse = run(
        inputs["q_nope"], inputs["q_pe"],
        inputs["ckv_cache"], inputs["kpe_cache"],
        inputs["kv_indptr"], inputs["kv_indices"],
        inputs["kv_last_page_len"], inputs["sm_scale"],
    )

    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")

    mla_wrapper.plan(
        qo_indptr=inputs["qo_indptr"], kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"], kv_len_arr=inputs["kv_len_arr"],
        num_heads=NUM_QO_HEADS, head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE,
        page_size=PAGE_SIZE, causal=False, sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = mla_wrapper.run(
        inputs["q_nope"], inputs["q_pe"], inputs["ckv_cache"], inputs["kpe_cache"], return_lse=True
    )

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
    print("Testing Batch MLA Paged Decode Reference Implementation (page_size=64, from definition)")
    test_configs = [(1, 64), (4, 128), (8, 256), (16, 512)]
    passed = sum(1 for cfg in test_configs if test_correctness(*cfg))
    print(f"\n{'='*60}\nSummary: {passed}/{len(test_configs)} tests passed\n{'='*60}")


if __name__ == "__main__":
    main()
