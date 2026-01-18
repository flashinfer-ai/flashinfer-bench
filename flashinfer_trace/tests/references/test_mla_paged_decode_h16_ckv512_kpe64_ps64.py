import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    ckv_cache_f32 = ckv_cache.to(torch.float32)
    kpe_cache_f32 = kpe_cache.to(torch.float32)

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        last_page_len = int(kv_last_page_len[b].item())

        if page_beg >= page_end:
            output[b].zero_()
            continue

        page_ids = kv_indices[page_beg:page_end].to(torch.long)
        num_pages_for_seq = page_ids.shape[0]

        # Calculate total tokens
        num_full_pages = num_pages_for_seq - 1
        L_tokens = num_full_pages * page_size + last_page_len

        if L_tokens <= 0:
            output[b].zero_()
            continue

        # Gather Kc and Kp from pages
        Kc = torch.zeros((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp = torch.zeros((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        token_idx = 0
        for p_idx, page_id in enumerate(page_ids):
            if p_idx < num_full_pages:
                Kc[token_idx : token_idx + page_size] = ckv_cache_f32[page_id]
                Kp[token_idx : token_idx + page_size] = kpe_cache_f32[page_id]
                token_idx += page_size
            else:
                Kc[token_idx : token_idx + last_page_len] = ckv_cache_f32[page_id, :last_page_len]
                Kp[token_idx : token_idx + last_page_len] = kpe_cache_f32[page_id, :last_page_len]
                token_idx += last_page_len

        qn = q_nope[b].to(torch.float32)
        qp = q_pe[b].to(torch.float32)

        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[b] = out.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_qo_heads=16,
    head_dim_ckv=512,
    head_dim_kpe=64,
    page_size=64,
    device="cuda",
):
    """Generate random inputs for MLA testing."""

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate pages needed for each sequence
    pages_per_seq = (seq_lens + page_size - 1) // page_size  # Ceiling division
    total_pages_needed = pages_per_seq.sum().item()

    # Generate kv_indptr based on pages per sequence
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    # Generate kv_indices (page indices for each sequence)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Calculate last_page_len for each sequence
    kv_last_page_len = ((seq_lens - 1) % page_size) + 1

    # kv_len_arr stores the actual sequence lengths
    kv_len_arr = seq_lens.clone()

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate compressed KV and positional caches
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    # MLA uses head dimension before matrix absorption (128 + 64 = 192)
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)
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
        "kv_last_page_len": kv_last_page_len,
        "kv_len_arr": kv_len_arr,
        "sm_scale": sm_scale,
        "qo_indptr": qo_indptr,
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=256, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing MLA batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Constants from kernel definition
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 64

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size, max_seq_len, num_qo_heads, head_dim_ckv, head_dim_kpe, page_size, device
    )

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['kv_last_page_len'].cpu().numpy()}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
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
        num_heads=num_qo_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,  # For decode, causal doesn't matter as each query has length 1
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

        if not output_close:
            flat_abs_diff = abs_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            top_errors, top_indices = torch.topk(flat_abs_diff, top_k)

            print(f"\nTop {top_k} output tensor error locations:")
            for i in range(top_k):
                idx = top_indices[i].item()
                batch_idx = idx // (num_qo_heads * head_dim_ckv)
                head_idx = (idx % (num_qo_heads * head_dim_ckv)) // head_dim_ckv
                dim_idx = idx % head_dim_ckv

                ref_val = ref_o_f32.flatten()[idx].item()
                fi_val = fi_output_f32.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}, {dim_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_errors[i].item():.6e}"
                )

        if not lse_close:
            flat_lse_diff = lse_abs_diff.flatten()
            top_k = min(5, flat_lse_diff.numel())
            top_lse_errors, top_lse_indices = torch.topk(flat_lse_diff, top_k)

            print(f"\nTop {top_k} LSE error locations:")
            for i in range(top_k):
                idx = top_lse_indices[i].item()
                batch_idx = idx // num_qo_heads
                head_idx = idx % num_qo_heads

                ref_val = ref_lse.flatten()[idx].item()
                fi_val = fi_lse.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_lse_errors[i].item():.6e}"
                )

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch MLA Paged Decode Reference Implementation (page_size=64)")

    test_configs = [(1, 64), (4, 128), (8, 256), (16, 512), (32, 1024)]

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
