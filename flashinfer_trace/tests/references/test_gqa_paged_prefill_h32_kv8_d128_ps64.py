import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 64

    # Check constraints
    assert total_q == qo_indptr[-1].item()

    device = q.device
    batch_size = len_indptr - 1

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        last_page_len = int(kv_last_page_len[b].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        num_pages_for_seq = page_ids.shape[0]

        # Calculate total KV tokens
        num_full_pages = num_pages_for_seq - 1
        num_kv_tokens = num_full_pages * page_size + last_page_len

        # Gather K and V from pages
        k_batch = torch.zeros(
            (num_kv_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device
        )
        v_batch = torch.zeros(
            (num_kv_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device
        )

        token_idx = 0
        for p_idx, page_id in enumerate(page_ids):
            if p_idx < num_full_pages:
                k_batch[token_idx : token_idx + page_size] = k_cache_f32[page_id]
                v_batch[token_idx : token_idx + page_size] = v_cache_f32[page_id]
                token_idx += page_size
            else:
                k_batch[token_idx : token_idx + last_page_len] = k_cache_f32[
                    page_id, :last_page_len
                ]
                v_batch[token_idx : token_idx + last_page_len] = v_cache_f32[
                    page_id, :last_page_len
                ]
                token_idx += last_page_len

        q_batch = q_f32[q_start:q_end]
        num_q_tokens = q_batch.shape[0]

        # Delta for causal masking
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Apply causal mask
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            q_pos = q_batch[q_idx]

            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio

                q_head = q_pos[h]
                k_head = k_batch[:max_kv_idx, kv_head]
                v_head = v_batch[:max_kv_idx, kv_head]

                logits = torch.matmul(q_head, k_head.T)
                logits_scaled = logits * sm_scale

                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)
                out_head = torch.matmul(attn, v_head)
                output[global_q_idx, h] = out_head.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    max_pages,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    page_size=64,
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

    # Create qo_indptr
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    # Calculate pages needed for each sequence
    pages_per_seq = (kv_lens + page_size - 1) // page_size  # Ceiling division
    total_pages_needed = pages_per_seq.sum().item()

    # Create kv_indptr based on pages per sequence
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq.to(device), dim=0)

    # Generate page indices
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Calculate last_page_len for each sequence
    kv_last_page_len = ((kv_lens - 1) % page_size) + 1
    kv_last_page_len = kv_last_page_len.to(device)

    # Get total tokens
    total_q = qo_indptr[-1].item()

    # Generate KV cache (paged storage)
    k_cache = torch.randn(
        max_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        max_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Generate query tensor
    q = torch.randn(total_q, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # Convert causal to tensor
    causal = torch.tensor(causal, dtype=torch.bool, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "sm_scale": sm_scale,
        "causal": causal,
        "page_size": page_size,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=128, causal=True, atol=1e-2, rtol=5e-2):
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

    # Constants from kernel definition
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    page_size = 64

    # Maximum number of pages
    max_pages = (max_kv_len * batch_size * 2 + page_size - 1) // page_size + 100

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        max_pages,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_size,
        causal,
        device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Last page lengths: {inputs['kv_last_page_len'].cpu().numpy()}")
    print(f"Total query tokens: {inputs['total_q']}")
    print(f"Total pages: {inputs['kv_indices'].shape[0]}")
    print(f"Causal mode: {inputs['causal'].item()}")
    print(f"Page size: {inputs['page_size']}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # Combine k_cache and v_cache into paged_kv_cache format
    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    # Plan the attention computation
    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        causal=inputs["causal"].item(),
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

        if not output_close:
            flat_abs_diff = abs_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            top_errors, top_indices = torch.topk(flat_abs_diff, top_k)

            print(f"\nTop {top_k} output tensor error locations:")
            for i in range(top_k):
                idx = top_indices[i].item()
                q_idx = idx // (num_attention_heads * head_dim)
                head_idx = (idx % (num_attention_heads * head_dim)) // head_dim
                dim_idx = idx % head_dim

                ref_val = ref_o_f32.flatten()[idx].item()
                fi_val = fi_output_f32.flatten()[idx].item()

                print(
                    f"  [q_idx={q_idx}, head={head_idx}, dim={dim_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_errors[i].item():.6e}"
                )

        if not lse_close:
            flat_lse_diff = lse_abs_diff.flatten()
            top_k = min(5, flat_lse_diff.numel())
            top_lse_errors, top_lse_indices = torch.topk(flat_lse_diff, top_k)

            print(f"\nTop {top_k} LSE error locations:")
            for i in range(top_k):
                idx = top_lse_indices[i].item()
                q_idx = idx // num_attention_heads
                head_idx = idx % num_attention_heads

                ref_val = ref_lse.flatten()[idx].item()
                fi_val = fi_lse.flatten()[idx].item()

                print(
                    f"  [q_idx={q_idx}, head={head_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_lse_errors[i].item():.6e}"
                )

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch GQA Paged Prefill Reference Implementation (page_size=64)")

    test_configs = [(1, 16, 64, True), (4, 32, 128, True), (8, 64, 256, True), (16, 128, 512, True)]

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
