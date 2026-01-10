import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    # Flatten page dimension since page_size=1
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)

        # Number of KV tokens is equal to number of pages for page_size=1
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]

        # Get queries for this sequence
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
        num_q_tokens = q_batch.shape[0]

        # Delta for causal masking
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Apply causal mask
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            q_pos = q_batch[q_idx]  # [num_qo_heads, head_dim]

            for h in range(num_qo_heads):
                # Find corresponding KV head for GQA
                kv_head = h // gqa_ratio

                q_head = q_pos[h]  # [head_dim]
                k_head = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]
                v_head = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]

                logits = torch.matmul(q_head, k_head.T)  # [max_kv_idx]
                logits_scaled = logits * sm_scale

                # Compute 2-base LSE
                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)  # [max_kv_idx]
                out_head = torch.matmul(attn, v_head)  # [head_dim]
                output[global_q_idx, h] = out_head.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    max_pages,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    page_size=1,
    causal=True,
    device="cuda",
):
    """Generate random inputs for paged prefill testing."""

    # Generate random query lengths for each batch element
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)

    # Generate random KV lengths for each batch element
    # For prefill, KV length is typically >= query length (includes previous context)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        # KV length should be at least as long as query length for causal attention
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

    # Generate page indices (for page_size=1, we need num_kv_indices unique pages)
    # Simulate scattered memory allocation
    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]

    # Create kv_indices by assigning pages to each sequence
    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        kv_indices[idx : idx + seq_len] = all_page_ids[idx : idx + seq_len]
        idx += seq_len

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
        "page_size": page_size,
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

    # Constants from kernel definition
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    page_size = 1

    # Maximum number of pages (should be large enough to hold all KV tokens)
    max_pages = max_kv_len * batch_size * 2  # Extra buffer for scattered allocation

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
    print(f"Total query tokens: {inputs['total_q']}")
    print(f"Total KV indices: {inputs['num_kv_indices']}")
    print(f"Max page ID used: {inputs['kv_indices'].max().item()}")
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
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"  # Layout for K/V tensors
    )

    # Combine k_cache and v_cache into paged_kv_cache format that FlashInfer expects
    # FlashInfer expects shape [max_num_pages, 2, page_size, num_kv_heads, head_dim] for NHD layout
    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    # Plan the attention computation
    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["last_page_len"],
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

    # Convert to float32 for comparison
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    # Compute errors for output tensor
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

    # Compute cosine similarity and MSE for output tensor
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_o_f32.flatten(), fi_output_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_o_f32 - fi_output_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    # Compare LSE values
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

    # Check if outputs match within tolerance
    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and LSE match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        if not output_close:
            # Find indices with largest errors for debugging
            flat_abs_diff = abs_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            top_errors, top_indices = torch.topk(flat_abs_diff, top_k)

            print(f"\nTop {top_k} output tensor error locations:")
            for i in range(top_k):
                idx = top_indices[i].item()
                # Convert flat index back to 3D indices
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
            # Find LSE errors
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
    print("Testing Batch GQA Paged Prefill Reference Implementation")

    # Test different configurations
    test_configs = [
        # (batch_size, max_q_len, max_kv_len, causal)
        (1, 8, 16, True),  # Single batch, small, causal
        # (1, 8, 16, False),  # Single batch, small, non-causal
        (4, 16, 32, True),  # Small batch, causal
        # (4, 16, 32, False),  # Small batch, non-causal
        (8, 32, 64, True),  # Medium batch, causal
        # (8, 32, 64, False),  # Medium batch, non-causal
        (16, 64, 128, True),  # Large batch, causal
        # (16, 64, 128, False),  # Large batch, non-causal
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
