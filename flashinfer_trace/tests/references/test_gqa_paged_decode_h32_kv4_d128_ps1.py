import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_start >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        # Pages are the token indices for page_size=1
        token_indices = kv_indices[page_start:page_end].to(torch.long)
        # Number of tokens is the number of pages for page_size=1
        num_tokens = token_indices.shape[0]

        if num_tokens == 0:
            output[b].zero_()
            continue

        # Get Q, K, V for this batch
        k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            # Find corresponding KV head for GQA
            kv_head = h // gqa_ratio

            q_head = q_batch[h]  # [head_dim]
            k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
            v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]

            logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
            logits_scaled = logits * sm_scale

            # Compute 2-base LSE
            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
            out_head = torch.matmul(attn, v_head)  # [head_dim]
            output[b, h] = out_head.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    page_size=1,
    device="cuda",
):
    """Generate random inputs for testing."""

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate total pages needed
    # Since page_size = 1, num_pages = total_tokens
    total_pages_needed = seq_lens.sum().item()

    # Generate kv_indptr based on sequence lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

    # Generate kv_indices (page indices for each sequence)
    # We'll use consecutive pages for simplicity
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # For page_size=1, last page always has 1 token
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Generate query tensor
    q = torch.randn(batch_size, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Generate K and V caches
    # Add some extra pages to simulate a real scenario
    num_pages = total_pages_needed + 100
    k_cache = torch.randn(
        num_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Generate attention parameters
    sm_scale = 1.0 / np.sqrt(head_dim)
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


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}, max_seq_len={max_seq_len}")
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

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_seq_len,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_size,
        device,
    )

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"  # Matches our cache layout
    )

    # Plan the attention computation
    decode_wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    # FlashInfer expects tuple of (k_cache, v_cache) for paged_kv_cache
    fi_output, fi_lse = decode_wrapper.run(
        inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True
    )

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
                batch_idx = idx // (num_attention_heads * head_dim)
                head_idx = (idx % (num_attention_heads * head_dim)) // head_dim
                dim_idx = idx % head_dim

                ref_val = ref_o_f32.flatten()[idx].item()
                fi_val = fi_output_f32.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}, {dim_idx}]: "
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
                batch_idx = idx // num_attention_heads
                head_idx = idx % num_attention_heads

                ref_val = ref_lse.flatten()[idx].item()
                fi_val = fi_lse.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_lse_errors[i].item():.6e}"
                )

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch GQA Paged Decode Reference Implementation")

    # Test different configurations
    test_configs = [
        # (batch_size, max_seq_len)
        (1, 16),  # Single batch
        (4, 32),  # Small batch
        (8, 64),  # Medium batch
        (16, 128),  # Large batch
    ]

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
