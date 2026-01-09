import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        # Get Q, K, V for this batch
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
        k_batch = k_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, head_dim]
        v_batch = v_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, head_dim]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

        # Compute attention scores: Q @ K^T
        logits = torch.einsum("qhd,khd->qhk", q_batch, k_expanded) * sm_scale

        # For position q_idx, can attend to KV positions [0, min(q_idx + 1 + delta, num_kv_tokens))
        q_positions = torch.arange(num_q_tokens, device=device)  # [num_q_tokens]
        kv_positions = torch.arange(num_kv_tokens, device=device)  # [num_kv_tokens]

        # Apply causal mask
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float("-inf"))

        # Compute 2-base LSE
        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)  # [num_q_tokens, num_qo_heads, num_kv_tokens]
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_expanded)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    causal=True,
    device="cuda",
):
    """Generate random inputs for ragged prefill testing."""

    # Generate random query lengths for each batch element
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)

    # Generate random KV lengths for each batch element
    # For prefill, KV length is typically >= query length (includes previous context)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        # KV length should be at least as long as query length for causal attention
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    # Create indptr arrays
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    # Get total tokens
    total_q = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()

    # Generate tensors
    q = torch.randn(total_q, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # Convert causal to tensor
    causal = torch.tensor(causal, dtype=torch.bool, device=device)

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
        f"Testing GQA Ragged Prefill batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}, causal={causal}"
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

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        causal,
        device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Total query tokens: {inputs['total_q']}")
    print(f"Total KV tokens: {inputs['total_kv']}")
    print(f"Causal mode: {inputs['causal'].item()}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"  # Layout for K/V tensors
    )

    # Plan the attention computation
    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim_qk=head_dim,  # head dimension for query/key
        head_dim_vo=head_dim,  # head dimension for value/output (same as qk for standard attention)
        causal=inputs["causal"].item(),  # Use the randomly generated causal flag
        sm_scale=inputs["sm_scale"],  # Scale factor for softmax
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

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
    print("Testing Batch GQA Ragged Prefill Reference Implementation")

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
        (32, 128, 256, True),  # Very large batch, causal
        # (32, 128, 256, False),  # Very large batch, non-causal
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
