import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, qk_dim = q.shape
    total_kv, num_kv_heads, vo_dim = v.shape
    len_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert num_kv_heads == 16
    assert qk_dim == 192
    assert vo_dim == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((total_q, num_qo_heads, vo_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, qk_dim]
        k_batch = k_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, qk_dim]
        v_batch = v_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, vo_dim]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        # num_kv_heads == num_qo_heads for absorbed MLA, no GQA expansion needed
        logits = torch.einsum("qhd,khd->qhk", q_batch, k_batch) * sm_scale

        # Apply causal mask
        q_positions = torch.arange(num_q_tokens, device=device)
        kv_positions = torch.arange(num_kv_tokens, device=device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float("-inf"))

        # Compute 2-base LSE
        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)  # [num_q_tokens, num_qo_heads, num_kv_tokens]
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_batch)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    num_qo_heads=16,
    num_kv_heads=16,
    qk_dim=192,
    vo_dim=128,
    causal=True,
    device="cuda",
):
    """Generate random inputs for MLA ragged prefill testing."""

    # Generate random sequence lengths for each batch
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)

    # For prefill with causal mask, kv_len >= q_len
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    # Create indptr arrays
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())

    # Generate tensors
    q = torch.randn(total_q, num_qo_heads, qk_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, num_kv_heads, qk_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, num_kv_heads, vo_dim, dtype=torch.bfloat16, device=device)

    # sm_scale: 1/sqrt(qk_dim) = 1/sqrt(192)
    sm_scale = torch.tensor(1.0 / math.sqrt(qk_dim), dtype=torch.float32, device=device)

    causal_tensor = torch.tensor(causal, dtype=torch.bool, device=device)

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
        "causal": causal_tensor,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA ragged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing MLA Ragged Prefill batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}, causal={causal}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    # Constants from kernel definition
    num_qo_heads = 16
    num_kv_heads = 16
    qk_dim = 192
    vo_dim = 128

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        num_qo_heads,
        num_kv_heads,
        qk_dim,
        vo_dim,
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
    print("\nSetting up FlashInfer BatchPrefillWithRaggedKVCacheWrapper...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=qk_dim,
        head_dim_vo=vo_dim,
        causal=inputs["causal"].item(),
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

    # Compare outputs
    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_o.shape}")
    print(f"FlashInfer output shape: {fi_output.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")
    print(f"FlashInfer LSE shape: {fi_lse.shape}")

    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    rel_diff = abs_diff / (torch.abs(fi_output_f32) + 1e-8)

    print(f"\nOutput tensor comparison:")
    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"Max relative difference: {rel_diff.max().item():.6e}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6e}")
    print(f"Mean relative difference: {rel_diff.mean().item():.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    lse_rel_diff = lse_abs_diff / (torch.abs(fi_lse) + 1e-8)

    print(f"\nLSE comparison:")
    print(f"Max absolute difference: {lse_abs_diff.max().item():.6e}")
    print(f"Max relative difference: {lse_rel_diff.max().item():.6e}")
    print(f"Mean absolute difference: {lse_abs_diff.mean().item():.6e}")
    print(f"Mean relative difference: {lse_rel_diff.mean().item():.6e}")

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
                q_idx = idx // (num_qo_heads * vo_dim)
                head_idx = (idx % (num_qo_heads * vo_dim)) // vo_dim
                dim_idx = idx % vo_dim

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
                q_idx = idx // num_qo_heads
                head_idx = idx % num_qo_heads

                ref_val = ref_lse.flatten()[idx].item()
                fi_val = fi_lse.flatten()[idx].item()

                print(
                    f"  [q_idx={q_idx}, head={head_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_lse_errors[i].item():.6e}"
                )

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing MLA Ragged Prefill Reference Implementation")
    print("Definition: mla_ragged_prefill_causal_h16_qk192_vo128")
    print("API: flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper")
    print(f"Constants: num_qo_heads=16, num_kv_heads=16, qk_dim=192, vo_dim=128")

    test_configs = [
        # (batch_size, max_q_len, max_kv_len, causal)
        (1, 8, 16, True),  # Small causal
        (4, 16, 32, True),  # Medium causal
        (8, 32, 64, True),  # Large causal
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
