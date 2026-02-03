import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        kv_len = page_end - page_beg
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        tok_idx = pages[:kv_len].to(torch.long)
        Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        for i in range(q_len):
            qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
            qp = q_pe_batch[i]  # [num_heads, head_dim_kpe]

            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, kv_len]
            logits_scaled = logits * sm_scale

            # Apply causal mask
            prefix_len = kv_len - q_len  # Number of previously cached tokens
            query_abs_pos = prefix_len + i  # Absolute position of current query

            causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
            logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

            # Compute 2-base LSE
            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_heads, L_tokens]
            out = attn @ Kc  # [num_heads, head_dim_ckv]
            output[q_start + i] = out.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    num_qo_heads=16,
    head_dim_ckv=512,
    head_dim_kpe=64,
    page_size=1,
    causal=True,
    device="cuda",
):
    """Generate random inputs for MLA paged prefill testing."""

    # Generate random sequence lengths for each batch
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32, device=device)
    kv_lens = torch.randint(1, max_kv_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # For prefill, ensure kv_len >= q_len for causal attention
    for i in range(batch_size):
        kv_lens[i] = max(kv_lens[i], q_lens[i])

    total_q = q_lens.sum().item()

    # Calculate total pages needed (since page_size=1, num_pages = total_kv_tokens)
    total_pages_needed = kv_lens.sum().item()

    # Generate qo_indptr based on query lengths
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0)

    # Generate kv_indptr based on KV lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

    # Generate kv_indices (page indices for each sequence)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # kv_len_arr stores the actual KV sequence lengths
    kv_len_arr = kv_lens.clone()

    # Generate query tensors with Matrix Absorption dimensions
    q_nope = torch.randn(total_q, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(total_q, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate compressed KV and positional caches
    # Add some extra pages to simulate a real scenario
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    # MLA uses head dimension before matrix absorption
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # Convert causal to tensor
    causal = torch.tensor(causal, dtype=torch.bool, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr,
        "sm_scale": sm_scale,
        "causal": causal,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, causal=True, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA paged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}, causal={causal}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Constants from kernel definition
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        num_qo_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Total Q tokens: {inputs['total_q']}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")
    print(f"Causal: {inputs['causal'].item()}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # For paged prefill with Matrix Absorption, use BatchMLAPagedAttentionWrapper
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
        causal=inputs["causal"].item(),  # Causal masking configuration
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    flashinfer_o, flashinfer_lse = mla_wrapper.run(
        q_nope=inputs["q_nope"],
        q_pe=inputs["q_pe"],
        ckv_cache=inputs["ckv_cache"],
        kpe_cache=inputs["kpe_cache"],
        return_lse=True,
    )

    # Compare outputs
    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_o.shape}")
    print(f"FlashInfer output shape: {flashinfer_o.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")
    print(f"FlashInfer LSE shape: {flashinfer_lse.shape}")

    # Check numerical accuracy
    o_diff = torch.abs(ref_o - flashinfer_o)
    lse_diff = torch.abs(ref_lse - flashinfer_lse)

    print(f"\nOutput max diff: {o_diff.max().item():.6f}")
    print(f"Output mean diff: {o_diff.mean().item():.6f}")
    print(f"LSE max diff: {lse_diff.max().item():.6f}")
    print(f"LSE mean diff: {lse_diff.mean().item():.6f}")

    # Check if outputs match within tolerance
    output_close = torch.allclose(ref_o.float(), flashinfer_o.float(), atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, flashinfer_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and LSE match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        if not output_close:
            # Find indices with largest errors for debugging
            flat_abs_diff = o_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            top_errors, top_indices = torch.topk(flat_abs_diff, top_k)

            print(f"\nTop {top_k} output tensor error locations:")
            for i in range(top_k):
                idx = top_indices[i].item()
                # Convert flat index back to 3D indices
                _, num_qo_heads, head_dim_ckv = ref_o.shape
                batch_idx = idx // (num_qo_heads * head_dim_ckv)
                head_idx = (idx % (num_qo_heads * head_dim_ckv)) // head_dim_ckv
                dim_idx = idx % head_dim_ckv

                ref_val = ref_o.flatten()[idx].item()
                fi_val = flashinfer_o.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}, {dim_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_errors[i].item():.6e}"
                )

        if not lse_close:
            # Find LSE errors
            flat_lse_diff = lse_diff.flatten()
            top_k = min(5, flat_lse_diff.numel())
            top_lse_errors, top_lse_indices = torch.topk(flat_lse_diff, top_k)

            print(f"\nTop {top_k} LSE error locations:")
            for i in range(top_k):
                idx = top_lse_indices[i].item()
                _, num_qo_heads = ref_lse.shape
                batch_idx = idx // num_qo_heads
                head_idx = idx % num_qo_heads

                ref_val = ref_lse.flatten()[idx].item()
                fi_val = flashinfer_lse.flatten()[idx].item()

                print(
                    f"  [{batch_idx}, {head_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_lse_errors[i].item():.6e}"
                )

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Batch MLA Paged Prefill Reference Implementation")

    # Test different configurations
    test_configs = [
        # (batch_size, max_q_len, max_kv_len, causal)
        (1, 8, 16, True),  # Small causal
        # (1, 8, 16, False),  # Small non-causal
        (4, 16, 32, True),  # Medium causal
        # (4, 16, 32, False), # Medium non-causal
        (8, 32, 64, True),  # Large causal
        # (8, 32, 64, False), # Large non-causal
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
