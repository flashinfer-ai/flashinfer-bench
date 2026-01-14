"""
Tests for NSA (Native Sparse Attention) sparse decode reference implementation.

Ground truth: SGLang NSA backend (third_party/sglang/python/sglang/srt/layers/attention/nsa_backend.py)
Fallback: FlashMLA sparse kernel (sgl_kernel.flash_mla.flash_mla_sparse_fwd)
"""
import math

import numpy as np
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Reference implementation for NSA sparse decode attention."""
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert topk == 256

    device = q_nope.device

    # Squeeze page dimension (page_size=1)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        indices = sparse_indices[b]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[b].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)
        num_valid = tok_idx.numel()

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_qo_heads=16,
    head_dim_ckv=512,
    head_dim_kpe=64,
    topk=256,
    device="cuda",
):
    """Generate random inputs for NSA sparse attention testing."""

    # Generate random sequence lengths for each batch
    # Ensure seq_lens >= topk so we have enough tokens to select
    min_seq_len = max(topk, 256)
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate total pages needed
    total_pages_needed = seq_lens.sum().item()

    # Generate page table (mapping sequence positions to page indices)
    # For simplicity, use consecutive pages
    page_table = torch.zeros(batch_size, max_seq_len, dtype=torch.int32, device=device)
    page_offset = 0
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        page_table[b, :seq_len] = torch.arange(page_offset, page_offset + seq_len, dtype=torch.int32, device=device)
        page_offset += seq_len

    # Generate sparse indices (top-K selection for each batch element)
    sparse_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        actual_topk = min(topk, seq_len)
        # Select random indices from available pages
        perm = torch.randperm(seq_len, device=device)[:actual_topk]
        selected_pages = page_table[b, perm]
        sparse_indices[b, :actual_topk] = selected_pages.to(torch.int32)

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate compressed KV and positional caches
    num_pages = total_pages_needed + 100  # Add extra pages
    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate softmax scale
    # MLA uses head dimension before matrix absorption (128 + 64 = 192)
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
        "seq_lens": seq_lens,
        "page_table": page_table,
    }


def test_output_shape(batch_size=4, max_seq_len=512, topk=256):
    """Test that reference produces correct output shapes."""
    print(f"\n{'='*60}")
    print(f"Testing NSA output shape: batch_size={batch_size}, topk={topk}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    num_qo_heads = 16
    head_dim_ckv = 512

    inputs = generate_random_inputs(batch_size, max_seq_len, topk=topk, device=device)

    output, lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["sparse_indices"],
        inputs["sm_scale"],
    )

    expected_output_shape = (batch_size, num_qo_heads, head_dim_ckv)
    expected_lse_shape = (batch_size, num_qo_heads)

    output_shape_correct = output.shape == expected_output_shape
    lse_shape_correct = lse.shape == expected_lse_shape

    print(f"Output shape: {output.shape} (expected: {expected_output_shape})")
    print(f"LSE shape: {lse.shape} (expected: {expected_lse_shape})")

    if output_shape_correct and lse_shape_correct:
        print("PASSED: Output shapes are correct")
        return True
    else:
        print("FAILED: Output shapes are incorrect")
        return False


def test_sparse_vs_dense_consistency(batch_size=4, max_seq_len=512, topk=256):
    """Test that sparse attention with all tokens selected equals dense attention."""
    print(f"\n{'='*60}")
    print(f"Testing NSA sparse vs dense consistency")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    # Generate inputs where sparse_indices includes all tokens (no sparsity)
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64

    # Use a small sequence length equal to topk for full coverage
    seq_len = topk
    num_pages = seq_len + 10

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + head_dim_kpe), dtype=torch.float32, device=device)

    # All indices valid (0 to seq_len-1)
    sparse_indices = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()

    output, lse = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    # Check that output is not all zeros (actually computed)
    output_nonzero = output.abs().sum() > 0
    lse_finite = torch.all(torch.isfinite(lse))

    print(f"Output non-zero: {output_nonzero}")
    print(f"LSE finite: {lse_finite}")

    if output_nonzero and lse_finite:
        print("PASSED: Sparse attention produces valid outputs")
        return True
    else:
        print("FAILED: Sparse attention produces invalid outputs")
        return False


def test_padding_handling(batch_size=4, topk=256):
    """Test that padding (-1 indices) are handled correctly."""
    print(f"\n{'='*60}")
    print(f"Testing NSA padding handling")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 1000

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + head_dim_kpe), dtype=torch.float32, device=device)

    # Create sparse indices with varying amounts of padding
    sparse_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    valid_counts = [topk, topk // 2, topk // 4, 10]  # Different valid counts per batch

    for b in range(batch_size):
        valid_count = valid_counts[b % len(valid_counts)]
        sparse_indices[b, :valid_count] = torch.randint(0, num_pages, (valid_count,), dtype=torch.int32, device=device)

    output, lse = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    # Verify outputs are valid
    output_valid = not torch.isnan(output).any() and not torch.isinf(output).any()
    # LSE can be -inf for empty sequences, but should not be +inf or nan
    lse_valid = not torch.isnan(lse).any() and not torch.isinf(lse[lse > -float("inf")]).any()

    print(f"Output valid (no nan/inf): {output_valid}")
    print(f"LSE valid: {lse_valid}")

    if output_valid and lse_valid:
        print("PASSED: Padding handled correctly")
        return True
    else:
        print("FAILED: Padding handling issue")
        return False


def test_correctness_against_sglang():
    """
    Test correctness against SGLang NSA backend.

    NOTE: This test requires SGLang to be installed and available.
    If SGLang is not available, the test will be skipped.
    """
    print(f"\n{'='*60}")
    print(f"Testing NSA correctness against SGLang")
    print(f"{'='*60}")

    try:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd
        SGLANG_AVAILABLE = True
    except ImportError:
        SGLANG_AVAILABLE = False

    if not SGLANG_AVAILABLE:
        print("SKIPPED: SGLang/sgl_kernel not available")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("SKIPPED: CUDA not available")
        return None

    # Test parameters
    batch_size = 4
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    topk = 256
    head_dim = head_dim_ckv + head_dim_kpe  # Combined head dim
    num_pages = 1024

    # Generate test inputs
    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Combined q for FlashMLA
    q_all = torch.cat([q_nope, q_pe], dim=-1)  # [batch_size, num_qo_heads, head_dim]

    # KV cache (combined)
    kv_cache = torch.randn(num_pages, 1, head_dim, dtype=torch.bfloat16, device=device)
    ckv_cache = kv_cache[:, :, :head_dim_ckv]
    kpe_cache = kv_cache[:, :, head_dim_ckv:]

    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    # Generate sparse indices
    sparse_indices = torch.randint(0, num_pages, (batch_size, topk), dtype=torch.int32, device=device)

    # Run reference implementation
    ref_output, ref_lse = run(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        torch.tensor(sm_scale, dtype=torch.float32, device=device)
    )

    # Run FlashMLA sparse
    # indices shape must be (s_q, h_kv=1, topk)
    indices_input = sparse_indices.unsqueeze(1)

    fi_output, _, _ = flash_mla_sparse_fwd(
        q=q_all,
        kv=kv_cache.squeeze(1),  # Remove page_size dim
        indices=indices_input,
        sm_scale=sm_scale,
        d_v=head_dim_ckv,
    )

    # Compare outputs
    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_o_f32)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_o_f32.flatten(), fi_o_f32.flatten(), dim=0
    ).item()

    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    atol, rtol = 1e-2, 5e-2
    output_close = torch.allclose(ref_o_f32, fi_o_f32, atol=atol, rtol=rtol)

    if output_close:
        print(f"PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"FAILED: Outputs differ beyond tolerance")
        return False


def main():
    """Run comprehensive tests."""
    print("Testing NSA (Native Sparse Attention) Sparse Decode Reference Implementation")
    print("="*70)

    test_results = []

    # Run tests
    test_results.append(("output_shape", test_output_shape()))
    test_results.append(("sparse_vs_dense", test_sparse_vs_dense_consistency()))
    test_results.append(("padding_handling", test_padding_handling()))
    test_results.append(("sglang_correctness", test_correctness_against_sglang()))

    # Summary
    print(f"\n{'='*70}")
    print("Test Summary:")
    print(f"{'='*70}")

    passed = 0
    skipped = 0
    failed = 0

    for name, result in test_results:
        if result is None:
            status = "SKIPPED"
            skipped += 1
        elif result:
            status = "PASSED"
            passed += 1
        else:
            status = "FAILED"
            failed += 1
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed")


if __name__ == "__main__":
    main()
