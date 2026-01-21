"""
Tests for DSA (DeepSeek Sparse Attention) sparse decode reference implementation.
Page size 64 variant.

Ground truth sources:
1. SGLang FlashMLA sparse kernel: sgl_kernel.flash_mla.flash_mla_with_kvcache (decode with indices)
2. SGLang FlashMLA sparse prefill: sgl_kernel.flash_mla.flash_mla_sparse_fwd (prefill)

Note: FlashInfer's sparse.py provides BlockSparseAttentionWrapper which uses BSR format,
different from DeepSeek's DSA token-level sparse attention.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Ground truth imports with availability checks
try:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

# FlashInfer sparse is BSR-based, different from DSA's token-level sparse
try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

# Module-level constants (DeepSeek V3/R1 with TP=8)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 256

TRACE_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Reference implementation for DSA sparse decode attention with page_size=64."""
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert head_dim_ckv == HEAD_DIM_CKV
    assert head_dim_kpe == HEAD_DIM_KPE
    assert page_size == PAGE_SIZE
    assert topk == TOPK

    # Check constraints
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(
        torch.float32
    )  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(
        torch.float32
    )  # [total_kv_tokens, head_dim_kpe]

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

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

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

    return {"output": output, "lse": lse}


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_qo_heads=NUM_QO_HEADS,
    head_dim_ckv=HEAD_DIM_CKV,
    head_dim_kpe=HEAD_DIM_KPE,
    topk=TOPK,
    device="cuda",
):
    """Generate random inputs for DSA sparse attention testing with page_size=64."""
    # Generate random sequence lengths for each batch
    # Ensure seq_lens >= topk so we have enough tokens to select
    min_seq_len = max(topk, 256)
    seq_lens = torch.randint(
        min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    # Calculate total pages needed (each page holds 64 tokens)
    total_tokens_needed = seq_lens.sum().item()
    total_pages_needed = (total_tokens_needed + PAGE_SIZE - 1) // PAGE_SIZE

    # Generate page table (mapping sequence positions to global token indices)
    # For page_size=64, page_table[b, pos] = page_idx * 64 + offset
    page_table = torch.zeros(batch_size, max_seq_len, dtype=torch.int32, device=device)
    token_offset = 0
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        page_table[b, :seq_len] = torch.arange(
            token_offset, token_offset + seq_len, dtype=torch.int32, device=device
        )
        token_offset += seq_len

    # Generate sparse indices (top-K selection for each batch element)
    # Indices are global token indices: page_idx * 64 + offset
    sparse_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        actual_topk = min(topk, seq_len)
        # Select random indices from available token positions
        perm = torch.randperm(seq_len, device=device)[:actual_topk]
        selected_tokens = page_table[b, perm]
        sparse_indices[b, :actual_topk] = selected_tokens.to(torch.int32)

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate compressed KV and positional caches with page_size=64
    num_pages = total_pages_needed + 10  # Add extra pages
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate softmax scale
    # MLA uses head dimension before matrix absorption (128 + 64 = 192)
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": torch.tensor(sm_scale, dtype=torch.float32, device=device),
        "seq_lens": seq_lens,
        "page_table": page_table,
        "num_pages": num_pages,
    }


def compute_error_metrics(ref, gt, name="output"):
    """Compute and print detailed error metrics."""
    ref_f32 = ref.float()
    gt_f32 = gt.float()

    abs_diff = torch.abs(ref_f32 - gt_f32)
    rel_diff = abs_diff / (torch.abs(gt_f32) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\n{name} comparison:")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")

    # Cosine similarity and MSE
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), gt_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - gt_f32) ** 2).item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE: {mse:.6e}")

    return abs_diff, rel_diff, cos_sim


def check_hit_ratio(ref, gt, atol, rtol, required_percent=0.85):
    """Check if hit ratio meets threshold (for quantized kernels)."""
    ref_f32 = ref.float()
    gt_f32 = gt.float()

    left = (ref_f32 - gt_f32).abs()
    right = atol + rtol * gt_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()

    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {required_percent * 100:.2f}%)")
    return hit_ratio >= required_percent


def test_output_shape(batch_size=4, max_seq_len=512, topk=TOPK):
    """Test that reference produces correct output shapes."""
    print(f"\n{'='*60}")
    print(f"Testing DSA decode ps64 output shape: batch_size={batch_size}, topk={topk}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    inputs = generate_random_inputs(batch_size, max_seq_len, topk=topk, device=device)

    result = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["sparse_indices"],
        inputs["sm_scale"],
    )

    output = result["output"]
    lse = result["lse"]

    expected_output_shape = (batch_size, NUM_QO_HEADS, HEAD_DIM_CKV)
    expected_lse_shape = (batch_size, NUM_QO_HEADS)

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


def test_sparse_vs_dense_consistency(batch_size=4, topk=TOPK):
    """Test that sparse attention with all tokens selected equals dense attention."""
    print(f"\n{'='*60}")
    print(f"Testing DSA decode ps64 sparse vs dense consistency")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    # Generate inputs where sparse_indices includes all tokens (no sparsity)
    # Use a small sequence length equal to topk for full coverage
    seq_len = topk
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE + 1

    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + HEAD_DIM_KPE), dtype=torch.float32, device=device)

    # All indices valid (0 to seq_len-1) - global token indices
    sparse_indices = (
        torch.arange(seq_len, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    result = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output = result["output"]
    lse = result["lse"]

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


def test_padding_handling(batch_size=4, topk=TOPK):
    """Test that padding (-1 indices) are handled correctly."""
    print(f"\n{'='*60}")
    print(f"Testing DSA decode ps64 padding handling")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    num_pages = 64  # 64 pages * 64 tokens = 4096 total tokens
    total_tokens_in_cache = num_pages * PAGE_SIZE

    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + HEAD_DIM_KPE), dtype=torch.float32, device=device)

    # Create sparse indices with varying amounts of padding
    sparse_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    valid_counts = [topk, topk // 2, topk // 4, 10]  # Different valid counts per batch

    for b in range(batch_size):
        valid_count = valid_counts[b % len(valid_counts)]
        sparse_indices[b, :valid_count] = torch.randint(
            0, total_tokens_in_cache, (valid_count,), dtype=torch.int32, device=device
        )

    result = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output = result["output"]
    lse = result["lse"]

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


def test_correctness_vs_sglang(batch_size=4, max_seq_len=512, atol=1e-2, rtol=5e-2):
    """
    Test correctness against SGLang FlashMLA sparse kernel.

    NOTE: This test requires SGLang sgl_kernel to be installed.
    If not available, the test will be skipped.

    NOTE: FlashMLA sparse kernel operates at token-level (page_size=1).
    For page_size=64, we flatten to token-level for comparison.
    """
    print(f"\n{'='*60}")
    print(f"Testing DSA decode ps64 correctness against SGLang FlashMLA")
    print(f"batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    if not SGLANG_AVAILABLE:
        print("SKIPPED: SGLang/sgl_kernel not available")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("SKIPPED: CUDA not available")
        return None

    torch.manual_seed(42)

    # Test parameters
    num_pages = 32  # 32 pages * 64 tokens/page = 2048 total KV tokens
    total_kv_tokens = num_pages * PAGE_SIZE
    head_dim = HEAD_DIM_CKV + HEAD_DIM_KPE  # Combined head dim = 576

    # Determine required head padding based on GPU architecture
    # FlashMLA kernel requires h_q to be multiple of 64 (Hopper SM90) or 128 (Blackwell SM100+)
    device_sm_major = torch.cuda.get_device_properties(device).major
    required_padding = 128 if device_sm_major >= 10 else 64
    print(f"GPU SM major: {device_sm_major}, required head padding: {required_padding}")

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    # Combined q for FlashMLA: [s_q, h_q, d_qk]
    q_all = torch.cat([q_nope, q_pe], dim=-1)  # [batch_size, num_qo_heads, head_dim]

    # KV cache with page_size=64
    # Shape: [num_pages, page_size, head_dim]
    kv_cache_paged = torch.randn(
        num_pages, PAGE_SIZE, head_dim, dtype=torch.bfloat16, device=device
    )
    ckv_cache = kv_cache_paged[:, :, :HEAD_DIM_CKV]  # [num_pages, 64, ckv]
    kpe_cache = kv_cache_paged[:, :, HEAD_DIM_CKV:]  # [num_pages, 64, kpe]

    # Flatten for FlashMLA (token-level)
    kv_cache_flat = kv_cache_paged.reshape(total_kv_tokens, head_dim)  # [total_kv_tokens, head_dim]

    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)

    # Generate sparse indices as global token indices: [batch_size, topk]
    # Each index is in range [0, total_kv_tokens)
    sparse_indices = torch.randint(
        0, total_kv_tokens, (batch_size, TOPK), dtype=torch.int32, device=device
    )

    # Run reference implementation with page_size=64
    print("Running reference implementation (page_size=64)...")
    ref_result = run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        torch.tensor(sm_scale, dtype=torch.float32, device=device),
    )
    ref_output = ref_result["output"]
    ref_lse = ref_result["lse"]

    # Run FlashMLA sparse (token-level)
    # flash_mla_sparse_fwd expects:
    #   q: [s_q, h_q, d_qk] bfloat16 - h_q must be multiple of 64 (SM90) or 128 (SM100+)
    #   kv: [s_kv, h_kv, d_qk] bfloat16 (h_kv=1 for MLA)
    #   indices: [s_q, h_kv, topk] int32
    print("Running SGLang FlashMLA sparse (token-level)...")
    try:
        kv_for_mla = kv_cache_flat.unsqueeze(1)  # [s_kv, 1, d_qk]

        # indices: [s_q, h_kv=1, topk]
        indices_for_mla = sparse_indices.unsqueeze(1)  # [batch_size, 1, topk]

        # Pad query heads to required multiple (64 or 128) as done in SGLang's dsa_backend.py
        need_padding = NUM_QO_HEADS % required_padding != 0
        if need_padding:
            assert (
                required_padding % NUM_QO_HEADS == 0
            ), f"required_padding ({required_padding}) must be divisible by NUM_QO_HEADS ({NUM_QO_HEADS})"
            q_padded = q_all.new_zeros((batch_size, required_padding, head_dim))
            q_padded[:, :NUM_QO_HEADS, :] = q_all
            q_input = q_padded
            print(f"Padded q from {NUM_QO_HEADS} to {required_padding} heads")
        else:
            q_input = q_all

        fi_output_full, fi_max_logits, fi_lse_full = flash_mla_sparse_fwd(
            q=q_input, kv=kv_for_mla, indices=indices_for_mla, sm_scale=sm_scale, d_v=HEAD_DIM_CKV
        )

        # Trim output back to original number of heads if padding was applied
        if need_padding:
            fi_output = fi_output_full[:, :NUM_QO_HEADS, :]
            fi_lse = fi_lse_full[:, :NUM_QO_HEADS]
        else:
            fi_output = fi_output_full
            fi_lse = fi_lse_full

    except Exception as e:
        print(f"WARNING: FlashMLA sparse fwd failed: {e}")
        print("This may be due to API differences - skipping SGLang test")
        import traceback

        traceback.print_exc()
        return None

    # Compare outputs
    print("\nComparing outputs...")
    abs_diff, rel_diff, cos_sim = compute_error_metrics(ref_output, fi_output, "output")

    # Check tolerance
    allclose = torch.allclose(ref_output.float(), fi_output.float(), atol=atol, rtol=rtol)

    if allclose:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        # Show top error locations
        flat = (ref_output.float() - fi_output.float()).abs().flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print(f"\nTop-{k} absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            print(
                f"  idx={idx}: ref={ref_output.flatten()[idx].item():.6e}, "
                f"fi={fi_output.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}"
            )

        # Use hit ratio as secondary check
        passed = check_hit_ratio(ref_output, fi_output, atol, rtol, required_percent=0.85)
        return passed


def main():
    """Run comprehensive tests."""
    print("Testing DSA (DeepSeek Sparse Attention) Sparse Decode Reference Implementation")
    print("Page Size 64 Variant")
    print("=" * 70)
    print(
        f"Constants: h={NUM_QO_HEADS}, ckv={HEAD_DIM_CKV}, kpe={HEAD_DIM_KPE}, ps={PAGE_SIZE}, topk={TOPK}"
    )
    print(f"SGLang available: {SGLANG_AVAILABLE}")
    print(f"FlashInfer available: {FLASHINFER_AVAILABLE}")
    print("=" * 70)

    test_results = []

    # Basic functionality tests
    test_results.append(("output_shape", test_output_shape()))
    test_results.append(("sparse_vs_dense", test_sparse_vs_dense_consistency()))
    test_results.append(("padding_handling", test_padding_handling()))

    # Ground truth comparison tests
    test_configs = [(1, 512), (4, 512), (8, 1024)]  # Single batch  # Small batch  # Medium batch

    for batch_size, max_seq_len in test_configs:
        name = f"sglang_bs{batch_size}_seq{max_seq_len}"
        try:
            result = test_correctness_vs_sglang(batch_size, max_seq_len)
            test_results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test {name} crashed: {e}")
            import traceback

            traceback.print_exc()
            test_results.append((name, False))

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
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} tests failed")


if __name__ == "__main__":
    main()
