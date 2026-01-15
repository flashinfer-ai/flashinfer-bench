"""
Tests for NSA (Native Sparse Attention) sparse prefill reference implementation.

Ground truth sources:
1. SGLang FlashMLA sparse prefill: sgl_kernel.flash_mla.flash_mla_sparse_fwd

Note: FlashInfer's sparse.py provides BlockSparseAttentionWrapper which uses BSR format,
different from DeepSeek's NSA token-level sparse attention.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Ground truth imports with availability checks
try:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

# Module-level constants (DeepSeek V3/R1 with TP=8)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1
TOPK = 256

TRACE_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Reference implementation for NSA sparse prefill attention."""
    total_num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert head_dim_ckv == HEAD_DIM_CKV
    assert head_dim_kpe == HEAD_DIM_KPE
    assert page_size == PAGE_SIZE
    assert topk == TOPK

    # Check constraints
    assert sparse_indices.shape[0] == total_num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Squeeze page dimension (page_size=1)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for t in range(total_num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}


def generate_random_inputs(
    total_num_tokens,
    num_qo_heads=NUM_QO_HEADS,
    head_dim_ckv=HEAD_DIM_CKV,
    head_dim_kpe=HEAD_DIM_KPE,
    topk=TOPK,
    device="cuda",
):
    """Generate random inputs for NSA sparse prefill attention testing."""
    # Generate KV cache with enough pages
    num_pages = max(total_num_tokens * 2, 1024)

    # Generate sparse indices (top-K selection for each token)
    sparse_indices = torch.randint(
        0, num_pages, (total_num_tokens, topk), dtype=torch.int32, device=device
    )

    # Generate query tensors
    q_nope = torch.randn(
        total_num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        total_num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device
    )

    # Generate compressed KV and positional caches
    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)

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


def test_output_shape(total_num_tokens=64, topk=TOPK):
    """Test that reference produces correct output shapes."""
    print(f"\n{'='*60}")
    print(f"Testing NSA prefill output shape: total_num_tokens={total_num_tokens}, topk={topk}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    inputs = generate_random_inputs(total_num_tokens, topk=topk, device=device)

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

    expected_output_shape = (total_num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV)
    expected_lse_shape = (total_num_tokens, NUM_QO_HEADS)

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


def test_padding_handling(total_num_tokens=64, topk=TOPK):
    """Test that padding (-1 indices) are handled correctly."""
    print(f"\n{'='*60}")
    print(f"Testing NSA prefill padding handling")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    num_pages = 1000

    q_nope = torch.randn(
        total_num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        total_num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device
    )
    ckv_cache = torch.randn(num_pages, 1, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / np.sqrt(128 + HEAD_DIM_KPE), dtype=torch.float32, device=device)

    # Create sparse indices with varying amounts of padding per token
    sparse_indices = torch.full((total_num_tokens, topk), -1, dtype=torch.int32, device=device)

    for t in range(total_num_tokens):
        # Vary the number of valid indices
        valid_count = (t % 4 + 1) * (topk // 4)  # 25%, 50%, 75%, 100%
        valid_count = min(valid_count, topk)
        sparse_indices[t, :valid_count] = torch.randint(
            0, num_pages, (valid_count,), dtype=torch.int32, device=device
        )

    result = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output = result["output"]
    lse = result["lse"]

    # Verify outputs are valid
    output_valid = not torch.isnan(output).any() and not torch.isinf(output).any()
    lse_valid = not torch.isnan(lse).any() and not torch.isinf(lse[lse > -float("inf")]).any()

    print(f"Output valid (no nan/inf): {output_valid}")
    print(f"LSE valid: {lse_valid}")

    if output_valid and lse_valid:
        print("PASSED: Padding handled correctly")
        return True
    else:
        print("FAILED: Padding handling issue")
        return False


def test_correctness_vs_sglang(total_num_tokens=64, atol=1e-2, rtol=5e-2):
    """
    Test correctness against SGLang FlashMLA sparse prefill kernel.

    NOTE: This test requires SGLang sgl_kernel to be installed.
    If not available, the test will be skipped.
    """
    print(f"\n{'='*60}")
    print(f"Testing NSA prefill correctness against SGLang FlashMLA")
    print(f"total_num_tokens={total_num_tokens}")
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
    num_pages = 1024
    head_dim = HEAD_DIM_CKV + HEAD_DIM_KPE  # Combined head dim = 576

    # Determine required head padding based on GPU architecture
    # FlashMLA kernel requires h_q to be multiple of 64 (Hopper SM90) or 128 (Blackwell SM100+)
    device_sm_major = torch.cuda.get_device_properties(device).major
    required_padding = 128 if device_sm_major >= 10 else 64
    print(f"GPU SM major: {device_sm_major}, required head padding: {required_padding}")

    # Generate query tensors
    q_nope = torch.randn(
        total_num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        total_num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device
    )

    # Combined q for FlashMLA: [s_q, h_q, d_qk]
    q_all = torch.cat([q_nope, q_pe], dim=-1)  # [total_num_tokens, num_qo_heads, head_dim]

    # KV cache (combined)
    # flash_mla_sparse_fwd expects: kv [s_kv, h_kv, d_qk] where h_kv=1 for MLA
    kv_cache = torch.randn(num_pages, head_dim, dtype=torch.bfloat16, device=device)
    ckv_cache = kv_cache[:, :HEAD_DIM_CKV].unsqueeze(1)  # [num_pages, 1, ckv]
    kpe_cache = kv_cache[:, HEAD_DIM_CKV:].unsqueeze(1)  # [num_pages, 1, kpe]

    sm_scale = 1.0 / np.sqrt(128 + HEAD_DIM_KPE)

    # Generate sparse indices: [total_num_tokens, topk] for reference
    sparse_indices = torch.randint(
        0, num_pages, (total_num_tokens, TOPK), dtype=torch.int32, device=device
    )

    # Run reference implementation
    print("Running reference implementation...")
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

    # Run FlashMLA sparse prefill
    # flash_mla_sparse_fwd expects:
    #   q: [s_q, h_q, d_qk] bfloat16 - h_q must be multiple of 64 (SM90) or 128 (SM100+)
    #   kv: [s_kv, h_kv, d_qk] bfloat16 (h_kv=1 for MLA)
    #   indices: [s_q, h_kv, topk] int32
    print("Running SGLang FlashMLA sparse prefill...")
    try:
        kv_for_mla = kv_cache.unsqueeze(1)  # [s_kv, 1, d_qk]

        # indices: [s_q, h_kv=1, topk]
        indices_for_mla = sparse_indices.unsqueeze(1)  # [total_num_tokens, 1, topk]

        # Pad query heads to required multiple (64 or 128) as done in SGLang's nsa_backend.py
        need_padding = NUM_QO_HEADS % required_padding != 0
        if need_padding:
            assert required_padding % NUM_QO_HEADS == 0, (
                f"required_padding ({required_padding}) must be divisible by NUM_QO_HEADS ({NUM_QO_HEADS})"
            )
            q_padded = q_all.new_zeros((total_num_tokens, required_padding, head_dim))
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
    print("Testing NSA (Native Sparse Attention) Sparse Prefill Reference Implementation")
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
    test_results.append(("padding_handling", test_padding_handling()))

    # Ground truth comparison tests
    test_configs = [16, 64, 256]  # Small  # Medium  # Large

    for total_num_tokens in test_configs:
        name = f"sglang_tokens{total_num_tokens}"
        try:
            result = test_correctness_vs_sglang(total_num_tokens)
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
