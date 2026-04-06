import flashinfer
import torch


@torch.no_grad()
def run(q, k, cos_sin_cache, positions):
    head_size = 128
    rotary_dim = 64

    num_tokens = q.shape[0]

    # Check constants
    assert q.shape[-1] == head_size
    assert cos_sin_cache.shape[-1] == rotary_dim

    # Look up cos/sin from cache using position indices
    cos_sin = cos_sin_cache[positions]  # [num_tokens, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)  # each [num_tokens, rotary_dim/2]

    def apply_rotary_emb_neox(x, cos, sin):
        """NeoX-style: split into first/second half, rotate, concatenate."""
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    # Process Q
    q_f32 = q.to(torch.float32)
    q_shape = q_f32.shape
    q_3d = q_f32.view(num_tokens, -1, head_size)
    q_rot = q_3d[..., :rotary_dim]
    q_pass = q_3d[..., rotary_dim:]
    q_rot = apply_rotary_emb_neox(q_rot, cos, sin)
    q_out = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape).to(q.dtype)

    # Process K
    k_f32 = k.to(torch.float32)
    k_shape = k_f32.shape
    k_3d = k_f32.view(num_tokens, -1, head_size)
    k_rot = k_3d[..., :rotary_dim]
    k_pass = k_3d[..., rotary_dim:]
    k_rot = apply_rotary_emb_neox(k_rot, cos, sin)
    k_out = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape).to(k.dtype)

    return q_out, k_out


def generate_cos_sin_cache(max_seq_len, rotary_dim, rope_theta=10000.0, device="cuda"):
    """Generate cos_sin_cache matching FlashInfer/SGLang format."""
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, rotary_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_seq_len, rotary_dim]
    return cache


def generate_random_inputs(
    batch_size,
    seq_len,
    num_qo_heads=6,
    num_kv_heads=1,
    head_size=128,
    rotary_dim=64,
    rope_theta=10000.0,
    device="cuda",
):
    """Generate random inputs for testing."""
    nnz = batch_size * seq_len
    max_seq_len = seq_len + 128  # extra room in cache

    q = torch.randn(nnz, num_qo_heads, head_size, dtype=torch.bfloat16, device=device)
    k = torch.randn(nnz, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

    cos_sin_cache = generate_cos_sin_cache(max_seq_len, rotary_dim, rope_theta, device)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)

    return {
        "q": q,
        "k": k,
        "cos_sin_cache": cos_sin_cache,
        "positions": positions,
        "nnz": nnz,
        "head_size": head_size,
        "rotary_dim": rotary_dim,
    }


def test_correctness(batch_size=4, seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of NeoX-style reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}, seq_len={seq_len} (NeoX style)")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Constants from kernel definition
    num_qo_heads = 6
    num_kv_heads = 1
    head_size = 128
    rotary_dim = 64

    inputs = generate_random_inputs(
        batch_size, seq_len, num_qo_heads, num_kv_heads, head_size, rotary_dim, device=device
    )

    print(f"nnz (total tokens): {inputs['nnz']}")
    print(f"num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}")
    print(f"head_size: {head_size}, rotary_dim: {rotary_dim}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_q, ref_k = run(
        inputs["q"].clone(), inputs["k"].clone(), inputs["cos_sin_cache"], inputs["positions"]
    )

    # Run FlashInfer (cache-based API matching the definition's fi_api tag)
    print("Running FlashInfer...")
    fi_q = inputs["q"].clone()
    fi_k = inputs["k"].clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        inputs["positions"],
        fi_q.view(inputs["nnz"], -1),
        fi_k.view(inputs["nnz"], -1),
        head_size,
        inputs["cos_sin_cache"],
        is_neox=True,
    )

    # Compare outputs
    print("\nComparing outputs...")

    ref_q_f32 = ref_q.float()
    fi_q_f32 = fi_q.float()
    ref_k_f32 = ref_k.float()
    fi_k_f32 = fi_k.float()

    q_abs_diff = torch.abs(ref_q_f32 - fi_q_f32)
    k_abs_diff = torch.abs(ref_k_f32 - fi_k_f32)

    print("\nQuery comparison:")
    print(f"  Max absolute difference: {q_abs_diff.max().item():.6e}")
    print(f"  Mean absolute difference: {q_abs_diff.mean().item():.6e}")

    print("\nKey comparison:")
    print(f"  Max absolute difference: {k_abs_diff.max().item():.6e}")
    print(f"  Mean absolute difference: {k_abs_diff.mean().item():.6e}")

    # Check pass-through dimensions are unchanged
    q_pass_diff = torch.abs(
        inputs["q"][:, :, rotary_dim:].float() - ref_q[:, :, rotary_dim:].float()
    )
    k_pass_diff = torch.abs(
        inputs["k"][:, :, rotary_dim:].float() - ref_k[:, :, rotary_dim:].float()
    )
    print("\nPass-through dimensions (should be zero):")
    print(f"  Q pass-through max diff: {q_pass_diff.max().item():.6e}")
    print(f"  K pass-through max diff: {k_pass_diff.max().item():.6e}")

    q_close = torch.allclose(ref_q_f32, fi_q_f32, atol=atol, rtol=rtol)
    k_close = torch.allclose(ref_k_f32, fi_k_f32, atol=atol, rtol=rtol)
    pass_close = q_pass_diff.max().item() == 0.0 and k_pass_diff.max().item() == 0.0
    all_close = q_close and k_close and pass_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")
        if not q_close:
            print("  Q tensor mismatch")
        if not k_close:
            print("  K tensor mismatch")
        if not pass_close:
            print("  Pass-through dimensions were modified (should be unchanged)")

    return all_close


def main():
    """Run comprehensive tests for NeoX-style RoPE."""
    print("Testing RoPE Reference Implementation (NeoX-style, d128_rd64, partial, cos_sin_cache)")

    test_configs = [(1, 16), (4, 32), (8, 64), (16, 128), (32, 256)]

    passed = 0
    total = len(test_configs)

    for batch_size, seq_len in test_configs:
        try:
            if test_correctness(batch_size, seq_len):
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
