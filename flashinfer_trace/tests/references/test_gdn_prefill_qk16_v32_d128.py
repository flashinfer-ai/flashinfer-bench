import math

import numpy as np
import torch

# Import FlashInfer GDN API
try:
    from flashinfer import chunk_gated_delta_rule
    from flashinfer.utils import is_sm90a_supported

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    return a_f32 @ b_f32


@torch.no_grad()
def run(q, k, v, g, beta, cu_seqlens, initial_state, scale):
    """
    Gated Delta Net reference implementation.

    Delta rule update:
    state_new = alpha * state_old - k^T * (k @ state_old) + k^T * (beta * v + (1-beta) * k @ state_old)
    output = scale * q @ state_new
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    # Check constants
    assert num_q_heads == 16
    assert num_k_heads == 16
    assert num_v_heads == 32
    assert head_size == 128

    # Default scale
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Expand for GVA: q and k get repeated to match v heads
    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, 32, 128]
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [T, 32, 128]

    # Initialize outputs
    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    final_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    # Process each sequence
    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        # Get initial state for this sequence
        if initial_state is not None:
            state_HKV = initial_state[seq_idx].clone().to(torch.float32)
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        # Process token by token (reference implementation)
        for i in range(seq_len):
            t = seq_start + i
            q_H1Q = q_exp[t].unsqueeze(1).to(torch.float32)  # [H, 1, K]
            k_H1K = k_exp[t].unsqueeze(1).to(torch.float32)  # [H, 1, K]
            v_H1V = v[t].unsqueeze(1).to(torch.float32)  # [H, 1, V]
            alpha_H11 = g[t].unsqueeze(1).unsqueeze(2)  # [H, 1, 1]
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)  # [H, 1, 1]

            # Delta rule update
            old_state_HKV = alpha_H11 * state_HKV
            old_v_H1V = matmul(k_H1K, old_state_HKV)  # [H, 1, V]
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum("htv,htk->hkv", old_v_H1V, k_H1K)
            state_update = torch.einsum("htv,htk->hkv", new_v_H1V, k_H1K)
            state_HKV = old_state_HKV - state_remove + state_update

            # Compute output
            o_H1V = scale * matmul(q_H1Q, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        final_state[seq_idx] = state_HKV

    return {"output": output, "final_state": final_state}


def generate_random_inputs(
    num_seqs,
    max_seq_len,
    num_q_heads=16,
    num_k_heads=16,
    num_v_heads=32,
    head_size=128,
    device="cuda",
):
    """Generate random inputs for GDN prefill testing."""

    # Generate random sequence lengths for each sequence
    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32, device=device)
    total_seq_len = seq_lens.sum().item()

    # Generate cu_seqlens
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens.to(torch.int64), dim=0)

    # Number of state/attention/beta heads
    num_sab_heads = max(num_q_heads, num_v_heads)

    # Generate input tensors
    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device)

    # Gate tensors (float32 for numerical stability)
    # g (alpha/forget gate): values typically in (0, 1)
    g = torch.sigmoid(torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device=device))
    # beta (update gate): values typically in (0, 1)
    beta = torch.sigmoid(
        torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
    )

    # Optional initial state
    initial_state = torch.randn(
        num_seqs, num_sab_heads, head_size, head_size, dtype=torch.float32, device=device
    ) * 0.01  # Small initial state

    # Scale factor
    scale = 1.0 / math.sqrt(head_size)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "scale": scale,
        "seq_lens": seq_lens,
        "total_seq_len": total_seq_len,
        "num_seqs": num_seqs,
    }


def test_correctness(num_seqs=4, max_seq_len=32, atol=1e-2, rtol=5e-2):
    """Test correctness of GDN prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing num_seqs={num_seqs}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    if not FLASHINFER_AVAILABLE:
        print("WARNING: FlashInfer not available, skipping test")
        return True

    if not is_sm90a_supported(torch.device(device)):
        print("WARNING: SM90a (Hopper) not supported, skipping test")
        return True

    # Constants from kernel definition (Qwen3 Next GVA configuration)
    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    head_size = 128

    # Generate inputs
    inputs = generate_random_inputs(
        num_seqs,
        max_seq_len,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        device,
    )

    print(f"Generated sequence lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Total tokens: {inputs['total_seq_len']}")
    print(f"Configuration: num_q_heads={num_q_heads}, num_v_heads={num_v_heads}, head_size={head_size}")
    print(f"Mode: GVA (Grouped Value Attention)")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_results = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
        inputs["scale"],
    )
    ref_output = ref_results["output"]
    ref_final_state = ref_results["final_state"]

    # Run FlashInfer
    print("Running FlashInfer chunk_gated_delta_rule...")
    fi_output, fi_final_state = chunk_gated_delta_rule(
        q=inputs["q"].contiguous(),
        k=inputs["k"].contiguous(),
        v=inputs["v"].contiguous(),
        g=inputs["g"].contiguous(),
        beta=inputs["beta"].contiguous(),
        scale=inputs["scale"],
        initial_state=inputs["initial_state"].contiguous(),
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
    )

    # Compare outputs
    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"FlashInfer output shape: {fi_output.shape}")
    print(f"Reference final_state shape: {ref_final_state.shape}")
    print(f"FlashInfer final_state shape: {fi_final_state.shape}")

    # Check numerical accuracy for output
    o_diff = torch.abs(ref_output.float() - fi_output.float())
    print(f"\nOutput max diff: {o_diff.max().item():.6f}")
    print(f"Output mean diff: {o_diff.mean().item():.6f}")

    # Check numerical accuracy for final state
    state_diff = torch.abs(ref_final_state - fi_final_state)
    print(f"Final state max diff: {state_diff.max().item():.6f}")
    print(f"Final state mean diff: {state_diff.mean().item():.6f}")

    # Compute cosine similarity
    ref_flat = ref_output.float().flatten()
    fi_flat = fi_output.float().flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), fi_flat.unsqueeze(0)
    ).item()
    print(f"Output cosine similarity: {cosine_sim:.6f}")

    ref_state_flat = ref_final_state.flatten()
    fi_state_flat = fi_final_state.flatten()
    state_cosine_sim = torch.nn.functional.cosine_similarity(
        ref_state_flat.unsqueeze(0), fi_state_flat.unsqueeze(0)
    ).item()
    print(f"Final state cosine similarity: {state_cosine_sim:.6f}")

    # Check if outputs match within tolerance
    output_close = torch.allclose(ref_output.float(), fi_output.float(), atol=atol, rtol=rtol)
    state_close = torch.allclose(ref_final_state, fi_final_state, atol=atol, rtol=rtol)
    all_close = output_close and state_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and final_state match within tolerance (atol={atol}, rtol={rtol})")
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
                total_seq_len, num_o_heads, head_size_out = ref_output.shape
                seq_idx = idx // (num_o_heads * head_size_out)
                head_idx = (idx % (num_o_heads * head_size_out)) // head_size_out
                dim_idx = idx % head_size_out

                ref_val = ref_output.float().flatten()[idx].item()
                fi_val = fi_output.float().flatten()[idx].item()

                print(
                    f"  [{seq_idx}, {head_idx}, {dim_idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_errors[i].item():.6e}"
                )

        if not state_close:
            # Find state errors
            flat_state_diff = state_diff.flatten()
            top_k = min(5, flat_state_diff.numel())
            top_state_errors, top_state_indices = torch.topk(flat_state_diff, top_k)

            print(f"\nTop {top_k} final_state error locations:")
            for i in range(top_k):
                idx = top_state_indices[i].item()

                ref_val = ref_final_state.flatten()[idx].item()
                fi_val = fi_final_state.flatten()[idx].item()

                print(
                    f"  [flat_idx={idx}]: "
                    f"ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_state_errors[i].item():.6e}"
                )

    return all_close


def test_no_initial_state(num_seqs=2, max_seq_len=16, atol=1e-2, rtol=5e-2):
    """Test GDN with no initial state (zeros)."""
    print(f"\n{'='*60}")
    print(f"Testing with no initial state: num_seqs={num_seqs}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    if not FLASHINFER_AVAILABLE:
        print("WARNING: FlashInfer not available, skipping test")
        return True

    if not is_sm90a_supported(torch.device(device)):
        print("WARNING: SM90a (Hopper) not supported, skipping test")
        return True

    # Generate inputs without initial state
    inputs = generate_random_inputs(num_seqs, max_seq_len, device=device)

    # Run reference with None initial_state
    print("\nRunning reference implementation (no initial state)...")
    ref_results = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        None,  # No initial state
        inputs["scale"],
    )

    # Run FlashInfer with None initial_state
    print("Running FlashInfer (no initial state)...")
    fi_output, fi_final_state = chunk_gated_delta_rule(
        q=inputs["q"].contiguous(),
        k=inputs["k"].contiguous(),
        v=inputs["v"].contiguous(),
        g=inputs["g"].contiguous(),
        beta=inputs["beta"].contiguous(),
        scale=inputs["scale"],
        initial_state=None,
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
    )

    # Compare outputs
    o_diff = torch.abs(ref_results["output"].float() - fi_output.float())
    print(f"Output max diff: {o_diff.max().item():.6f}")
    print(f"Output mean diff: {o_diff.mean().item():.6f}")

    output_close = torch.allclose(
        ref_results["output"].float(), fi_output.float(), atol=atol, rtol=rtol
    )

    if output_close:
        print(f"✓ PASSED: No initial state test")
    else:
        print(f"✗ FAILED: No initial state test")

    return output_close


def main():
    """Run comprehensive tests."""
    print("Testing Gated Delta Net (GDN) Prefill Reference Implementation")
    print("Configuration: num_q_heads=16, num_k_heads=16, num_v_heads=32, head_size=128 (GVA)")

    # Test different configurations
    test_configs = [
        # (num_seqs, max_seq_len)
        (1, 8),  # Single sequence, short
        (2, 16),  # Two sequences, medium
        (4, 32),  # Multiple sequences, longer
    ]

    passed = 0
    total = len(test_configs) + 1  # +1 for no initial state test

    for num_seqs, max_seq_len in test_configs:
        try:
            if test_correctness(num_seqs, max_seq_len):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback

            traceback.print_exc()

    # Test with no initial state
    try:
        if test_no_initial_state():
            passed += 1
    except Exception as e:
        print(f"✗ No initial state test failed with exception: {str(e)}")
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
