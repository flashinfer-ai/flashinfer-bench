import torch
import torch.nn.functional as F


@torch.no_grad()
def run(A, B):
    """
    Reference implementation of GEMM C = A @ B.T with N=8192, K=8192.

    This corresponds to Llama 3.1/3.3 70B attn.o_proj (attention output projection):
      Input:  num_heads * head_dim = 64 * 128 = 8192
      Output: hidden_size = 8192

    Args:
        A: Input tensor of shape (M, 8192) in float16
        B: Weight tensor of shape (8192, 8192) in float16

    Returns:
        C: Output tensor of shape (M, 8192) in float16
    """
    N, K = B.shape
    assert K == 8192, f"Expected K=8192, got {K}"
    assert N == 8192, f"Expected N=8192, got {N}"
    assert A.shape[1] == K, f"Expected A.shape[1]={K}, got {A.shape[1]}"

    C = torch.matmul(A, B.T)
    return C


def generate_random_inputs(M, device="cuda"):
    """Generate random inputs for testing GEMM N=8192, K=8192."""
    N = 8192
    K = 8192

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(N, K, dtype=torch.float16, device=device)

    return {"A": A, "B": B}


def test_correctness(M=128, atol=1e-2, rtol=1e-2):
    """Test correctness of reference GEMM against torch.nn.functional.linear."""
    print(f"\n{'='*60}")
    print(f"Testing GEMM N=8192, K=8192: M={M}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    # Generate inputs
    inputs = generate_random_inputs(M, device)
    A = inputs["A"]
    B = inputs["B"]

    print(f"A shape: {A.shape}, dtype: {A.dtype}")
    print(f"B shape: {B.shape}, dtype: {B.dtype}")

    # Run reference implementation (matmul-based)
    print("\nRunning reference implementation (A @ B.T)...")
    ref_output = run(A, B)

    # Run F.linear implementation (what FlashInfer baseline uses)
    print("Running F.linear implementation...")
    fi_output = F.linear(A, B)

    # Compare outputs
    print("\nComparing outputs...")

    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    abs_diff = torch.abs(ref_f32 - fi_f32)
    rel_diff = abs_diff / (torch.abs(fi_f32) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\nOutput tensor comparison:")
    print(f"  Output shape: {ref_output.shape}")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")

    output_close = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)

    if output_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

    return output_close


def main():
    """Run comprehensive tests for GEMM N=8192, K=8192."""
    print("Testing GEMM N=8192, K=8192 Reference Implementation")

    test_M_values = [1, 4, 16, 64, 128, 256]

    passed = 0
    total = len(test_M_values)

    for M in test_M_values:
        try:
            if test_correctness(M):
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
        exit(1)


if __name__ == "__main__":
    main()
