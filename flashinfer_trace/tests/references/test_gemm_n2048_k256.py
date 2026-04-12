import pytest
import torch


def run(A, B):
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2
    assert N == 2048
    assert K == 256
    C = torch.matmul(A, B.T)
    return C


def generate_random_inputs(M, N=2048, K=256, device="cuda"):
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    return {"A": A, "B": B}


def test_correctness(M=32, atol=1e-2, rtol=1e-2):
    print(f"\n{'='*60}")
    print(f"Testing GEMM N=2048, K=256, M={M}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA not available")

    inputs = generate_random_inputs(M, device=device)

    ref_C = run(inputs["A"], inputs["B"])

    A_f32 = inputs["A"].float()
    B_f32 = inputs["B"].float()
    expected = torch.matmul(A_f32, B_f32.T).to(torch.bfloat16)

    abs_diff = torch.abs(ref_C.float() - expected.float())
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")

    close = torch.allclose(ref_C.float(), expected.float(), atol=atol, rtol=rtol)
    if close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
    assert close, f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol})"


def main():
    print("Testing GEMM N=2048, K=256 Reference Implementation")

    test_configs = [1, 4, 16, 64, 256]
    passed = 0
    total = len(test_configs)

    for M in test_configs:
        try:
            test_correctness(M)
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
