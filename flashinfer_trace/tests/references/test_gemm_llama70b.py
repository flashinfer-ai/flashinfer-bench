"""Reference tests for Llama 3.1/3.3 70B GEMM definitions (qkv_proj, o_proj, gate_up, down)."""

import math
from pathlib import Path

import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

# Llama 3.1/3.3 70B GEMM shape constants
# hidden_size=8192, num_q_heads=64, num_kv_heads=8, head_dim=128, intermediate_size=28672
HIDDEN_SIZE = 8192
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
INTERMEDIATE_SIZE = 28672


def load_definition(name: str) -> Definition:
    """Load a definition by name from definitions directory."""
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    """Compile reference implementation to callable function."""
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_random_gemm_inputs(M, N, K, device="cuda"):
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(N, K, dtype=torch.float16, device=device)
    return {"A": A, "B": B}


def test_correctness_single(definition_name, M, N, K, atol=1e-3, rtol=1e-3):
    """Test correctness of a single GEMM definition against PyTorch ground truth."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return None

    definition = load_definition(definition_name)
    run = compile_reference(definition.reference)

    inputs = generate_random_gemm_inputs(M, N, K, device)

    # Run reference
    ref_output = run(inputs["A"], inputs["B"])

    # Run PyTorch ground truth: C = A @ B.T
    gt_output = torch.matmul(inputs["A"], inputs["B"].T)

    # Compare
    ref_f32 = ref_output.float()
    gt_f32 = gt_output.float()

    abs_diff = torch.abs(ref_f32 - gt_f32)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    all_close = torch.allclose(ref_f32, gt_f32, atol=atol, rtol=rtol)
    if all_close:
        print(f"  ✓ PASSED {definition_name} M={M}: max_diff={max_abs_diff:.2e}, mean_diff={mean_abs_diff:.2e}")
    else:
        print(f"  ✗ FAILED {definition_name} M={M}: max_diff={max_abs_diff:.2e}, mean_diff={mean_abs_diff:.2e}")

    return all_close


def test_gemm_n10240_k8192(M=8, atol=1e-3, rtol=1e-3):
    """Test qkv_proj GEMM: (64+8+8)*128=10240 out, hidden=8192 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n10240_k8192 (Llama 70B qkv_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n10240_k8192", M, N=10240, K=8192, atol=atol, rtol=rtol)


def test_gemm_n8192_k8192(M=8, atol=1e-3, rtol=1e-3):
    """Test o_proj GEMM: hidden=8192 out, 64*128=8192 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n8192_k8192 (Llama 70B o_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n8192_k8192", M, N=8192, K=8192, atol=atol, rtol=rtol)


def test_gemm_n57344_k8192(M=8, atol=1e-3, rtol=1e-3):
    """Test gate_up GEMM: 2*28672=57344 out, hidden=8192 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n57344_k8192 (Llama 70B gate_up): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n57344_k8192", M, N=57344, K=8192, atol=atol, rtol=rtol)


def test_gemm_n8192_k28672(M=8, atol=1e-3, rtol=1e-3):
    """Test down_proj GEMM: hidden=8192 out, intermediate=28672 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n8192_k28672 (Llama 70B down_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n8192_k28672", M, N=8192, K=28672, atol=atol, rtol=rtol)


def main():
    """Run comprehensive tests for all Llama 3.1/3.3 70B GEMM definitions."""
    print("Testing Llama 3.1/3.3 70B GEMM Reference Implementations vs PyTorch")

    test_cases = [
        ("gemm_n10240_k8192", test_gemm_n10240_k8192),
        ("gemm_n8192_k8192", test_gemm_n8192_k8192),
        ("gemm_n57344_k8192", test_gemm_n57344_k8192),
        ("gemm_n8192_k28672", test_gemm_n8192_k28672),
    ]

    batch_sizes = [1, 4, 8, 16]
    total = len(test_cases) * len(batch_sizes)
    passed = 0
    skipped = 0

    for name, test_fn in test_cases:
        for M in batch_sizes:
            try:
                result = test_fn(M=M)
                if result is None:
                    skipped += 1
                elif result:
                    passed += 1
            except Exception as e:
                print(f"✗ {name} M={M} failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    counted = total - skipped
    print(f"Summary: {passed}/{counted} tests passed ({skipped} skipped)")
    print(f"{'='*60}")
    if skipped == total:
        print("WARNING: All tests skipped (CUDA not available)")
    elif passed == counted:
        print("✓ All tests passed!")
    else:
        print(f"✗ {counted - passed} tests failed")


if __name__ == "__main__":
    main()
