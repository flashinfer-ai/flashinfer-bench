"""Reference tests for Qwen2.5 7B GEMM definitions (qkv_proj, o_proj, gate_up, down)."""

import math
from pathlib import Path

import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

# Qwen2.5 7B GEMM shape constants
# hidden_size=3584, num_q_heads=28, num_kv_heads=4, head_dim=128, intermediate_size=18944
HIDDEN_SIZE = 3584
HEAD_DIM = 128
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 18944


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


def test_gemm_n4608_k3584(M=8, atol=1e-3, rtol=1e-3):
    """Test qkv_proj GEMM: (28+4+4)*128=4608 out, hidden=3584 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n4608_k3584 (Qwen2.5 7B qkv_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n4608_k3584", M, N=4608, K=3584, atol=atol, rtol=rtol)


def test_gemm_n3584_k3584(M=8, atol=1e-3, rtol=1e-3):
    """Test o_proj GEMM: hidden=3584 out, 28*128=3584 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n3584_k3584 (Qwen2.5 7B o_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n3584_k3584", M, N=3584, K=3584, atol=atol, rtol=rtol)


def test_gemm_n37888_k3584(M=8, atol=1e-3, rtol=1e-3):
    """Test gate_up GEMM: 2*18944=37888 out, hidden=3584 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n37888_k3584 (Qwen2.5 7B gate_up): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n37888_k3584", M, N=37888, K=3584, atol=atol, rtol=rtol)


def test_gemm_n3584_k18944(M=8, atol=1e-3, rtol=1e-3):
    """Test down_proj GEMM: hidden=3584 out, intermediate=18944 in."""
    print(f"\n{'='*60}")
    print(f"Testing gemm_n3584_k18944 (Qwen2.5 7B down_proj): M={M}")
    print(f"{'='*60}")
    return test_correctness_single("gemm_n3584_k18944", M, N=3584, K=18944, atol=atol, rtol=rtol)


def main():
    """Run comprehensive tests for all Qwen2.5 7B GEMM definitions."""
    print("Testing Qwen2.5 7B GEMM Reference Implementations vs PyTorch")

    test_cases = [
        ("gemm_n4608_k3584", test_gemm_n4608_k3584),
        ("gemm_n3584_k3584", test_gemm_n3584_k3584),
        ("gemm_n37888_k3584", test_gemm_n37888_k3584),
        ("gemm_n3584_k18944", test_gemm_n3584_k18944),
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
