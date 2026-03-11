"""Reference test for rmsnorm_h3584 (Qwen2.5 7B)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

HIDDEN_SIZE = 3584
EPS = 1e-6


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


def generate_random_inputs(batch_size, device="cuda"):
    hidden_states = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    return {"hidden_states": hidden_states, "weight": weight}


def test_correctness(batch_size=8, atol=8e-3, rtol=1e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing RMSNorm h3584 (Qwen2.5 7B): batch_size={batch_size}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return None

    definition = load_definition("rmsnorm_h3584")
    run = compile_reference(definition.reference)

    inputs = generate_random_inputs(batch_size, device)

    print(f"Input shape: {inputs['hidden_states'].shape}")
    print(f"Weight shape: {inputs['weight'].shape}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_output = run(inputs["hidden_states"].clone(), inputs["weight"])

    # Run FlashInfer
    print("Running FlashInfer implementation...")
    input_fi = inputs["hidden_states"].clone().contiguous()
    weight_fi = inputs["weight"].contiguous()
    fi_output = flashinfer.norm.rmsnorm(input_fi, weight_fi, eps=EPS)

    # Compare
    print("\nComparing outputs...")
    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    abs_diff = torch.abs(ref_f32 - fi_f32)
    rel_diff = abs_diff / (torch.abs(fi_f32) + 1e-8)

    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"Max relative difference: {rel_diff.max().item():.6e}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6e}")

    all_close = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

    return all_close


def main():
    """Run comprehensive tests for RMSNorm h3584."""
    print("Testing RMSNorm h3584 (Qwen2.5 7B) Reference Implementation")

    test_configs = [1, 4, 8, 16, 32]
    atol, rtol = 8e-3, 1e-2

    passed = 0
    skipped = 0
    for batch_size in test_configs:
        try:
            result = test_correctness(batch_size, atol, rtol)
            if result is None:
                skipped += 1
            elif result:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    counted = len(test_configs) - skipped
    print(f"Summary: {passed}/{counted} tests passed ({skipped} skipped)")
    print(f"{'='*60}")
    if skipped == len(test_configs):
        print("WARNING: All tests skipped (CUDA not available)")
    elif passed == counted:
        print("✓ All tests passed!")
    else:
        print(f"✗ {counted - passed} tests failed")


if __name__ == "__main__":
    main()
