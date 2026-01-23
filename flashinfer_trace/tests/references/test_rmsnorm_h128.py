"""
Test RMSNorm h128 reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import flashinfer
import torch

from test_utils import get_reference_run

# Load reference implementations from definitions
run_rmsnorm = get_reference_run("rmsnorm_h128")

# Hidden size constant
HIDDEN_SIZE = 128


def generate_random_inputs(batch_size, device="cuda"):
    """Generate random inputs for testing RMSNorm with hidden_size=128."""
    hidden_states = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    return {"hidden_states": hidden_states, "weight": weight}


def test_correctness(batch_size=8, atol=8e-3, rtol=1e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing RMSNorm h128: batch_size={batch_size}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    # Generate inputs
    inputs = generate_random_inputs(batch_size, device)

    print(f"Hidden states shape: {inputs['hidden_states'].shape}")
    print(f"Weight shape: {inputs['weight'].shape}")

    # Run reference implementation from definition
    print("\nRunning reference implementation from definition...")
    ref_output = run_rmsnorm(
        inputs["hidden_states"].clone(),
        inputs["weight"],
    )

    # Run FlashInfer implementation
    print("Running FlashInfer implementation...")
    input_fi = inputs["hidden_states"].clone().contiguous()
    weight_fi = inputs["weight"].contiguous()
    fi_output = flashinfer.norm.rmsnorm(input_fi, weight_fi, eps=1e-6)

    # Compare outputs
    print("\nComparing outputs...")

    # Convert to float32 for comparison
    ref_out_f32 = ref_output.float()
    fi_out_f32 = fi_output.float()

    # Compute errors
    abs_diff = torch.abs(ref_out_f32 - fi_out_f32)
    rel_diff = abs_diff / (torch.abs(fi_out_f32) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\nOutput tensor comparison:")
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")

    # Check if outputs match within tolerance
    output_close = torch.allclose(ref_out_f32, fi_out_f32, atol=atol, rtol=rtol)

    if output_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

    return output_close


def main():
    """Run comprehensive tests for RMSNorm h128."""
    print("Testing RMSNorm h128 Reference Implementation (from definition)")

    # Test different batch sizes
    test_configs = [1, 4, 8, 16, 32]

    passed = 0
    total = len(test_configs)

    # Use bfloat16-appropriate tolerance
    atol = 8e-3  # 0.8% absolute tolerance
    rtol = 1e-2  # 1% relative tolerance

    for batch_size in test_configs:
        try:
            if test_correctness(batch_size, atol, rtol):
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
