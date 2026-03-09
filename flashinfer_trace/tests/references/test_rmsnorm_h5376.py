"""Reference test for rmsnorm_h5376 (Gemma 3 27B)."""

import flashinfer
import torch

HIDDEN_SIZE = 5376
EPS = 1e-6


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape

    # Check constants
    assert hidden_size == HIDDEN_SIZE

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


def generate_random_inputs(batch_size, device="cuda"):
    hidden_states = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    return {"hidden_states": hidden_states, "weight": weight}


def test_correctness(batch_size=8, atol=8e-3, rtol=1e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing RMSNorm h5376 (Gemma 3 27B): batch_size={batch_size}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

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
    """Run comprehensive tests for RMSNorm h5376."""
    print("Testing RMSNorm h5376 (Gemma 3 27B) Reference Implementation")

    test_configs = [1, 4, 8, 16, 32]
    atol, rtol = 8e-3, 1e-2

    passed = 0
    for batch_size in test_configs:
        try:
            if test_correctness(batch_size, atol, rtol):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(test_configs)} tests passed")
    print(f"{'='*60}")
    if passed == len(test_configs):
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(test_configs) - passed} tests failed")


if __name__ == "__main__":
    main()
