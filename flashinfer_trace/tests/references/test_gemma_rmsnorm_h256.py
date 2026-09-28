import flashinfer
import pytest
import torch


@torch.no_grad()
def run(hidden_states, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 256

    EPS = 1e-6

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * (1.0 + weight.to(torch.float32))
    return y.to(hidden_states.dtype)


def generate_random_inputs(batch_size, device="cuda"):
    hidden_size = 256
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    return {"hidden_states": hidden_states, "weight": weight}


def test_correctness(batch_size=8, atol=8e-3, rtol=1e-2):
    print(f"\n{'='*60}")
    print(f"Testing Gemma RMSNorm h256: batch_size={batch_size}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA not available")

    inputs = generate_random_inputs(batch_size, device)

    print(f"Input shape: {inputs['hidden_states'].shape}")
    print(f"Weight shape: {inputs['weight'].shape}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_output = run(inputs["hidden_states"].clone(), inputs["weight"])

    # Run FlashInfer implementation
    print("Running FlashInfer implementation...")
    fi_output = flashinfer.norm.gemma_rmsnorm(
        inputs["hidden_states"].clone().contiguous(),
        inputs["weight"].contiguous(),
        eps=1e-6,
    )

    # Compare outputs
    print("\nComparing outputs...")
    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    abs_diff = torch.abs(ref_f32 - fi_f32)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(fi_f32) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")

    close = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    if close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
    assert close, f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol})"


def main():
    print("Testing Gemma RMSNorm h256 Reference Implementation")

    test_configs = [1, 4, 8, 16, 32]
    passed = 0
    total = len(test_configs)

    for batch_size in test_configs:
        try:
            test_correctness(batch_size)
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
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
