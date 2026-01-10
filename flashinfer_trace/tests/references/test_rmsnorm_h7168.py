import flashinfer
import torch


@torch.no_grad()
def run(input, weight, eps, residual=None):
    """
    Reference implementation of RMSNorm with hidden_size=7168.

    Args:
        input: Input tensor of shape (B, 7168) in bfloat16
        weight: Weight tensor of shape (7168,) in bfloat16
        eps: Small epsilon value for numerical stability
        residual: Optional residual tensor of shape (B, 7168) in bfloat16

    Returns:
        dict with 'output' key containing normalized output in bfloat16
    """
    batch_size, hidden_size = input.shape

    # Check constants
    assert hidden_size == 7168

    # Perform computation in float32 for accuracy
    orig_dtype = input.dtype
    input_fp32 = input.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)

    if residual is not None:
        residual_fp32 = residual.to(torch.float32)
        input_fp32 = input_fp32 + residual_fp32

    # Compute RMS
    variance = input_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)

    # Apply normalization and weight
    output = (input_fp32 * rstd) * weight_fp32

    # Convert back to original dtype
    return {"output": output.to(orig_dtype)}


def generate_random_inputs(batch_size, with_residual=True, device="cuda"):
    """Generate random inputs for testing RMSNorm with hidden_size=7168."""

    hidden_size = 7168
    eps = 1e-6  # Common value for this configuration

    # Generate input tensor
    input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Generate weight tensor
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    # Generate residual if needed
    residual = None
    if with_residual:
        residual = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    return {"input": input, "weight": weight, "eps": eps, "residual": residual}


def test_correctness(batch_size=8, with_residual=True, atol=8e-3, rtol=1e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing RMSNorm h7168: batch_size={batch_size}, with_residual={with_residual}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    # Generate inputs
    inputs = generate_random_inputs(batch_size, with_residual, device)

    print(f"Input shape: {inputs['input'].shape}")
    print(f"Weight shape: {inputs['weight'].shape}")
    print(f"Epsilon: {inputs['eps']}")
    print(f"Has residual: {inputs['residual'] is not None}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_output = run(
        inputs["input"].clone(),
        inputs["weight"],
        inputs["eps"],
        inputs["residual"].clone() if inputs["residual"] is not None else None,
    )

    # Run FlashInfer implementation
    print("Running FlashInfer implementation...")
    input_fi = inputs["input"].clone().contiguous()
    weight_fi = inputs["weight"].contiguous()

    if inputs["residual"] is not None:
        residual_fi = inputs["residual"].clone().contiguous()
        # Use fused kernel for residual case
        flashinfer.norm.fused_add_rmsnorm(input_fi, residual_fi, weight_fi, inputs["eps"])
        fi_output = {"output": input_fi}
    else:
        # Standard RMSNorm without residual
        fi_out = flashinfer.norm.rmsnorm(input_fi, weight_fi, eps=inputs["eps"])
        fi_output = {"output": fi_out}

    # Compare outputs
    print("\nComparing outputs...")

    # Convert to float32 for comparison
    ref_out_f32 = ref_output["output"].float()
    fi_out_f32 = fi_output["output"].float()

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
    """Run comprehensive tests for RMSNorm h7168."""
    print("Testing RMSNorm h7168 Reference Implementation")

    # Test different configurations
    test_configs = [
        # (batch_size, with_residual)
        (1, True),  # Single batch with residual
        (1, False),  # Single batch without residual
        (4, True),  # Small batch with residual
        (8, True),  # Medium batch with residual
        (16, True),  # Large batch with residual
        (32, True),  # Very large batch with residual
    ]

    passed = 0
    total = len(test_configs)

    # Use bfloat16-appropriate tolerance
    atol = 8e-3  # 0.8% absolute tolerance
    rtol = 1e-2  # 1% relative tolerance

    for batch_size, with_residual in test_configs:
        try:
            if test_correctness(batch_size, with_residual, atol, rtol):
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
