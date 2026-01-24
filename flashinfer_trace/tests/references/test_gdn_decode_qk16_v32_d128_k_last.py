"""
Test GDN decode k-last reference implementation against FlashInfer kernel.

Run with:
    pytest test_gdn_decode_qk16_v32_d128_k_last.py -v
    python test_gdn_decode_qk16_v32_d128_k_last.py
"""

import math

import pytest
import torch
import torch.nn.functional as F
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability
from test_utils import compare_tensors, get_reference_run, print_comparison_metrics


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Run FlashInfer kernel (pretranspose version uses k-last layout)."""
    B, T, num_q_heads, K = q.shape

    # Pre-allocate output
    output = torch.empty(B, T, v.shape[2], K, dtype=q.dtype, device=q.device)

    # Call kernel
    out, new_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state.clone(),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )

    return out, new_state


def generate_random_inputs(
    batch_size,
    num_q_heads=16,
    num_k_heads=16,
    num_v_heads=32,
    head_size=128,
    device="cuda",
    seed=42,
):
    """Generate random inputs for testing."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B = batch_size
    T = 1
    K = head_size
    V = head_size
    dtype = torch.bfloat16

    # Use smaller magnitude for better numerical stability
    q = torch.randn(B, T, num_q_heads, K, dtype=dtype, device=device) * 0.8
    k = torch.randn(B, T, num_k_heads, K, dtype=dtype, device=device) * 0.8
    # Normalize k for better conditioning (as done in prefill test)
    k = F.normalize(k.float(), p=2.0, dim=-1).to(dtype)
    v = torch.randn(B, T, num_v_heads, V, dtype=dtype, device=device) * 0.8

    # Gate parameters with smaller scales
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.05
    a = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.05
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device) * 0.05
    b = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.1

    # k-last layout: [B, H, V, K] - keep small for stability
    state = torch.randn(B, num_v_heads, V, K, dtype=torch.float32, device=device) * 0.01

    # Use proper attention scaling
    scale = 1.0 / math.sqrt(head_size)

    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "scale": scale,
    }


def test_correctness(batch_size=4, atol=5e-3, rtol=5e-3):
    """Test correctness of reference implementation against FlashInfer."""
    _skip_if_not_sm90_or_later()

    print(f"\n{'='*60}")
    print(f"Testing GDN decode k-last, batch_size={batch_size}")
    print(f"{'='*60}")

    # Load reference from definition
    run = get_reference_run("gdn_decode_qk16_v32_d128_k_last")

    device = "cuda"
    inputs = generate_random_inputs(batch_size=batch_size, device=device)

    # Run reference from definition
    print("Running reference implementation from definition...")
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )
    ref_output, ref_new_state = ref_result

    # Run kernel
    print("Running FlashInfer kernel...")
    kernel_output, kernel_new_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )

    # Compare outputs using test_utils
    print("\nComparing outputs...")
    output_metrics = compare_tensors(ref_output, kernel_output, atol=atol, rtol=rtol)
    print_comparison_metrics(output_metrics, tensor_name="Output tensor")

    state_metrics = compare_tensors(ref_new_state, kernel_new_state, atol=atol, rtol=rtol)
    print_comparison_metrics(state_metrics, tensor_name="State tensor")

    if output_metrics.all_close and state_metrics.all_close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
        return False


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_gdn_decode_k_last(batch_size: int):
    """Pytest parametrized test for various batch sizes."""
    _skip_if_not_sm90_or_later()

    # Load reference from definition
    run = get_reference_run("gdn_decode_qk16_v32_d128_k_last")

    device = "cuda"
    inputs = generate_random_inputs(batch_size=batch_size, device=device)

    # Run reference from definition
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )
    ref_output, ref_new_state = ref_result

    # Run kernel
    kernel_output, kernel_new_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )

    atol, rtol = 1e-2, 1e-2

    torch.testing.assert_close(
        kernel_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        msg=f"Output mismatch for batch_size={batch_size}",
    )
    torch.testing.assert_close(
        kernel_new_state,
        ref_new_state,
        atol=atol,
        rtol=rtol,
        msg=f"State mismatch for batch_size={batch_size}",
    )

    print(f"✓ GDN decode k-last test passed (batch_size={batch_size})")


def main():
    """Run tests."""
    print("Testing GDN Decode K-Last Reference Implementation")
    print(
        "Loading definition from: flashinfer_trace/definitions/gdn/gdn_decode_qk16_v32_d128_k_last.json"
    )

    test_configs = [1, 4, 16, 64, 256]

    passed = 0
    total = len(test_configs)

    for batch_size in test_configs:
        try:
            if test_correctness(batch_size):
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
