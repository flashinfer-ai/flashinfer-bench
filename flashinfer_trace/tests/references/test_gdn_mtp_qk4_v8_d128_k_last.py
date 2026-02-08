"""
Test GDN MTP k-last reference implementation against FlashInfer kernel.

Run with:
    pytest test_gdn_mtp_qk4_v8_d128_k_last.py -v
    python test_gdn_mtp_qk4_v8_d128_k_last.py
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.utils import get_compute_capability

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"


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
    namespace = {"torch": torch, "math": math, "F": F}
    exec(reference_code, namespace)
    return namespace["run"]


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN MTP requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def run_kernel(
    q,
    k,
    v,
    initial_state,
    initial_state_indices,
    A_log,
    a,
    dt_bias,
    b,
    scale,
    cache_intermediate=True,
):
    """Run FlashInfer MTP kernel."""
    B, T, num_q_heads, K = q.shape
    num_v_heads = v.shape[2]
    pool_size = initial_state.shape[0]

    # Pre-allocate output
    output = torch.empty(B, T, num_v_heads, K, dtype=q.dtype, device=q.device)

    # Intermediate states buffer (optional)
    if cache_intermediate:
        intermediate_states_buffer = torch.zeros(
            pool_size, T, num_v_heads, K, K, dtype=torch.float32, device=q.device
        )
    else:
        intermediate_states_buffer = None

    # Call kernel
    out, final_state = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        initial_state=initial_state.clone(),
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=True,  # Don't update state for testing
        use_qk_l2norm=False,
    )

    return out, final_state


def generate_random_inputs(
    batch_size,
    seq_len,
    num_q_heads=4,
    num_k_heads=4,
    num_v_heads=8,
    head_size=128,
    device="cuda",
    seed=42,
):
    """Generate random inputs for testing."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B = batch_size
    T = seq_len
    K = head_size
    V = head_size
    dtype = torch.bfloat16

    # Use smaller magnitude for better numerical stability
    q = torch.randn(B, T, num_q_heads, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, num_k_heads, K, dtype=dtype, device=device) * 0.1
    k = F.normalize(k.float(), p=2.0, dim=-1).to(dtype)
    v = torch.randn(B, T, num_v_heads, V, dtype=dtype, device=device) * 0.1

    # Gate parameters with smaller scales
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.05
    a = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.05
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.05
    b = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.1

    # k-last layout: [pool_size, H, V, K]
    pool_size = B
    initial_state = (
        torch.randn(pool_size, num_v_heads, V, K, dtype=torch.float32, device=device) * 0.01
    )
    initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)

    # Use proper attention scaling
    scale = 1.0 / math.sqrt(head_size)

    return {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": initial_state,
        "initial_state_indices": initial_state_indices,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "scale": scale,
    }


def test_correctness(batch_size=2, seq_len=4, atol=5e-3, rtol=5e-3):
    """Test correctness of reference implementation against FlashInfer."""
    _skip_if_not_sm90_or_later()

    print(f"\n{'='*60}")
    print(f"Testing GDN MTP, batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    # Load definition and compile reference
    definition = load_definition("gdn_mtp_qk4_v8_d128_k_last")
    run = compile_reference(definition.reference)

    device = "cuda"
    inputs = generate_random_inputs(batch_size=batch_size, seq_len=seq_len, device=device)

    # Run reference from definition
    print("Running reference implementation from definition...")
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["initial_state"].clone(),
        inputs["initial_state_indices"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
        None,  # intermediate_states_buffer
    )
    ref_output, ref_final_state = ref_result

    # Run kernel
    print("Running FlashInfer kernel...")
    kernel_output, kernel_final_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["initial_state"].clone(),
        inputs["initial_state_indices"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
        cache_intermediate=True,
    )

    # Compare outputs
    print("\nComparing outputs...")

    ref_o_f32 = ref_output.float()
    kernel_o_f32 = kernel_output.float()

    # Absolute difference metrics
    abs_diff_o = torch.abs(ref_o_f32 - kernel_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    # Relative difference metrics (avoid division by zero)
    rel_diff_o = abs_diff_o / (torch.abs(ref_o_f32) + 1e-10)
    max_rel_diff_o = rel_diff_o.max().item()
    mean_rel_diff_o = rel_diff_o.mean().item()

    # Cosine similarity
    ref_flat = ref_o_f32.reshape(-1)
    kernel_flat = kernel_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), kernel_flat.unsqueeze(0)).item()

    # Mean Squared Error
    mse_o = ((ref_o_f32 - kernel_o_f32) ** 2).mean().item()

    print("\nOutput tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_o:.6e}")
    print(f"  Max relative difference: {max_rel_diff_o:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_o:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_o:.6e}")
    print(f"  Cosine similarity: {cosine_sim_o:.6f}")
    print(f"  MSE: {mse_o:.6e}")

    output_close = torch.allclose(ref_o_f32, kernel_o_f32, atol=atol, rtol=rtol)

    if output_close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
        return False


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [2, 4, 8])
def test_gdn_mtp(batch_size: int, seq_len: int):
    """Pytest parametrized test for various batch sizes and sequence lengths."""
    _skip_if_not_sm90_or_later()

    # Load definition and compile reference
    definition = load_definition("gdn_mtp_qk4_v8_d128_k_last")
    run = compile_reference(definition.reference)

    device = "cuda"
    inputs = generate_random_inputs(batch_size=batch_size, seq_len=seq_len, device=device)

    # Run reference from definition
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["initial_state"].clone(),
        inputs["initial_state_indices"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
        None,
    )
    ref_output, ref_final_state = ref_result

    # Run kernel
    kernel_output, kernel_final_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["initial_state"].clone(),
        inputs["initial_state_indices"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
        cache_intermediate=True,
    )

    atol, rtol = 1e-2, 1e-2

    torch.testing.assert_close(
        kernel_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        msg=f"Output mismatch for batch_size={batch_size}, seq_len={seq_len}",
    )

    print(f"✓ GDN MTP test passed (batch_size={batch_size}, seq_len={seq_len})")


def main():
    """Run tests."""
    print("Testing GDN MTP K-Last Reference Implementation")
    print(
        "Loading definition from: flashinfer_trace/definitions/gdn/gdn_mtp_qk4_v8_d128_k_last.json"
    )

    test_configs = [(2, 2), (2, 4), (4, 4)]

    passed = 0
    total = len(test_configs)

    for batch_size, seq_len in test_configs:
        try:
            if test_correctness(batch_size, seq_len):
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
