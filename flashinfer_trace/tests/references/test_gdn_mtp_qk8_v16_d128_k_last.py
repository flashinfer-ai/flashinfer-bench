import os

"""
Test GDN MTP (Multi-Token Prediction) k-last reference implementation.
Configuration: num_q_heads=8, num_v_heads=16, head_size=128 (Qwen3.5 TP=2).

MTP processes multiple tokens (T > 1) sequentially with state updates,
used for speculative decoding verification.

No FlashInfer ground truth kernel for MTP — reference-only validation.

Run with:
    pytest test_gdn_mtp_qk8_v16_d128_k_last.py -v
    python test_gdn_mtp_qk8_v16_d128_k_last.py
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(
    os.environ.get("DEFINITIONS_DIR", Path(__file__).parent.parent.parent / "definitions")
)


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


requires_cuda = pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA devices not available"
)

# Load definition and compile reference
definition = load_definition("gdn_mtp_qk8_v16_d128_k_last")
reference_gdn_mtp = compile_reference(definition.reference)


def generate_random_inputs(
    batch_size,
    seq_len,
    pool_size=None,
    num_q_heads=8,
    num_k_heads=8,
    num_v_heads=16,
    head_size=128,
    device="cuda",
    seed=42,
):
    """Generate random inputs for MTP testing."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if pool_size is None:
        pool_size = batch_size

    B = batch_size
    T = seq_len
    K = head_size
    dtype = torch.bfloat16

    q = torch.randn(B, T, num_q_heads, K, dtype=dtype, device=device) * 0.8
    k = torch.randn(B, T, num_k_heads, K, dtype=dtype, device=device) * 0.8
    k = F.normalize(k.float(), p=2.0, dim=-1).to(dtype)
    v = torch.randn(B, T, num_v_heads, K, dtype=dtype, device=device) * 0.8

    # State pool in k-last layout: [pool_size, H, V, K]
    initial_state = (
        torch.randn(pool_size, num_v_heads, K, K, dtype=torch.float32, device=device) * 0.01
    )

    # Indices mapping each batch to its state in the pool
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.05
    a = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.05
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.05
    b = torch.randn(B, T, num_v_heads, dtype=dtype, device=device) * 0.1

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


def verify_mtp_sequential_consistency(batch_size=2, seq_len=4, device="cuda"):
    """
    Verify MTP produces the same results as running decode step-by-step.
    This validates sequential state updates are correct.
    """
    # Load decode definition for comparison
    decode_def = load_definition("gdn_decode_qk8_v16_d128_k_last")
    decode_run = compile_reference(decode_def.reference)

    inputs = generate_random_inputs(batch_size, seq_len, device=device)

    # Run MTP (processes all T tokens at once)
    mtp_output, mtp_final_state = reference_gdn_mtp(
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
    )

    # Run decode step-by-step for comparison
    num_v_heads = 16
    head_size = 128
    decode_outputs = []

    for b_idx in range(batch_size):
        state_idx = int(inputs["initial_state_indices"][b_idx].item())
        current_state = inputs["initial_state"][state_idx].clone()

        for t in range(seq_len):
            q_t = inputs["q"][b_idx, t : t + 1].unsqueeze(0)  # [1, 1, H, K]
            k_t = inputs["k"][b_idx, t : t + 1].unsqueeze(0)
            v_t = inputs["v"][b_idx, t : t + 1].unsqueeze(0)
            a_t = inputs["a"][b_idx, t : t + 1].unsqueeze(0)
            b_t = inputs["b"][b_idx, t : t + 1].unsqueeze(0)

            out_t, new_state = decode_run(
                q_t,
                k_t,
                v_t,
                current_state.unsqueeze(0),
                inputs["A_log"].clone(),
                a_t,
                inputs["dt_bias"].clone(),
                b_t,
                inputs["scale"],
            )
            decode_outputs.append(out_t[0, 0])  # [H, V]
            current_state = new_state[0]

    # Compare MTP output with sequential decode
    for b_idx in range(batch_size):
        for t in range(seq_len):
            idx = b_idx * seq_len + t
            mtp_out = mtp_output[b_idx, t].float()
            decode_out = decode_outputs[idx].float()

            abs_diff = torch.abs(mtp_out - decode_out)
            max_diff = abs_diff.max().item()

            if max_diff > 1e-4:
                return False, f"Mismatch at batch={b_idx}, t={t}: max_diff={max_diff:.6e}"

    return True, "Sequential consistency verified"


@requires_cuda
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [2, 4, 8])
def test_gdn_mtp_runs(batch_size: int, seq_len: int):
    """Test that MTP reference runs without error and produces valid output."""
    device = "cuda"
    inputs = generate_random_inputs(batch_size, seq_len, device=device)

    output, final_state = reference_gdn_mtp(
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
    )

    # Check output shapes
    assert output.shape == (batch_size, seq_len, 16, 128), f"Output shape mismatch: {output.shape}"
    assert output.dtype == torch.bfloat16

    # Check no NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print(f"✓ GDN MTP test passed (batch_size={batch_size}, seq_len={seq_len})")


@requires_cuda
def test_gdn_mtp_sequential_consistency():
    """Verify MTP produces same results as step-by-step decode."""
    passed, msg = verify_mtp_sequential_consistency(batch_size=2, seq_len=4)
    assert passed, msg
    print(f"✓ {msg}")


def main():
    """Run tests manually."""
    print("Testing GDN MTP qk8_v16 K-Last Reference Implementation")

    test_configs = [(1, 2), (2, 4), (4, 8)]

    passed = 0
    total = len(test_configs)

    for batch_size, seq_len in test_configs:
        try:
            device = "cuda"
            inputs = generate_random_inputs(batch_size, seq_len, device=device)

            output, final_state = reference_gdn_mtp(
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
            )

            assert output.shape == (batch_size, seq_len, 16, 128)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

            print(f"\n✓ PASSED batch_size={batch_size}, seq_len={seq_len}")
            print(
                f"  Output shape: {output.shape}, range: [{output.float().min():.4f}, {output.float().max():.4f}]"
            )
            passed += 1

        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Sequential consistency check
    print("\nVerifying sequential consistency (MTP vs step-by-step decode)...")
    try:
        ok, msg = verify_mtp_sequential_consistency()
        print(f"  {msg}")
        if ok:
            passed += 1
        total += 1
    except Exception as e:
        print(f"  Sequential consistency check failed: {e}")
        total += 1

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
