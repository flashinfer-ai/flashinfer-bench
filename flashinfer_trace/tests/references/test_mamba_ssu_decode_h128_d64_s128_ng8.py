"""
Test Mamba2 SSU decode reference implementation against FlashInfer kernel.

Definition: mamba_ssu_decode_h128_d64_s128_ng8
Model: NVIDIA NemotronH-8B (TP=1)
  - nheads=128, head_dim=64, dstate=128, ngroups=8
  - nheads/ngroups=16 (supported by FlashInfer)

FlashInfer kernel: flashinfer.mamba.selective_state_update

Run with:
    pytest test_mamba_ssu_decode_h128_d64_s128_ng8.py -v
    python test_mamba_ssu_decode_h128_d64_s128_ng8.py
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

# Kernel constants (NemotronH-8B, TP=1)
NHEADS = 128
HEAD_DIM = 64
DSTATE = 128
NGROUPS = 8
RATIO = NHEADS // NGROUPS  # = 16, supported by FlashInfer


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
    namespace = {"torch": torch, "F": F}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_inputs(batch_size, seed=42, device="cuda"):
    """
    Generate test inputs matching NemotronH-8B Mamba layer parameters.

    Tensor shapes and dtypes match the FlashInfer kernel requirements:
      - state: (state_cache_size, nheads, head_dim, dstate), bfloat16
      - x, dt: (batch_size, nheads, head_dim), bfloat16
      - B, C: (batch_size, ngroups, dstate), bfloat16
      - A: (nheads, head_dim, dstate), float32 (model weight)
      - D, dt_bias: (nheads, head_dim), bfloat16 (model weights)
      - state_batch_indices: (batch_size,), int32
    """
    torch.manual_seed(seed)

    # State cache is larger than batch (paged KV cache pattern)
    state_cache_size = max(256, batch_size * 8)

    # Unique random slot indices for each batch element
    perm = torch.randperm(state_cache_size, device=device)
    slot_idx = perm[:batch_size].to(torch.int32)

    # SSM state cache: bfloat16 (user-configurable; bfloat16 is common deployment)
    state = (
        torch.randn(state_cache_size, NHEADS, HEAD_DIM, DSTATE, dtype=torch.bfloat16, device=device)
        * 0.01
    )  # small initial state for numerical stability

    # Input: bfloat16 (from hidden states after in_proj), contiguous (stride(1)==HEAD_DIM required)
    x = torch.randn(batch_size, NHEADS, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1

    # Time step (delta): TIE_HDIM=True — one scalar per head, broadcast over head_dim.
    # FlashInfer requires dt.stride(1)==1, dt.stride(2)==0.
    # Shape: (batch, nheads, head_dim), strides: (nheads, 1, 0)
    dt_scalar = torch.randn(batch_size, NHEADS, dtype=torch.bfloat16, device=device) * 0.1
    dt = dt_scalar.unsqueeze(-1).expand(batch_size, NHEADS, HEAD_DIM)

    # Decay matrix A: TIE_HDIM=True — one scalar per head, broadcast over dim and dstate.
    # FlashInfer requires A.stride(0)==1, A.stride(1)==0, A.stride(2)==0.
    # Shape: (nheads, head_dim, dstate), strides: (1, 0, 0)
    A_scalar = -torch.rand(NHEADS, dtype=torch.float32, device=device) - 0.5  # negative
    A = A_scalar.view(NHEADS, 1, 1).expand(NHEADS, HEAD_DIM, DSTATE)

    # B and C gates: bfloat16, per group
    B = torch.randn(batch_size, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device) * 0.1
    C = torch.randn(batch_size, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device) * 0.1

    # Skip connection D: TIE_HDIM=True — one scalar per head, broadcast over head_dim.
    # FlashInfer requires D.stride(0)==1, D.stride(1)==0.
    # Shape: (nheads, head_dim), strides: (1, 0)
    D_scalar = torch.randn(NHEADS, dtype=torch.bfloat16, device=device) * 0.1
    D = D_scalar.view(NHEADS, 1).expand(NHEADS, HEAD_DIM)

    # Dt bias: TIE_HDIM=True — one scalar per head, broadcast over head_dim.
    # FlashInfer requires dt_bias.stride(0)==1, dt_bias.stride(1)==0.
    # Shape: (nheads, head_dim), strides: (1, 0)
    dt_bias_scalar = torch.randn(NHEADS, dtype=torch.bfloat16, device=device) * 0.1
    dt_bias = dt_bias_scalar.view(NHEADS, 1).expand(NHEADS, HEAD_DIM)

    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state_batch_indices": slot_idx,
        "state_cache_size": state_cache_size,
    }


def run_flashinfer_kernel(inputs):
    """Run the FlashInfer selective_state_update kernel."""
    import flashinfer.mamba

    state_for_kernel = inputs["state"].clone()
    output = flashinfer.mamba.selective_state_update(
        state_for_kernel,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=None,
        dt_bias=inputs["dt_bias"],
        dt_softplus=True,
        state_batch_indices=inputs["state_batch_indices"],
        pad_slot_id=-1,
    )
    return output, state_for_kernel


def test_correctness(batch_size=4, atol=1e-2, rtol=1e-2):
    """Test that definition reference matches FlashInfer kernel output."""
    print(f"\n{'='*60}")
    print(f"Testing mamba_ssu_decode_h128_d64_s128_ng8, batch_size={batch_size}")
    print(f"NemotronH-8B: nheads={NHEADS}, head_dim={HEAD_DIM}, dstate={DSTATE}, ngroups={NGROUPS}")
    print(f"{'='*60}")

    device = "cuda"

    # Load definition and compile reference
    definition = load_definition("mamba_ssu_decode_h128_d64_s128_ng8")
    run = compile_reference(definition.reference)

    inputs = generate_inputs(batch_size=batch_size, device=device)

    # Run reference from definition
    print("Running reference implementation from definition...")
    ref_output, ref_state = run(
        inputs["state"].clone(),
        inputs["x"].clone(),
        inputs["dt"].clone(),
        inputs["A"].clone(),
        inputs["B"].clone(),
        inputs["C"].clone(),
        inputs["D"].clone(),
        inputs["dt_bias"].clone(),
        inputs["state_batch_indices"].clone(),
    )

    # Run FlashInfer kernel
    print("Running FlashInfer selective_state_update kernel...")
    kernel_output, kernel_state = run_flashinfer_kernel(inputs)

    # Compare outputs
    print("\nComparing outputs...")
    ref_o = ref_output.float()
    ker_o = kernel_output.float()

    abs_diff_o = (ref_o - ker_o).abs()
    print(f"  Output max abs diff: {abs_diff_o.max().item():.4e}")
    print(f"  Output mean abs diff: {abs_diff_o.mean().item():.4e}")

    # Compare states at active slots
    slot_idx = inputs["state_batch_indices"]
    ref_s = ref_state[slot_idx].float()
    ker_s = kernel_state[slot_idx].float()
    abs_diff_s = (ref_s - ker_s).abs()
    print(f"  State max abs diff:  {abs_diff_s.max().item():.4e}")
    print(f"  State mean abs diff: {abs_diff_s.mean().item():.4e}")

    output_ok = torch.allclose(ref_o, ker_o, atol=atol, rtol=rtol)
    state_ok = torch.allclose(ref_s, ker_s, atol=atol, rtol=rtol)

    if output_ok and state_ok:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        if not output_ok:
            print(f"\n✗ FAILED: output mismatch (atol={atol}, rtol={rtol})")
        if not state_ok:
            print(f"\n✗ FAILED: state mismatch (atol={atol}, rtol={rtol})")

    return output_ok and state_ok


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_mamba_ssu_decode_h128_d64_s128_ng8(batch_size: int):
    """Pytest parametrized test for mamba_ssu_decode across batch sizes."""
    device = "cuda"
    atol, rtol = 1e-2, 1e-2

    definition = load_definition("mamba_ssu_decode_h128_d64_s128_ng8")
    run = compile_reference(definition.reference)

    inputs = generate_inputs(batch_size=batch_size, device=device)

    # Reference from definition
    ref_output, ref_state = run(
        inputs["state"].clone(),
        inputs["x"].clone(),
        inputs["dt"].clone(),
        inputs["A"].clone(),
        inputs["B"].clone(),
        inputs["C"].clone(),
        inputs["D"].clone(),
        inputs["dt_bias"].clone(),
        inputs["state_batch_indices"].clone(),
    )

    # FlashInfer kernel
    kernel_output, kernel_state = run_flashinfer_kernel(inputs)

    # Check outputs
    torch.testing.assert_close(
        kernel_output.float(),
        ref_output.float(),
        atol=atol,
        rtol=rtol,
        msg=f"Output mismatch for batch_size={batch_size}",
    )

    # Check states at active slots
    slot_idx = inputs["state_batch_indices"]
    torch.testing.assert_close(
        kernel_state[slot_idx].float(),
        ref_state[slot_idx].float(),
        atol=atol,
        rtol=rtol,
        msg=f"State mismatch for batch_size={batch_size}",
    )

    print(f"✓ mamba_ssu_decode_h128_d64_s128_ng8 passed (batch_size={batch_size})")


def main():
    """Run standalone tests."""
    print("Testing Mamba2 SSU Decode Reference Implementation")
    print("Definition: mamba_ssu_decode_h128_d64_s128_ng8 (NemotronH-8B, TP=1)")

    test_configs = [1, 4, 16, 64, 256]

    passed = 0
    total = len(test_configs)

    for batch_size in test_configs:
        try:
            if test_correctness(batch_size):
                passed += 1
        except Exception as e:
            print(f"✗ batch_size={batch_size} failed with exception: {e}")
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
