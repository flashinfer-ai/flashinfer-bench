import os

"""
Test GDN prefill k-last reference implementation against FlashInfer kernel.
Configuration: num_q_heads=8, num_v_heads=16, head_size=128 (Qwen3.5 TP=2).

Run with:
    pytest test_gdn_prefill_qk8_v16_d128_k_last.py -v
    python test_gdn_prefill_qk8_v16_d128_k_last.py
"""

import math
import sys
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


def get_cuda_capability():
    """Get CUDA compute capability."""
    if torch.cuda.device_count() == 0:
        return (0, 0)
    return torch.cuda.get_device_capability(0)


requires_sm90_only = pytest.mark.skipif(
    get_cuda_capability()[0] != 9,
    reason="GDN prefill kernel only supports SM90 (Hopper), not SM80 or SM100+",
)

requires_cuda = pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA devices not available"
)


def compute_gates(A_log, a, dt_bias, b):
    """Compute g and beta from raw parameters."""
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return g, beta


# Load definition and compile reference
definition = load_definition("gdn_prefill_qk8_v16_d128_k_last")
reference_gdn_prefill = compile_reference(definition.reference)


@requires_cuda
@requires_sm90_only
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
def test_gdn_prefill_correctness(batch_size: int, seq_len: int):
    """Test GDN prefill kernel correctness against reference implementation."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 8
    num_k_heads = 8
    num_v_heads = 16
    head_size = 128
    num_sab_heads = max(num_q_heads, num_v_heads)

    total_seq_len = batch_size * seq_len

    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device=device)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device=device)

    # Raw gate parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device=device
    )

    scale = 1.0 / math.sqrt(head_size)

    # Reference from definition
    ref_result = reference_gdn_prefill(q, k, v, None, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output, ref_new_state = ref_result

    # FlashInfer uses pre-computed g/beta
    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    # Output comparison
    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    abs_diff_o = torch.abs(ref_o_f32 - fi_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    ref_flat = ref_o_f32.reshape(-1)
    fi_flat = fi_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), fi_flat.unsqueeze(0)).item()

    # State comparison
    abs_diff_s = torch.abs(ref_new_state - fi_new_state)
    max_abs_diff_s = abs_diff_s.max().item()

    print(f"\nBatch={batch_size}, SeqLen={seq_len}")
    print(f"  Output max abs diff: {max_abs_diff_o:.6e}, cosine sim: {cosine_sim_o:.6f}")
    print(f"  State max abs diff: {max_abs_diff_s:.6e}")

    atol = 0.1
    assert max_abs_diff_o < atol, f"Output max error {max_abs_diff_o} exceeds tolerance"
    assert max_abs_diff_s < atol, f"State max error {max_abs_diff_s} exceeds tolerance"


@requires_cuda
@requires_sm90_only
def test_gdn_prefill_with_initial_state():
    """Test GDN prefill kernel with non-zero initial state."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 8
    num_k_heads = 8
    num_v_heads = 16
    head_size = 128
    num_sab_heads = max(num_q_heads, num_v_heads)

    batch_size = 2
    seq_len = 32
    total_seq_len = batch_size * seq_len

    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device=device)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device=device)

    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device=device
    )

    # Non-zero initial state (k-last layout [N, H, V, K])
    state = (
        torch.randn(
            batch_size, num_sab_heads, head_size, head_size, dtype=torch.float32, device=device
        )
        * 0.1
    )

    scale = 1.0 / math.sqrt(head_size)

    ref_result = reference_gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output, ref_new_state = ref_result

    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    output_max_err = torch.abs(ref_o_f32 - fi_o_f32).max().item()
    state_max_err = torch.abs(ref_new_state - fi_new_state).max().item()

    print(
        f"\nWith initial state: output_max_err={output_max_err:.6e}, state_max_err={state_max_err:.6e}"
    )

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


def main():
    """Run tests manually."""
    print("Testing GDN Prefill qk8_v16 K-Last Reference Implementation")

    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 8
    num_k_heads = 8
    num_v_heads = 16
    head_size = 128
    num_sab_heads = max(num_q_heads, num_v_heads)

    test_configs = [(1, 16), (2, 32), (4, 64)]

    passed = 0
    total = len(test_configs)

    for batch_size, seq_len in test_configs:
        try:
            total_seq_len = batch_size * seq_len
            q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device=device)
            k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device=device)
            k = F.normalize(k, p=2.0, dim=-1)
            v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device=device)

            A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
            a = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)
            dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
            b = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)

            cu_seqlens = torch.arange(
                0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device=device
            )
            scale = 1.0 / math.sqrt(head_size)

            ref_output, ref_state = reference_gdn_prefill(
                q, k, v, None, A_log, a, dt_bias, b, cu_seqlens, scale
            )

            g, beta = compute_gates(A_log, a, dt_bias, b)
            fi_output, fi_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

            output_err = torch.abs(ref_output.float() - fi_output.float()).max().item()
            state_err = torch.abs(ref_state - fi_state).max().item()

            print(
                f"\nBatch={batch_size}, SeqLen={seq_len}: output_err={output_err:.6e}, state_err={state_err:.6e}"
            )
            if output_err < 0.1 and state_err < 0.1:
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
