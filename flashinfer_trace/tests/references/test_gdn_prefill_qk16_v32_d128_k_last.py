"""
Test GDN prefill k-last reference implementation against FlashInfer kernel.

Run with:
    pytest test_gdn_prefill_qk16_v32_d128_k_last.py -v
    python test_gdn_prefill_qk16_v32_d128_k_last.py
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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
    """Compute g and beta from raw parameters.

    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)
    """
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return g, beta


# Load definition and compile reference
definition = load_definition("gdn_prefill_qk16_v32_d128_k_last")
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

    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
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

    # Output comparison metrics
    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    abs_diff_o = torch.abs(ref_o_f32 - fi_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    rel_diff_o = abs_diff_o / (torch.abs(ref_o_f32) + 1e-10)
    max_rel_diff_o = rel_diff_o.max().item()
    mean_rel_diff_o = rel_diff_o.mean().item()

    ref_flat = ref_o_f32.reshape(-1)
    fi_flat = fi_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), fi_flat.unsqueeze(0)).item()

    mse_o = ((ref_o_f32 - fi_o_f32) ** 2).mean().item()

    # State comparison metrics
    abs_diff_s = torch.abs(ref_new_state - fi_new_state)
    max_abs_diff_s = abs_diff_s.max().item()
    mean_abs_diff_s = abs_diff_s.mean().item()

    rel_diff_s = abs_diff_s / (torch.abs(ref_new_state) + 1e-10)
    max_rel_diff_s = rel_diff_s.max().item()
    mean_rel_diff_s = rel_diff_s.mean().item()

    ref_state_flat = ref_new_state.reshape(-1)
    fi_state_flat = fi_new_state.reshape(-1)
    cosine_sim_s = F.cosine_similarity(
        ref_state_flat.unsqueeze(0), fi_state_flat.unsqueeze(0)
    ).item()

    mse_s = ((ref_new_state - fi_new_state) ** 2).mean().item()

    print(f"\nBatch={batch_size}, SeqLen={seq_len}")
    print("\nOutput tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_o:.6e}")
    print(f"  Max relative difference: {max_rel_diff_o:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_o:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_o:.6e}")
    print(f"  Cosine similarity: {cosine_sim_o:.6f}")
    print(f"  MSE: {mse_o:.6e}")

    print("\nState tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_s:.6e}")
    print(f"  Max relative difference: {max_rel_diff_s:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_s:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_s:.6e}")
    print(f"  Cosine similarity: {cosine_sim_s:.6f}")
    print(f"  MSE: {mse_s:.6e}")

    output_max_err = max_abs_diff_o
    state_max_err = max_abs_diff_s

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


@requires_cuda
@requires_sm90_only
def test_gdn_prefill_with_initial_state():
    """Test GDN prefill kernel with non-zero initial state."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    head_size = 128
    num_sab_heads = max(num_q_heads, num_v_heads)

    batch_size = 2
    seq_len = 32
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

    # Output comparison metrics
    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    abs_diff_o = torch.abs(ref_o_f32 - fi_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    rel_diff_o = abs_diff_o / (torch.abs(ref_o_f32) + 1e-10)
    max_rel_diff_o = rel_diff_o.max().item()
    mean_rel_diff_o = rel_diff_o.mean().item()

    ref_flat = ref_o_f32.reshape(-1)
    fi_flat = fi_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), fi_flat.unsqueeze(0)).item()

    mse_o = ((ref_o_f32 - fi_o_f32) ** 2).mean().item()

    # State comparison metrics
    abs_diff_s = torch.abs(ref_new_state - fi_new_state)
    max_abs_diff_s = abs_diff_s.max().item()
    mean_abs_diff_s = abs_diff_s.mean().item()

    rel_diff_s = abs_diff_s / (torch.abs(ref_new_state) + 1e-10)
    max_rel_diff_s = rel_diff_s.max().item()
    mean_rel_diff_s = rel_diff_s.mean().item()

    ref_state_flat = ref_new_state.reshape(-1)
    fi_state_flat = fi_new_state.reshape(-1)
    cosine_sim_s = F.cosine_similarity(
        ref_state_flat.unsqueeze(0), fi_state_flat.unsqueeze(0)
    ).item()

    mse_s = ((ref_new_state - fi_new_state) ** 2).mean().item()

    print(f"\nWith initial state:")
    print("\nOutput tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_o:.6e}")
    print(f"  Max relative difference: {max_rel_diff_o:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_o:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_o:.6e}")
    print(f"  Cosine similarity: {cosine_sim_o:.6f}")
    print(f"  MSE: {mse_o:.6e}")

    print("\nState tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_s:.6e}")
    print(f"  Max relative difference: {max_rel_diff_s:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_s:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_s:.6e}")
    print(f"  Cosine similarity: {cosine_sim_s:.6f}")
    print(f"  MSE: {mse_s:.6e}")

    output_max_err = max_abs_diff_o
    state_max_err = max_abs_diff_s

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


@requires_cuda
@requires_sm90_only
def test_gdn_prefill_variable_seqlen():
    """Test GDN prefill kernel with variable sequence lengths."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    head_size = 128
    num_sab_heads = max(num_q_heads, num_v_heads)

    seq_lens = [16, 32, 8, 64]
    total_seq_len = sum(seq_lens)

    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device=device)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device=device)

    # Raw gate parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(total_seq_len, num_sab_heads, dtype=dtype, device=device)

    cu_seqlens_list = [0]
    for sl in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int64, device=device)

    scale = 1.0 / math.sqrt(head_size)

    ref_result = reference_gdn_prefill(q, k, v, None, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output, ref_new_state = ref_result

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

    # Output comparison metrics
    ref_o_f32 = ref_output.float()
    fi_o_f32 = fi_output.float()

    abs_diff_o = torch.abs(ref_o_f32 - fi_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    rel_diff_o = abs_diff_o / (torch.abs(ref_o_f32) + 1e-10)
    max_rel_diff_o = rel_diff_o.max().item()
    mean_rel_diff_o = rel_diff_o.mean().item()

    ref_flat = ref_o_f32.reshape(-1)
    fi_flat = fi_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), fi_flat.unsqueeze(0)).item()

    mse_o = ((ref_o_f32 - fi_o_f32) ** 2).mean().item()

    # State comparison metrics
    abs_diff_s = torch.abs(ref_new_state - fi_new_state)
    max_abs_diff_s = abs_diff_s.max().item()
    mean_abs_diff_s = abs_diff_s.mean().item()

    rel_diff_s = abs_diff_s / (torch.abs(ref_new_state) + 1e-10)
    max_rel_diff_s = rel_diff_s.max().item()
    mean_rel_diff_s = rel_diff_s.mean().item()

    ref_state_flat = ref_new_state.reshape(-1)
    fi_state_flat = fi_new_state.reshape(-1)
    cosine_sim_s = F.cosine_similarity(
        ref_state_flat.unsqueeze(0), fi_state_flat.unsqueeze(0)
    ).item()

    mse_s = ((ref_new_state - fi_new_state) ** 2).mean().item()

    print(f"\nVariable seqlens={seq_lens}:")
    print("\nOutput tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_o:.6e}")
    print(f"  Max relative difference: {max_rel_diff_o:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_o:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_o:.6e}")
    print(f"  Cosine similarity: {cosine_sim_o:.6f}")
    print(f"  MSE: {mse_o:.6e}")

    print("\nState tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_s:.6e}")
    print(f"  Max relative difference: {max_rel_diff_s:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_s:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_s:.6e}")
    print(f"  Cosine similarity: {cosine_sim_s:.6f}")
    print(f"  MSE: {mse_s:.6e}")

    output_max_err = max_abs_diff_o
    state_max_err = max_abs_diff_s

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


if __name__ == "__main__":
    pytest.main(sys.argv)
