"""
Tests for GDN (Gated Delta Net) prefill kernel.

Compares reference implementation against FlashInfer's chunk_gated_delta_rule kernel.
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def get_cuda_capability():
    """Get CUDA compute capability."""
    if torch.cuda.device_count() == 0:
        return (0, 0)
    return torch.cuda.get_device_capability(0)


requires_sm90 = pytest.mark.skipif(
    get_cuda_capability()[0] < 9,
    reason="GDN prefill kernel requires SM90 (Hopper) or later",
)

requires_cuda = pytest.mark.skipif(
    torch.cuda.device_count() == 0,
    reason="CUDA devices not available",
)


def load_reference_impl(definition_path: Path):
    """Load reference implementation from definition JSON."""
    with open(definition_path, "r") as f:
        definition = json.load(f)
    ref_code = definition["reference"]
    namespace = {}
    exec(ref_code, namespace)
    return namespace["run"]


def compute_gates(A_log, a, dt_bias, b):
    """Compute g and beta from raw parameters.

    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)
    """
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return g, beta


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def reference_gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net reference implementation (k-last layout).

    State layout: [H, V, K] (k-last, K dimension at the end)
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    g, beta = compute_gates(A_log, a, dt_bias, b)

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)  # [H,V,K] -> [H,K,V]
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return {"output": output, "new_state": new_state}


@requires_cuda
@requires_sm90
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

    # Reference uses raw params
    ref_result = reference_gdn_prefill(q, k, v, None, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output = ref_result["output"]
    ref_new_state = ref_result["new_state"]

    # FlashInfer uses pre-computed g/beta
    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    output_diff = (ref_output.float() - fi_output.float()).abs()
    output_max_err = output_diff.max().item()
    output_mean_err = output_diff.mean().item()

    state_diff = (ref_new_state - fi_new_state).abs()
    state_max_err = state_diff.max().item()
    state_mean_err = state_diff.mean().item()

    print(f"\nBatch={batch_size}, SeqLen={seq_len}")
    print(f"  Output: max_err={output_max_err:.6f}, mean_err={output_mean_err:.6f}")
    print(f"  State:  max_err={state_max_err:.6f}, mean_err={state_mean_err:.6f}")

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


@requires_cuda
@requires_sm90
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
    state = torch.randn(
        batch_size, num_sab_heads, head_size, head_size,
        dtype=torch.float32, device=device
    ) * 0.1

    scale = 1.0 / math.sqrt(head_size)

    ref_result = reference_gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output = ref_result["output"]
    ref_new_state = ref_result["new_state"]

    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    output_diff = (ref_output.float() - fi_output.float()).abs()
    output_max_err = output_diff.max().item()

    state_diff = (ref_new_state - fi_new_state).abs()
    state_max_err = state_diff.max().item()

    print(f"\nWith initial state:")
    print(f"  Output max_err={output_max_err:.6f}")
    print(f"  State max_err={state_max_err:.6f}")

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


@requires_cuda
@requires_sm90
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
    ref_output = ref_result["output"]
    ref_new_state = ref_result["new_state"]

    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    output_diff = (ref_output.float() - fi_output.float()).abs()
    output_max_err = output_diff.max().item()

    state_diff = (ref_new_state - fi_new_state).abs()
    state_max_err = state_diff.max().item()

    print(f"\nVariable seqlens={seq_lens}:")
    print(f"  Output max_err={output_max_err:.6f}")
    print(f"  State max_err={state_max_err:.6f}")

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


@requires_cuda
@requires_sm90
def test_gdn_prefill_definition_reference():
    """Test that the definition JSON reference implementation matches the kernel."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    definition_path = Path(__file__).parent.parent.parent / "definitions" / "gdn" / "gdn_prefill_qk16_v32_d128_k_last.json"

    if not definition_path.exists():
        pytest.skip(f"Definition file not found: {definition_path}")

    ref_run = load_reference_impl(definition_path)

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

    scale = 1.0 / math.sqrt(head_size)

    # Definition reference uses: state, A_log, a, dt_bias, b, cu_seqlens, scale
    ref_result = ref_run(q, k, v, None, A_log, a, dt_bias, b, cu_seqlens, scale)
    ref_output = ref_result["output"]
    ref_new_state = ref_result["new_state"]

    g, beta = compute_gates(A_log, a, dt_bias, b)
    fi_output, fi_new_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    output_diff = (ref_output.float() - fi_output.float()).abs()
    output_max_err = output_diff.max().item()

    state_diff = (ref_new_state - fi_new_state).abs()
    state_max_err = state_diff.max().item()

    print(f"\nDefinition reference test:")
    print(f"  Output max_err={output_max_err:.6f}")
    print(f"  State max_err={state_max_err:.6f}")

    atol = 0.1
    assert output_max_err < atol, f"Output max error {output_max_err} exceeds tolerance"
    assert state_max_err < atol, f"State max error {state_max_err} exceeds tolerance"


if __name__ == "__main__":
    pytest.main(sys.argv)
