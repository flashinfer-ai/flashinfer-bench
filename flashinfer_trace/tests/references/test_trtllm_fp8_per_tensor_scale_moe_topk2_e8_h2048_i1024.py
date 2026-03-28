"""
Reference test for trtllm_fp8_per_tensor_scale_moe_topk2_e8_h2048_i1024.

Validates the definition's reference implementation against the FlashInfer
trtllm_fp8_per_tensor_scale_moe kernel.

Kernel: FP8 per-tensor scale MoE (TensorRT-LLM style)
  - hidden_states and weights are FP8 (float8_e4m3fn)
  - Per-expert scalar output scales compensate for quantization
  - Routing: TopK -> Renormalize via Softmax (routing_method_type=1)
  - Config: E=8, H=2048, I=1024, TOP_K=2

Requires SM100 (NVIDIA Blackwell) GPU.

Note: The kernel requires weights pre-shuffled with reorder_rows_for_gated_act_gemm
(for gemm1) and shuffle_matrix_a (for both gemm1 and gemm2). The reference run()
operates on unshuffled weights. Weight shuffling is applied in _run_flashinfer_kernel.
"""

import torch
import pytest
from pathlib import Path

# -----------------------------------------------------------------------
# Constants (must match the definition JSON)
# -----------------------------------------------------------------------
E = 8
H = 2048
I = 1024
TOP_K = 2
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
EPILOGUE_TILE_M = 128

TRACE_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_JSONL_PATH = (
    TRACE_ROOT
    / "workloads"
    / "moe"
    / "trtllm_fp8_per_tensor_scale_moe_topk2_e8_h2048_i1024.jsonl"
)


# -----------------------------------------------------------------------
# Reference implementation (mirrors definition JSON reference field)
# -----------------------------------------------------------------------
@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
):
    """
    FP8 per-tensor scale MoE reference implementation.

    Routing (routing_method_type=1: TopK -> Renormalize via Softmax):
        Select top_k=2 experts per token by raw logit value.
        Compute routing weights via softmax over selected experts' logits.

    Compute per selected expert:
        GEMM1: fp8_hidden @ fp8_w13.T   -> [Tk, 2I] (raw FP8 dot products)
        Scale: act  = g1[:, :I] * output1_scales_scalar[e]
               gate = g1[:, I:] * output1_scales_gate_scalar[e]
        SwiGLU: c = silu(gate) * act
        GEMM2: c @ fp8_w2.T            -> [Tk, H]
        Scale: o = g2 * output2_scales_scalar[e]
        Accumulate: output += o * routing_weight
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    assert routing_logits.shape == (T, E)
    assert hidden_states.shape == (T, H)
    assert gemm1_weights.shape == (E, 2 * I, H)
    assert gemm2_weights.shape == (E, H, I)

    logits_f32 = routing_logits.to(torch.float32)
    topk_logits, topk_idx = torch.topk(logits_f32, k=TOP_K, dim=-1)  # [T, 2]
    # Renormalize: softmax over selected logits
    probs_topk = torch.softmax(topk_logits, dim=-1)  # [T, 2]

    # Build a full [T, E] weight tensor (zero for non-selected experts)
    probs = torch.zeros(T, E, dtype=torch.float32, device=device)
    probs.scatter_(1, topk_idx, probs_topk)

    output = torch.zeros(T, H, dtype=torch.float32, device=device)

    for e in range(E):
        sel_mask = (topk_idx == e).any(dim=-1)
        if not sel_mask.any():
            continue

        token_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
        w = probs[token_idx, e]

        h_e = hidden_states[token_idx].float()
        w1_e = gemm1_weights[e].float()
        g1 = h_e @ w1_e.T

        act  = g1[:, :I] * output1_scales_scalar[e].float()
        gate = g1[:, I:] * output1_scales_gate_scalar[e].float()
        c = torch.nn.functional.silu(gate) * act

        w2_e = gemm2_weights[e].float()
        o = (c @ w2_e.T) * output2_scales_scalar[e].float()

        output.index_add_(0, token_idx, o * w.unsqueeze(1))

    return output.to(torch.bfloat16)


# -----------------------------------------------------------------------
# Input generation helpers
# -----------------------------------------------------------------------
def _quantize_fp8_per_tensor(tensor: torch.Tensor):
    """Quantize tensor to FP8 with per-tensor scale. Returns (fp8_tensor, dequant_scale).
    dequant_scale = amax/FP8_MAX  (fp32 ≈ fp8 * dequant_scale)
    """
    amax = tensor.abs().max().float().clamp(min=1e-6)
    scale = amax / FP8_MAX
    fp8 = (tensor.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8, scale.item()


@torch.no_grad()
def generate_random_inputs(seq_len: int, device: str = "cuda") -> dict:
    """
    Generate FP8-quantized inputs with proper per-tensor scales.

    Scale formula (c_global_sf = 1.0):
      output1_scales_scalar[e]      = hs_dequant * w1_dequant[e]
      output1_scales_gate_scalar[e] = hs_dequant * w1_dequant[e]
      output2_scales_scalar[e]      = w2_dequant[e]

    The kernel requires weights pre-shuffled; raw (unshuffled) weights are
    returned as "gemm1_weights_raw" / "gemm2_weights_raw" for the reference.
    """
    torch.manual_seed(42)

    hidden_f32 = torch.randn(seq_len, H, dtype=torch.float32, device=device) * 0.1
    routing_logits = torch.randn(seq_len, E, dtype=torch.bfloat16, device=device)

    hidden_fp8, hidden_scale = _quantize_fp8_per_tensor(hidden_f32)

    w1_fp8_list, w2_fp8_list = [], []
    out1_scales, out1_gate_scales, out2_scales = [], [], []

    for e in range(E):
        w1_f32 = torch.randn(2 * I, H, dtype=torch.float32, device=device) * 0.1
        w2_f32 = torch.randn(H, I, dtype=torch.float32, device=device) * 0.1

        w1_fp8, w1_scale = _quantize_fp8_per_tensor(w1_f32)
        w2_fp8, w2_scale = _quantize_fp8_per_tensor(w2_f32)

        w1_fp8_list.append(w1_fp8)
        w2_fp8_list.append(w2_fp8)
        # output scale = hidden_dequant * weight_dequant (combined dequant scale)
        out1_scales.append(hidden_scale * w1_scale)
        out1_gate_scales.append(hidden_scale * w1_scale)
        # For GEMM2: c_global_sf = 1.0, so output2_scale = w2_dequant
        out2_scales.append(w2_scale)

    gemm1_raw = torch.stack(w1_fp8_list)    # [E, 2I, H]
    gemm2_raw = torch.stack(w2_fp8_list)    # [E, H, I]

    return {
        "routing_logits": routing_logits,
        "hidden_states": hidden_fp8,
        "gemm1_weights_raw": gemm1_raw,
        "gemm2_weights_raw": gemm2_raw,
        "output1_scales_scalar": torch.tensor(out1_scales, dtype=torch.float32, device=device),
        "output1_scales_gate_scalar": torch.tensor(out1_gate_scales, dtype=torch.float32, device=device),
        "output2_scales_scalar": torch.tensor(out2_scales, dtype=torch.float32, device=device),
        "local_num_experts": E,
    }


def _shuffle_weights(gemm1_raw: torch.Tensor, gemm2_raw: torch.Tensor):
    """
    Apply the row permutations expected by the TRT-LLM MoE kernel.

    For gemm1: reorder_rows_for_gated_act_gemm (interleave gate/act rows) +
               shuffle_matrix_a (warp-layout permutation for MMA epilogue)
    For gemm2: shuffle_matrix_a only
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    num_experts = gemm1_raw.shape[0]
    cache = {}
    gemm1_shuffled = []
    gemm2_shuffled = []

    for e in range(num_experts):
        w1_e = gemm1_raw[e]   # [2I, H]
        w2_e = gemm2_raw[e]   # [H, I]

        perm1 = _maybe_get_cached_w3_w1_permute_indices(cache, w1_e, EPILOGUE_TILE_M)
        gemm1_shuffled.append(w1_e[perm1.to(w1_e.device)])

        perm2 = get_w2_permute_indices_with_cache(cache, w2_e, EPILOGUE_TILE_M)
        gemm2_shuffled.append(w2_e[perm2.to(w2_e.device)])

    return torch.stack(gemm1_shuffled), torch.stack(gemm2_shuffled)


# -----------------------------------------------------------------------
# Kernel runner
# -----------------------------------------------------------------------
def _run_flashinfer_kernel(inputs: dict, seq_len: int) -> torch.Tensor:
    from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe

    # Shuffle weights into the layout the kernel expects
    gemm1_kernel, gemm2_kernel = _shuffle_weights(
        inputs["gemm1_weights_raw"], inputs["gemm2_weights_raw"]
    )

    return trtllm_fp8_per_tensor_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        gemm1_weights=gemm1_kernel,
        output1_scales_scalar=inputs["output1_scales_scalar"],
        output1_scales_gate_scalar=inputs["output1_scales_gate_scalar"],
        gemm2_weights=gemm2_kernel,
        output2_scales_scalar=inputs["output2_scales_scalar"],
        num_experts=E,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=inputs["local_num_experts"],
        routed_scaling_factor=None,
        use_routing_scales_on_input=False,
        routing_method_type=1,
        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
    )


# -----------------------------------------------------------------------
# Comparison helper
# -----------------------------------------------------------------------
def _compare(ref_out: torch.Tensor, fi_out: torch.Tensor, atol: float, rtol: float, percent: float) -> bool:
    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    rel_diff = abs_diff / (fi_f32.abs() + 1e-8)

    print(f"  Max abs diff:  {abs_diff.max().item():.4e}")
    print(f"  Mean abs diff: {abs_diff.mean().item():.4e}")
    print(f"  Max rel diff:  {rel_diff.max().item():.4e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    print(f"  Cosine similarity: {cos_sim:.6f}")

    left = abs_diff
    right = atol + rtol * fi_f32.abs()
    hit_ratio = (left <= right).float().mean().item()
    print(f"  Hit ratio: {hit_ratio * 100:.2f}% (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


# -----------------------------------------------------------------------
# Pytest tests
# -----------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reference_correctness_fp8_per_tensor():
    """
    Validate reference vs FlashInfer trtllm_fp8_per_tensor_scale_moe kernel.
    Skips on non-SM100 GPUs (requires Blackwell architecture).
    """
    cc = torch.cuda.get_device_capability()
    if cc[0] < 10:
        pytest.skip(f"trtllm_fp8_per_tensor_scale_moe requires SM100+ (got SM{cc[0]}{cc[1]})")

    device = "cuda"
    atol, rtol, percent = 1e-1, 2e-1, 0.85

    configs = [1, 4, 8, 16, 32, 64]
    total, passed = len(configs), 0

    for seq_len in configs:
        print(f"\n--- seq_len={seq_len} ---")
        torch.manual_seed(42)
        inputs = generate_random_inputs(seq_len, device=device)

        ref_out = run(
            routing_logits=inputs["routing_logits"],
            hidden_states=inputs["hidden_states"],
            gemm1_weights=inputs["gemm1_weights_raw"],
            output1_scales_scalar=inputs["output1_scales_scalar"],
            output1_scales_gate_scalar=inputs["output1_scales_gate_scalar"],
            gemm2_weights=inputs["gemm2_weights_raw"],
            output2_scales_scalar=inputs["output2_scales_scalar"],
        )

        fi_out = _run_flashinfer_kernel(inputs, seq_len)
        ok = _compare(ref_out, fi_out, atol, rtol, percent)
        passed += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}")

    print(f"\nSummary: {passed}/{total} configurations passed")
    assert passed == total, f"Only {passed}/{total} configurations passed"


def main():
    """Run tests standalone (outside pytest)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    cc = torch.cuda.get_device_capability()
    if cc[0] < 10:
        print(f"SM{cc[0]}{cc[1]} detected — trtllm_fp8_per_tensor_scale_moe requires SM100+, skipping")
        return

    print("Testing FP8 Per-Tensor Scale MoE: reference vs FlashInfer kernel")
    device = "cuda"
    atol, rtol, percent = 1e-1, 2e-1, 0.85

    configs = [1, 4, 8, 16, 32, 64, 128, 256]
    total, passed = len(configs), 0

    for seq_len in configs:
        print(f"\n{'='*60}")
        print(f"seq_len={seq_len}")
        torch.manual_seed(42)
        inputs = generate_random_inputs(seq_len, device=device)

        ref_out = run(
            routing_logits=inputs["routing_logits"],
            hidden_states=inputs["hidden_states"],
            gemm1_weights=inputs["gemm1_weights_raw"],
            output1_scales_scalar=inputs["output1_scales_scalar"],
            output1_scales_gate_scalar=inputs["output1_scales_gate_scalar"],
            gemm2_weights=inputs["gemm2_weights_raw"],
            output2_scales_scalar=inputs["output2_scales_scalar"],
        )
        fi_out = _run_flashinfer_kernel(inputs, seq_len)
        ok = _compare(ref_out, fi_out, atol, rtol, percent)
        passed += int(ok)
        print(f"Result: {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} passed")


if __name__ == "__main__":
    main()
