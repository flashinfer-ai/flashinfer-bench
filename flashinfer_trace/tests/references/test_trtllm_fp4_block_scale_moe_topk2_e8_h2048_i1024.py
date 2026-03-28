"""
Reference test for trtllm_fp4_block_scale_moe_topk2_e8_h2048_i1024.

Tests the TensorRT-LLM FP4 block scale (MxFP4 weights, BF16 activations) MoE kernel
against the float32 reference implementation. Targets SM100 (Blackwell) GPUs.

Kernel: flashinfer.fused_moe.trtllm_fp4_block_scale_moe
  - Routing method: Renormalize (TopK -> Softmax), routing_method_type=1
  - Weight format: MxFP4 (use_ue8m0=True, sf_vec_size=32), pre-shuffled + scale-interleaved
  - Hidden states: BF16 (no activation quantization, c_global_sf=1.0)
  - Output scales: out1 = out1g = 1/gemm1_global[e], out2 = 1/gemm2_global[e]
"""

import sys
import pytest
import torch

sys.path.insert(0, "/home/averyh/flashinfer-bench/tmp/flashinfer")

# ── Fixed geometry ────────────────────────────────────────────────────────────
num_experts = 8
top_k = 2
hidden_size = 2048
intermediate_size = 1024
SF_VEC = 32         # MxFP4 block size (one scale per 32 elements)
TILE_M = 128        # epilogue tile size for weight shuffling

device = "cuda"


# ── Float32 reference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
) -> torch.Tensor:
    """Float32 reference: Renormalize routing + SwiGLU MoE."""
    E = num_experts
    I = intermediate_size
    T = routing_logits.shape[0]
    dev = routing_logits.device

    # Routing: TopK → Softmax (Renormalize)
    logits_f32 = routing_logits.to(torch.float32)
    topk_logits, topk_idx = torch.topk(logits_f32, k=top_k, dim=-1)
    probs_topk = torch.softmax(topk_logits, dim=-1)
    probs = torch.zeros(T, E, dtype=torch.float32, device=dev)
    probs.scatter_(1, topk_idx, probs_topk)

    A = hidden_states.to(torch.float32)
    W13 = gemm1_weights.to(torch.float32)
    W2 = gemm2_weights.to(torch.float32)

    output = torch.zeros((T, hidden_size), dtype=torch.float32, device=dev)

    for e in range(E):
        tok_mask = (topk_idx == e).any(dim=1)
        if not tok_mask.any():
            continue
        tok_idx = torch.nonzero(tok_mask, as_tuple=False).squeeze(1)
        A_e = A[tok_idx]       # [Tk, H]
        g1 = A_e @ W13[e].t() # [Tk, 2I]
        act = g1[:, :I]
        gate = g1[:, I:]
        c = torch.nn.functional.silu(gate) * act  # SwiGLU
        o = c @ W2[e].t()      # [Tk, H]
        w_tok = probs[tok_idx][:, e]
        output.index_add_(0, tok_idx, o * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


# ── Input generation ──────────────────────────────────────────────────────────

def generate_random_inputs(seq_len: int, dev: str = "cuda"):
    """Generate random float32 inputs and return them along with random weights."""
    E = num_experts
    H = hidden_size
    I = intermediate_size

    routing_logits = torch.randn(seq_len, E, dtype=torch.bfloat16, device=dev)
    hidden_f32 = torch.randn(seq_len, H, dtype=torch.float32, device=dev) * 0.1
    gemm1_f32 = torch.randn(E, 2 * I, H, dtype=torch.float32, device=dev) * 0.01
    gemm2_f32 = torch.randn(E, H, I, dtype=torch.float32, device=dev) * 0.01

    return routing_logits, hidden_f32, gemm1_f32, gemm2_f32


# ── Weight quantization and shuffling ────────────────────────────────────────

def _quantize_and_shuffle_weights(gemm1_f32: torch.Tensor, gemm2_f32: torch.Tensor):
    """
    Quantize weights to MxFP4 (use_ue8m0=True) and apply the required shuffles
    for the TRT-LLM FP4 block scale MoE kernel.

    Returns:
        G1K:   [E, 2I, H//2] uint8 - shuffled gemm1 packed weights
        G1SK:  [E, 2I, H//SF_VEC] float8_e4m3fn - shuffled+interleaved gemm1 scales
        G2K:   [E, H, I//2] uint8 - shuffled gemm2 packed weights
        G2SK:  [E, H, I//SF_VEC] float8_e4m3fn - shuffled+interleaved gemm2 scales
        gemm1_globals: [E] float32 - per-expert global scales for gemm1
        gemm2_globals: [E] float32 - per-expert global scales for gemm2
    """
    from flashinfer import fp4_quantize
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    E = gemm1_f32.shape[0]
    USE_UE8M0 = True  # MxFP4 uses UE8M0 scale format

    cache = {}
    gemm1_fp4_shuffled = []
    gemm1_sf_shuffled = []
    gemm2_fp4_shuffled = []
    gemm2_sf_shuffled = []
    gemm1_globals = []
    gemm2_globals = []

    for e in range(E):
        # Global scale = 1.0 for MxFP4 (use_ue8m0=True)
        g1_global = torch.tensor(1.0, dtype=torch.float32, device=device)
        g2_global = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Quantize to MxFP4 (linear layout)
        g1_fp4, g1_sf = fp4_quantize(
            gemm1_f32[e].to(torch.bfloat16), g1_global, SF_VEC, USE_UE8M0, False
        )
        g2_fp4, g2_sf = fp4_quantize(
            gemm2_f32[e].to(torch.bfloat16), g2_global, SF_VEC, USE_UE8M0, False
        )

        # Shuffle gemm1 weights and scales
        g1_u8 = g1_fp4.view(torch.uint8)
        g1_sf_u8 = g1_sf.view(torch.uint8)

        perm1 = _maybe_get_cached_w3_w1_permute_indices(cache, g1_u8, TILE_M)
        g1_w_shuffled = g1_u8[perm1.to(device)].contiguous()

        perm1_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache, g1_sf_u8, TILE_M, num_elts_per_sf=16
        )
        g1_sf_perm = g1_sf_u8[perm1_sf.to(device)].contiguous()
        g1_sf_interleaved = block_scale_interleave(g1_sf_perm)

        gemm1_fp4_shuffled.append(g1_w_shuffled)
        gemm1_sf_shuffled.append(g1_sf_interleaved)
        gemm1_globals.append(g1_global)

        # Shuffle gemm2 weights and scales
        g2_u8 = g2_fp4.view(torch.uint8)
        g2_sf_u8 = g2_sf.view(torch.uint8)

        perm2 = get_w2_permute_indices_with_cache(cache, g2_u8, TILE_M)
        g2_w_shuffled = g2_u8[perm2.to(device)].contiguous()

        perm2_sf = get_w2_permute_indices_with_cache(
            cache, g2_sf_u8, TILE_M, num_elts_per_sf=16
        )
        g2_sf_perm = g2_sf_u8[perm2_sf.to(device)].contiguous()
        g2_sf_interleaved = block_scale_interleave(g2_sf_perm)

        gemm2_fp4_shuffled.append(g2_w_shuffled)
        gemm2_sf_shuffled.append(g2_sf_interleaved)
        gemm2_globals.append(g2_global)

    H = hidden_size
    I = intermediate_size

    G1K = torch.stack(gemm1_fp4_shuffled)  # [E, 2I, H//2] uint8
    G1SK = (
        torch.stack(gemm1_sf_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(E, 2 * I, H // SF_VEC)
    )
    G2K = torch.stack(gemm2_fp4_shuffled)  # [E, H, I//2] uint8
    G2SK = (
        torch.stack(gemm2_sf_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(E, H, I // SF_VEC)
    )
    gemm1_globals = torch.stack(gemm1_globals)
    gemm2_globals = torch.stack(gemm2_globals)

    return G1K, G1SK, G2K, G2SK, gemm1_globals, gemm2_globals


# ── Kernel call ───────────────────────────────────────────────────────────────

def _run_kernel(
    routing_logits: torch.Tensor,
    hidden_f32: torch.Tensor,
    gemm1_f32: torch.Tensor,
    gemm2_f32: torch.Tensor,
) -> torch.Tensor:
    """Quantize + shuffle weights, then call the kernel."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    T = routing_logits.shape[0]

    G1K, G1SK, G2K, G2SK, gemm1_globals, gemm2_globals = _quantize_and_shuffle_weights(
        gemm1_f32, gemm2_f32
    )

    # Output scales (c_global_sf = 1.0 for BF16 activations):
    #   out1 = c_global_sf / gemm1_global
    #   out1g = 1 / gemm1_global
    #   out2 = 1 / (c_global_sf * gemm2_global)
    c_global_sf = 1.0
    out1 = (c_global_sf / gemm1_globals).to(torch.float32)
    out1g = (1.0 / gemm1_globals).to(torch.float32)
    out2 = (1.0 / (c_global_sf * gemm2_globals)).to(torch.float32)

    hidden_bf16 = hidden_f32.to(torch.bfloat16)

    result = trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_bf16,
        hidden_states_scale=None,
        gemm1_weights=G1K,
        gemm1_weights_scale=G1SK,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=G2K,
        gemm2_weights_scale=G2SK,
        gemm2_bias=None,
        output1_scale_scalar=out1,
        output1_scale_gate_scalar=out1g,
        output2_scale_scalar=out2,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # Renormalize: TopK -> Softmax
        tune_max_num_tokens=max(8, T * top_k),
    )

    return result[0].to(torch.bfloat16) if isinstance(result, list) else result.to(torch.bfloat16)


# ── Comparison ────────────────────────────────────────────────────────────────

def _compare(ref: torch.Tensor, kernel: torch.Tensor, seq_len: int):
    """Compare reference and kernel outputs."""
    ref_f = ref.to(torch.float32)
    ker_f = kernel.to(torch.float32)

    assert not ker_f.isnan().any(), f"Kernel output has NaN (seq_len={seq_len})"

    # Cosine similarity
    cosine = torch.nn.functional.cosine_similarity(
        ref_f.reshape(1, -1), ker_f.reshape(1, -1)
    ).item()

    # Hit ratio (atol=0.1, rtol=0.85 tolerance for FP4 quantization noise)
    diff = (ref_f - ker_f).abs()
    thresh = 0.1 + 0.85 * ref_f.abs()
    hit_ratio = (diff <= thresh).float().mean().item()

    return cosine, hit_ratio


# ── Pytest ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seq_len", [1, 4, 8, 16, 32, 64])
def test_trtllm_fp4_block_scale_moe(seq_len):
    """Test FP4 block scale (MxFP4+BF16) MoE kernel against float32 reference."""
    torch.manual_seed(seq_len)

    routing_logits, hidden_f32, gemm1_f32, gemm2_f32 = generate_random_inputs(
        seq_len, device
    )

    ref = run(routing_logits, hidden_f32, gemm1_f32, gemm2_f32)
    kernel_out = _run_kernel(routing_logits, hidden_f32, gemm1_f32, gemm2_f32)

    cosine, hit_ratio = _compare(ref, kernel_out, seq_len)

    print(f"seq_len={seq_len}: cosine={cosine:.4f}, hit_ratio={hit_ratio*100:.1f}%")

    assert cosine > 0.9, f"Cosine similarity too low: {cosine:.4f}"
    assert hit_ratio >= 0.9, f"Hit ratio too low: {hit_ratio*100:.1f}%"


if __name__ == "__main__":
    print(f"Testing trtllm_fp4_block_scale_moe_topk2_e8_h2048_i1024")
    for seq_len in [1, 4, 8, 16, 32, 64]:
        torch.manual_seed(seq_len)
        routing_logits, hidden_f32, gemm1_f32, gemm2_f32 = generate_random_inputs(
            seq_len, device
        )
        ref = run(routing_logits, hidden_f32, gemm1_f32, gemm2_f32)
        kernel_out = _run_kernel(routing_logits, hidden_f32, gemm1_f32, gemm2_f32)
        cosine, hit_ratio = _compare(ref, kernel_out, seq_len)
        print(
            f"  seq_len={seq_len:3d}: cosine={cosine:.4f}, hit={hit_ratio*100:.1f}%"
            + (" PASS" if cosine > 0.9 and hit_ratio >= 0.9 else " FAIL")
        )
