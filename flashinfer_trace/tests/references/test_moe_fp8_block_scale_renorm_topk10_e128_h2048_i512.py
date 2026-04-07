import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

# Qwen3-Next-80B: E=128, H=2048, I=512, topk=10, Renormalize routing (type 1)
E = 128
H = 2048
I = 512
TOP_K = 10
BLOCK = 128


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
):
    """
    FP8 block-scale MoE reference — Renormalize routing (routing_method_type=1).
    Routing: TopK -> Softmax (renormalize). No routing bias, no routed_scaling_factor.
    FP8 block-scale dequantization: float ≈ fp8 * scale (block size = 128).
    Activation: SwiGLU.
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    num_h_blocks = H // BLOCK  # 16
    num_i_blocks = I // BLOCK  # 4

    # 1) FP8 block-scale dequantization of hidden_states
    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A = (A_fp32.view(T, num_h_blocks, BLOCK) * A_scale_TH.unsqueeze(-1)).view(T, H)

    # W13: [E, 2I, H], scale: [E, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    W13 = (
        W13_fp32.view(E, 2 * num_i_blocks, BLOCK, num_h_blocks, BLOCK)
        * S13.unsqueeze(2).unsqueeze(4)
    ).view(E, 2 * I, H)

    # W2: [E, H, I], scale: [E, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    W2 = (
        W2_fp32.view(E, num_h_blocks, BLOCK, num_i_blocks, BLOCK) * S2.unsqueeze(2).unsqueeze(4)
    ).view(E, H, I)

    # 2) Renormalize routing: TopK -> Softmax
    logits = routing_logits.to(torch.float32)  # [T, E]
    topk_logits, topk_idx = torch.topk(logits, k=TOP_K, dim=-1)  # [T, K]
    probs = torch.softmax(topk_logits, dim=-1)  # [T, K]

    # 3) Expert compute and weighted accumulation
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    for e in range(E):
        for k in range(TOP_K):
            tok_mask = topk_idx[:, k] == e
            if not tok_mask.any():
                continue
            tok_idx = torch.nonzero(tok_mask, as_tuple=False).squeeze(1)
            w = probs[tok_idx, k].unsqueeze(1)  # [Tk, 1]
            g1 = A[tok_idx] @ W13[e].t()  # [Tk, 2I]
            up, gate = g1[:, :I], g1[:, I:]
            c = torch.nn.functional.silu(gate) * up  # [Tk, I]
            o = (c @ W2[e].t()) * w  # [Tk, H]
            output.index_add_(0, tok_idx, o)

    return output.to(torch.bfloat16)


def _fp8_block_quant_1d(x_bf16: torch.Tensor, block: int = 128):
    """Quantize [T, H] activations to FP8 with per-(token, block) scales."""
    assert x_bf16.dim() == 2
    T, Hx = x_bf16.shape
    assert Hx % block == 0
    nb = Hx // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    x_f32 = x_bf16.to(torch.float32)
    x_blocked = x_f32.view(T, nb, block)
    amax = torch.amax(torch.abs(x_blocked), dim=2)
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    x_fp8 = (x_blocked / scales.unsqueeze(2)).view(T, Hx).to(torch.float8_e4m3fn)
    return x_fp8, scales  # scales: [T, H/128]


def _fp8_block_quant_2d(w_bf16: torch.Tensor, block: int = 128):
    """Quantize weights [*, R, C] to FP8 with per-block scales [*, R/128, C/128]."""
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r, nb_c = R // block, C // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    w_f32 = w_bf16.to(torch.float32).contiguous()
    w_blocked = w_f32.view(*prefix, nb_r, block, nb_c, block)
    amax = torch.amax(torch.abs(w_blocked), dim=(-3, -1))
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    scales_exp = scales.unsqueeze(-2).unsqueeze(-1)
    w_fp8 = (w_blocked / scales_exp).view(*prefix, R, C).to(torch.float8_e4m3fn)
    return w_fp8, scales


def generate_random_inputs(seq_len: int, device: str = "cuda"):
    T = seq_len
    routing_logits = torch.randn(T, E, dtype=torch.bfloat16, device=device)

    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    hidden_states = a_fp8
    hidden_states_scale = a_scales.transpose(0, 1).contiguous()  # [H/128, T]

    w13_bf16 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device)
    w13_fp8, w13_scales = _fp8_block_quant_2d(w13_bf16)
    w2_fp8, w2_scales = _fp8_block_quant_2d(w2_bf16)

    return {
        "routing_logits": routing_logits,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
    }


def test_correctness_moe(
    seq_len: int = 32, atol: float = 1e-1, rtol: float = 2e-1, percent: float = 0.85
):
    print("\n" + "=" * 70)
    print(f"Testing MoE FP8 Block-Scale Renorm (Qwen3-Next-80B): seq_len={seq_len}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping.")
        return True

    if trtllm_fp8_block_scale_moe is None:
        print("WARNING: kernel not available.")
        return False

    device = "cuda"
    torch.manual_seed(42)
    inputs = generate_random_inputs(seq_len, device=device)

    ref_out = run(
        routing_logits=inputs["routing_logits"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
    )

    fi_out = trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"].to(torch.float32),
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"].to(torch.float32),
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=1,  # Renormalize: TopK -> Softmax
        use_shuffled_weight=False,
        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
    )

    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    print(f"Max abs diff: {abs_diff.max().item():.4e}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    left = abs_diff
    right = atol + rtol * fi_f32.abs()
    hit_ratio = (left <= right).float().mean().item()
    print(f"Hit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")
    return hit_ratio >= percent


def main():
    seq_lens = [1, 4, 8, 16, 32, 64, 256]
    passed = 0
    for T in seq_lens:
        try:
            ok = test_correctness_moe(seq_len=T, percent=0.85)
            passed += int(ok)
        except Exception as e:
            print(f"\n× Test crashed for seq_len={T}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{len(seq_lens)} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
