import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

# Kimi K2 EP=8: E_global=384, E_local=48 (384/8), H=7168, I=2048, topk=8, ng=1, kg=1
E_GLOBAL = 384
E_LOCAL = 48
H = 7168
I = 2048
TOP_K = 8
N_GROUP = 1
TOPK_GROUP = 1
BLOCK = 128
ROUTED_SCALING_FACTOR = 2.5


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
    """Quantize weights [R, C] to FP8 with per-block scales [R/128, C/128]."""
    R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nr, nc = R // block, C // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    w_f32 = w_bf16.to(torch.float32).view(nr, block, nc, block)
    amax = torch.amax(torch.abs(w_f32), dim=(1, 3))  # [nr, nc]
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    w_fp8 = (w_f32 / scales[:, None, :, None]).view(R, C).to(torch.float8_e4m3fn)
    return w_fp8, scales  # scales: [nr, nc]


@torch.no_grad()
def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE reference — DeepSeek routing (ng=1, kg=1 => direct top-k).
    E_local=48 for EP=8 on 8xB200 (384/8=48 local experts per GPU).
    """
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]  # 48
    num_h_blocks = H // BLOCK
    num_i_blocks = I // BLOCK
    device = routing_logits.device

    # Dequantize hidden_states: hidden_states_scale is [H/128, T]
    A_fp32 = hidden_states.to(torch.float32)
    A_scale_TH = hidden_states_scale.permute(1, 0).contiguous()  # [T, H/128]
    A = (A_fp32.view(T, num_h_blocks, BLOCK) * A_scale_TH.unsqueeze(-1)).view(T, H)

    # DeepSeek routing (ng=1, kg=1 => direct top-k, no group selection)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    _, topk_idx = torch.topk(s_with_bias, k=TOP_K, dim=-1)  # [T, K]

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = weights / weights_sum * routed_scaling_factor

    # Local expert computation
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)
    for le in range(E_local):
        ge = local_start + le
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        tok_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, tok_idx)

        W13_e = (gemm1_weights[le].to(torch.float32).view(
            2 * num_i_blocks, BLOCK, num_h_blocks, BLOCK
        ) * gemm1_weights_scale[le].to(torch.float32).unsqueeze(1).unsqueeze(3)).view(2 * I, H)
        g1 = A_e @ W13_e.t()
        up, gate = g1[:, :I], g1[:, I:]
        c = torch.nn.functional.silu(gate) * up

        W2_e = (gemm2_weights[le].to(torch.float32).view(
            num_h_blocks, BLOCK, num_i_blocks, BLOCK
        ) * gemm2_weights_scale[le].to(torch.float32).unsqueeze(1).unsqueeze(3)).view(H, I)
        o = c @ W2_e.t()
        w_tok = weights[tok_idx, ge].unsqueeze(1)
        output.index_add_(0, tok_idx, o * w_tok)

    return output.to(torch.bfloat16)


def generate_random_inputs(seq_len: int, local_expert_offset: int = 0, device: str = "cuda"):
    """Generate random FP8-quantized MoE inputs for E_local=48."""
    T = seq_len
    num_i_blocks = I // BLOCK
    num_h_blocks = H // BLOCK

    routing_logits = torch.randn(T, E_GLOBAL, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)

    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    hidden_states = a_fp8
    hidden_states_scale = a_scales.transpose(0, 1).contiguous()  # [H/128, T]
    del a_bf16, a_scales

    # Pre-allocate weight tensors (FP8) for E_LOCAL=48
    w13_fp8 = torch.empty(E_LOCAL, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
    w13_scales = torch.empty(E_LOCAL, 2 * num_i_blocks, num_h_blocks, device=device)
    w2_fp8 = torch.empty(E_LOCAL, H, I, dtype=torch.float8_e4m3fn, device=device)
    w2_scales = torch.empty(E_LOCAL, num_h_blocks, num_i_blocks, device=device)

    for e in range(E_LOCAL):
        w13_e = torch.randn(2 * I, H, dtype=torch.bfloat16, device=device) * 0.01
        fp8_e, sc_e = _fp8_block_quant_2d(w13_e)
        w13_fp8[e] = fp8_e
        w13_scales[e] = sc_e
        del w13_e, fp8_e, sc_e

        w2_e = torch.randn(H, I, dtype=torch.bfloat16, device=device) * 0.01
        fp8_e, sc_e = _fp8_block_quant_2d(w2_e)
        w2_fp8[e] = fp8_e
        w2_scales[e] = sc_e
        del w2_e, fp8_e, sc_e

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": E_LOCAL,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
    }


def test_correctness(seq_len: int = 4, local_expert_offset: int = 0, atol: float = 0.5, rtol: float = 0.1):
    print(f"\n{'='*60}")
    print(f"Testing MoE FP8 e48 (Kimi K2 EP=8), T={seq_len}, offset={local_expert_offset}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    inputs = generate_random_inputs(seq_len, local_expert_offset, device)

    # Reference
    ref_out = run(
        inputs["routing_logits"],
        inputs["routing_bias"],
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        inputs["gemm1_weights"],
        inputs["gemm1_weights_scale"],
        inputs["gemm2_weights"],
        inputs["gemm2_weights_scale"],
        inputs["local_expert_offset"],
        inputs["routed_scaling_factor"],
    )

    # FlashInfer
    try:
        fi_out = trtllm_fp8_block_scale_moe(
            routing_logits=inputs["routing_logits"],
            routing_bias=inputs["routing_bias"],
            hidden_states=inputs["hidden_states"],
            hidden_states_scale=inputs["hidden_states_scale"],
            gemm1_weights=inputs["gemm1_weights"],
            gemm1_weights_scale=inputs["gemm1_weights_scale"].to(torch.float32),
            gemm2_weights=inputs["gemm2_weights"],
            gemm2_weights_scale=inputs["gemm2_weights_scale"].to(torch.float32),
            num_experts=E_GLOBAL,
            top_k=TOP_K,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            intermediate_size=I,
            local_expert_offset=inputs["local_expert_offset"],
            local_num_experts=inputs["local_num_experts"],
            routed_scaling_factor=inputs["routed_scaling_factor"],
            routing_method_type=2,  # DeepSeek-V3 routing
            use_shuffled_weight=False,
            tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
        )
    except Exception as e:
        print(f"  FlashInfer call failed: {e}")
        return False

    abs_diff = torch.abs(ref_out.float() - fi_out.float())
    all_close = torch.allclose(ref_out.float(), fi_out.float(), atol=atol, rtol=rtol)

    if all_close:
        print(f"✓ PASSED max_abs_diff={abs_diff.max().item():.4e} (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ FAILED max_abs_diff={abs_diff.max().item():.4e} (atol={atol}, rtol={rtol})")

    return all_close


def main():
    print("Testing MoE FP8 block-scale ds-routing topk8 ng1 kg1 e48 h7168 i2048 (Kimi K2 EP=8)")

    test_configs = [(1, 0), (4, 0), (8, 48), (16, 0), (32, 96)]
    passed = sum(1 for T, offset in test_configs if test_correctness(T, offset))
    total = len(test_configs)

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
