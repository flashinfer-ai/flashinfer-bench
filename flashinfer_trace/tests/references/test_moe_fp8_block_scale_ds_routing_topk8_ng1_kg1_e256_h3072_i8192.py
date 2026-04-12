import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

# MiniMax M2: E=256, H=3072, I=8192, topk=8, DeepSeek routing (type 2), n_group=1, topk_group=1
E_GLOBAL = 256
H = 3072
I = 8192
TOP_K = 8
N_GROUP = 1
TOPK_GROUP = 1
BLOCK = 128
ROUTED_SCALING_FACTOR = 2.5


def _skip_if_low_vram(min_gb: float):
    """Decorator to skip test if GPU has less than min_gb VRAM."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                free_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if free_gb < min_gb:
                    print(f"SKIP: GPU has {free_gb:.1f} GB VRAM, need >= {min_gb} GB")
                    return True
            return fn(*args, **kwargs)

        return wrapper

    return decorator


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
    assert w_bf16.dim() == 2
    R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r, nb_c = R // block, C // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    w_f32 = w_bf16.to(torch.float32).contiguous()
    w_blocked = w_f32.view(nb_r, block, nb_c, block)
    amax = torch.amax(torch.abs(w_blocked), dim=(-3, -1))  # [nb_r, nb_c]
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    scales_exp = scales.unsqueeze(1).unsqueeze(3)  # [nb_r, 1, nb_c, 1]
    w_fp8 = (w_blocked / scales_exp).view(R, C).to(torch.float8_e4m3fn)
    return w_fp8, scales  # scales: [R/128, C/128]


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    FP8 block-scale MoE reference — DeepSeek routing (routing_method_type=2),
    n_group=1, topk_group=1 (no group selection, direct top-k).
    Routing: sigmoid(logits) + bias -> Top-K -> normalize s_nobias -> * rsf.
    FP8 block-scale dequantization: float ≈ fp8 * scale (block size = 128).
    Activation: SwiGLU.

    Weights are dequantized per-expert to avoid OOM with E=256, I=8192.
    """
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    device = routing_logits.device

    num_h_blocks = H // BLOCK  # 24
    num_i_blocks = I // BLOCK  # 64

    # 1) FP8 block-scale dequantization of hidden_states
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A = (A_fp32.view(T, num_h_blocks, BLOCK) * A_scale_TH.unsqueeze(-1)).view(T, H)

    # 2) DeepSeek routing (ng=1, kg=1 => direct top-k, no group selection)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)  # [T, E_global]
    s_with_bias = s + bias  # [T, E_global]
    _, topk_idx = torch.topk(s_with_bias, k=TOP_K, dim=-1)  # [T, K]

    # Combination weights: normalize s (without bias) over selected experts
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = weights / weights_sum * routed_scaling_factor  # [T, E_global]

    # 3) Local expert computation (per-expert dequant to avoid OOM)
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        tok_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # Dequant W13 for this expert
        W13_e_fp32 = gemm1_weights[le].to(torch.float32)  # [2I, H]
        S13_e = gemm1_weights_scale[le].to(torch.float32)  # [2I/128, H/128]
        W13_e = (
            W13_e_fp32.view(2 * num_i_blocks, BLOCK, num_h_blocks, BLOCK)
            * S13_e.unsqueeze(1).unsqueeze(3)
        ).view(2 * I, H)

        # Dequant W2 for this expert
        W2_e_fp32 = gemm2_weights[le].to(torch.float32)  # [H, I]
        S2_e = gemm2_weights_scale[le].to(torch.float32)  # [H/128, I/128]
        W2_e = (
            W2_e_fp32.view(num_h_blocks, BLOCK, num_i_blocks, BLOCK)
            * S2_e.unsqueeze(1).unsqueeze(3)
        ).view(H, I)

        A_e = A.index_select(0, tok_idx)
        g1 = A_e @ W13_e.t()  # [Tk, 2I]
        up, gate = g1[:, :I], g1[:, I:]
        c = torch.nn.functional.silu(gate) * up  # [Tk, I]
        o = c @ W2_e.t()  # [Tk, H]
        w_tok = weights[tok_idx, ge].unsqueeze(1)
        output.index_add_(0, tok_idx, o * w_tok)

        del W13_e_fp32, S13_e, W13_e, W2_e_fp32, S2_e, W2_e, A_e, g1, up, gate, c, o

    return output.to(torch.bfloat16)


def generate_random_inputs(
    seq_len: int,
    *,
    num_local_experts: int = E_GLOBAL,
    local_expert_offset: int = 0,
    device: str = "cuda",
):
    """Generate random FP8 inputs. Weights generated per-expert to avoid OOM."""
    T = seq_len
    E_local = num_local_experts

    routing_logits = torch.randn(T, E_GLOBAL, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)

    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    hidden_states = a_fp8
    hidden_states_scale = a_scales.transpose(0, 1).contiguous()  # [H/128, T]
    del a_bf16, a_scales

    num_i_blocks = I // BLOCK
    num_h_blocks = H // BLOCK

    # Pre-allocate weight tensors (FP8)
    w13_fp8 = torch.empty(E_local, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
    w13_scales = torch.empty(E_local, 2 * num_i_blocks, num_h_blocks, device=device)
    w2_fp8 = torch.empty(E_local, H, I, dtype=torch.float8_e4m3fn, device=device)
    w2_scales = torch.empty(E_local, num_h_blocks, num_i_blocks, device=device)

    # Generate weights one expert at a time to avoid OOM
    for e in range(E_local):
        w13_e_bf16 = torch.randn(2 * I, H, dtype=torch.bfloat16, device=device)
        fp8_e, sc_e = _fp8_block_quant_2d(w13_e_bf16)
        w13_fp8[e] = fp8_e
        w13_scales[e] = sc_e
        del w13_e_bf16, fp8_e, sc_e

        w2_e_bf16 = torch.randn(H, I, dtype=torch.bfloat16, device=device)
        fp8_e, sc_e = _fp8_block_quant_2d(w2_e_bf16)
        w2_fp8[e] = fp8_e
        w2_scales[e] = sc_e
        del w2_e_bf16, fp8_e, sc_e

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
        "local_expert_offset": int(local_expert_offset),
        "local_num_experts": E_local,
        "routed_scaling_factor": float(ROUTED_SCALING_FACTOR),
    }


@_skip_if_low_vram(40.0)
def test_correctness_moe(
    seq_len: int = 8, atol: float = 1e-1, rtol: float = 2e-1, percent: float = 0.85
):
    print("\n" + "=" * 70)
    print(f"Testing MoE FP8 Block-Scale DeepSeek ng=1 kg=1 (MiniMax M2): seq_len={seq_len}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping.")
        return True

    if trtllm_fp8_block_scale_moe is None:
        print("WARNING: kernel not available.")
        return False

    device = "cuda"
    torch.manual_seed(42)

    print("Generating inputs (per-expert to manage memory)...")
    inputs = generate_random_inputs(seq_len, device=device)

    print("Running reference...")
    ref_out = run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    print("Running FlashInfer kernel...")
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
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < 40.0:
        print(f"SKIP: GPU has {total_gb:.1f} GB VRAM, need >= 40 GB")
        return

    # Use small seq_lens to keep memory manageable
    seq_lens = [1, 4, 8, 16, 32]
    passed = 0
    for T in seq_lens:
        try:
            torch.cuda.empty_cache()
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
