"""Reference test for FP8 per-tensor scale MoE with DeepSeek V3 no-aux routing."""

import torch
from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe, RoutingMethodType

# Fixed geometry for DeepSeek V3
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUPS = 8
TOPK_GROUP = 4
ROUTED_SCALING_FACTOR = 2.5


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: float,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    FP8 per-tensor scale MoE with DeepSeek V3 no-aux routing.

    DeepSeek V3 no-aux routing:
    1. s = sigmoid(logits)
    2. s_with_bias = s + bias
    3. Group by n_group=8; per group take top-2 sum -> pick topk_group=4 groups
    4. On the kept groups, take global top_k=8 experts
    5. Combine with weights derived from s (without bias), normalized and
       scaled by routed_scaling_factor

    Local computation:
    - Only experts in [local_expert_offset, local_expert_offset + E_local) are computed
    - GEMM1 -> SwiGLU -> GEMM2, then per-token weighted accumulation
    """

    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert E_global == NUM_EXPERTS_GLOBAL, f"num_experts must be {NUM_EXPERTS_GLOBAL}"
    assert E_local == NUM_LOCAL_EXPERTS, f"num_local_experts must be {NUM_LOCAL_EXPERTS}"

    device = hidden_states.device

    # Keep FP8 tensors for GEMM
    A_fp8 = hidden_states
    W13_fp8 = gemm1_weights
    W2_fp8 = gemm2_weights
    S13 = gemm1_scales.to(torch.float32)
    S2 = gemm2_scales.to(torch.float32)

    # DeepSeek V3 no-aux routing
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    # Sigmoid scoring
    s = 1.0 / (1.0 + torch.exp(-logits))  # [T, E_global]
    s_with_bias = s + bias  # [T, E_global]

    # Group experts
    group_size = E_global // N_GROUPS  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUPS, group_size)  # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)  # [T, 8]

    # Select topk_group groups -> group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)  # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUPS, group_size).reshape(T, E_global)

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)  # [T, E_global]
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M  # [T, E_global]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)

        A_e_fp8 = A_fp8.index_select(0, token_idx)
        W13_e_fp8 = W13_fp8[le]
        W2_e_fp8 = W2_fp8[le]

        # GEMM1: FP8 GEMM then dequantize
        G1_raw = A_e_fp8.to(torch.float32).matmul(W13_e_fp8.to(torch.float32).t())
        gemm1_dequant_scale = (1.0 / S13[le]) * (1.0 / hidden_states_scale)
        G1 = G1_raw * gemm1_dequant_scale

        # SwiGLU
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1

        # Quantize intermediate for GEMM2
        C_fp8, C_scale = _fp8_per_tensor_quant_single(C)

        # GEMM2: FP8 GEMM then dequantize
        O_raw = C_fp8.to(torch.float32).matmul(W2_e_fp8.to(torch.float32).t())
        gemm2_dequant_scale = (1.0 / S2[le]) * (1.0 / C_scale)
        O = O_raw * gemm2_dequant_scale

        # Weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


def _fp8_per_tensor_quant_single(x: torch.Tensor):
    """Quantize a single tensor to FP8 with per-tensor scale (TRT-LLM convention)."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max  # 448

    x_f32 = x.to(torch.float32)
    amax = torch.amax(torch.abs(x_f32)).nan_to_num()

    scale = (max_fp8 / amax) if amax > 0 else torch.tensor(1.0, device=x.device)
    x_fp8 = (x_f32 * scale).to(torch.float8_e4m3fn)

    return x_fp8, scale


# -----------------------------
# Helpers: FP8 per-tensor quantization
# -----------------------------
def _fp8_per_tensor_quant(x: torch.Tensor):
    """Quantize tensor to FP8 with per-tensor scale."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max  # 448

    x_f32 = x.to(torch.float32)
    amax = torch.amax(torch.abs(x_f32)).nan_to_num()

    scale = (max_fp8 / amax) if amax > 0 else torch.tensor(1.0, device=x.device)
    x_fp8 = (x_f32 * scale).to(torch.float8_e4m3fn)

    return x_fp8, scale.item()


def _fp8_per_tensor_quant_batched(w: torch.Tensor):
    """Quantize batched weights [E, ...] to FP8 with per-expert scale."""
    E = w.shape[0]
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max  # 448

    w_f32 = w.to(torch.float32)
    w_fp8 = torch.empty_like(w_f32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(E, dtype=torch.float32, device=w.device)

    for i in range(E):
        amax = torch.amax(torch.abs(w_f32[i])).nan_to_num()
        s = (max_fp8 / amax) if amax > 0 else torch.tensor(1.0, device=w.device)
        w_fp8[i] = (w_f32[i] * s).to(torch.float8_e4m3fn)
        scales[i] = s

    return w_fp8, scales


# -----------------------------
# Random input generator
# -----------------------------
@torch.no_grad()
def generate_random_inputs_moe(
    seq_len: int,
    *,
    num_experts_global: int = NUM_EXPERTS_GLOBAL,
    num_local_experts: int = NUM_LOCAL_EXPERTS,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = INTERMEDIATE_SIZE,
    use_bias: bool = True,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    device: str = "cuda",
):
    T, H, I = seq_len, hidden_size, intermediate_size
    E_global, E_local = num_experts_global, num_local_experts

    # Inputs for routing - DeepSeek V3 uses float32 for logits
    routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)

    if use_bias:
        routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
    else:
        routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device=device)

    # Boost logits AND bias for local expert range to ensure they get selected
    # DeepSeek V3 routing uses s_with_bias = sigmoid(logits) + bias for group selection
    local_end = min(local_expert_offset + E_local, E_global)
    routing_logits[:, local_expert_offset:local_end] += 10.0
    routing_bias[local_expert_offset:local_end] = routing_bias[local_expert_offset:local_end].abs() + 5.0

    # Activations: start from bf16, then FP8 per-tensor quant
    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scale = _fp8_per_tensor_quant(a_bf16)

    # Weights per local expert
    w13_bf16 = torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device)

    w13_fp8, w13_scales = _fp8_per_tensor_quant_batched(w13_bf16)
    w2_fp8, w2_scales = _fp8_per_tensor_quant_batched(w2_bf16)

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": a_fp8,
        "hidden_states_scale": a_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_scales": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_scales": w2_scales,
        "local_expert_offset": int(local_expert_offset),
        "local_num_experts": E_local,
        "routed_scaling_factor": float(routed_scaling_factor),
    }


# -----------------------------
# Test driver
# -----------------------------
def test_correctness_moe(
    seq_len: int = 32,
    *,
    local_expert_offset: int = 0,
    use_bias: bool = True,
    atol: float = 1e-1,
    rtol: float = 8.5e-1,
    percent: float = 0.925,
):
    print("\n" + "=" * 70)
    print(
        f"Testing MoE FP8 Per-Tensor Scale (DeepSeek V3): seq_len={seq_len}, "
        f"offset={local_expert_offset}, use_bias={use_bias}"
    )
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping test.")
        return True

    if trtllm_fp8_per_tensor_scale_moe is None:
        print("WARNING: flashinfer trtllm_fp8_per_tensor_scale_moe kernel not available.")
        return False

    device = "cuda"
    torch.manual_seed(42)

    # Generate random but consistent inputs
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=NUM_EXPERTS_GLOBAL,
        num_local_experts=NUM_LOCAL_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        use_bias=use_bias,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        device=device,
    )

    # Run reference
    print("Running reference...")
    ref_out = run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_scales=inputs["gemm1_scales"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_scales=inputs["gemm2_scales"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    # Compute output scales for the kernel
    c_global_sf = 1.0
    hidden_states_scale = inputs["hidden_states_scale"]
    scale_c_fc1 = c_global_sf * (1.0 / inputs["gemm1_scales"]) * (1.0 / hidden_states_scale)
    scale_gate_fc1 = (1.0 / inputs["gemm1_scales"]) * (1.0 / hidden_states_scale)
    scale_c_fc2 = (1.0 / c_global_sf) * (1.0 / inputs["gemm2_scales"])

    # Run FlashInfer fused kernel
    print("Running FlashInfer kernel...")
    num_tokens = inputs["hidden_states"].shape[0]
    fi_out = trtllm_fp8_per_tensor_scale_moe(
        inputs["routing_logits"],  # float32 for DeepSeek V3
        inputs["routing_bias"],  # bf16
        inputs["hidden_states"],  # fp8
        inputs["gemm1_weights"],  # fp8
        scale_c_fc1,  # [E_local]
        scale_gate_fc1,  # [E_local]
        inputs["gemm2_weights"],  # fp8
        scale_c_fc2,  # [E_local]
        NUM_EXPERTS_GLOBAL,  # num_experts
        TOP_K,  # top_k
        N_GROUPS,  # n_group (8 for DeepSeek V3)
        TOPK_GROUP,  # topk_group (4 for DeepSeek V3)
        INTERMEDIATE_SIZE,  # intermediate_size
        inputs["local_expert_offset"],  # local_expert_offset
        inputs["local_num_experts"],  # local_num_experts
        inputs["routed_scaling_factor"],  # routed_scaling_factor
        False,  # use_routing_scales_on_input
        routing_method_type=RoutingMethodType.DeepSeekV3.value,  # DeepSeek V3 routing
        tune_max_num_tokens=max(8, min(64, num_tokens)),
    )

    # Compare
    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    rel_diff = abs_diff / (fi_f32.abs() + 1e-8)

    print("\nComparison stats:")
    print(f"Max abs diff:  {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Max rel diff:  {rel_diff.max().item():.6e}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6e}")

    # Cosine similarity and MSE
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - fi_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    # Strict allclose
    allclose = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    print(f"\nAllclose(atol={atol}, rtol={rtol}): {allclose}")

    if not allclose:
        flat = abs_diff.flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print("\nTop-5 absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            t = idx // HIDDEN_SIZE
            h = idx % HIDDEN_SIZE
            print(
                f"  [t={t}, h={h}]: ref={ref_f32.flatten()[idx].item():.6e}, "
                f"fi={fi_f32.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}"
            )

    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


def main():
    print("Testing FP8 Per-Tensor Scale MoE with DeepSeek V3 Routing")

    configs = [
        # (seq_len, local_expert_offset, use_bias)
        (1, 0, False),
        (4, 0, True),
        (8, 64, True),
        (16, 32, True),
        (64, 128, True),
        (256, 64, True),
    ]

    passed = 0
    for T, off, use_bias in configs:
        try:
            ok = test_correctness_moe(
                seq_len=T, local_expert_offset=off, use_bias=use_bias, percent=0.925
            )
            passed += int(ok)
        except Exception as e:
            print(f"\n× Test crashed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{len(configs)} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
