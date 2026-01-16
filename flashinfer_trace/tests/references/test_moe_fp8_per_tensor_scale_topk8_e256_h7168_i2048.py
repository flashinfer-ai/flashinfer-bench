"""Reference test for FP8 per-tensor scale MoE with Renormalize routing (TopK -> Softmax)."""

import torch
from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe, RoutingMethodType

# Fixed geometry
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
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
    FP8 per-tensor scale MoE with Renormalize routing (TopK -> Softmax).

    • FP8 per-tensor dequantization (TRT-LLM convention): float = fp8 / scale
      where scale = 448/amax, so effectively float = fp8 * (amax/448)
    • Renormalize routing: TopK(logits + bias) → Softmax on top-k values
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """

    # Fixed geometry
    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    E_local = gemm1_weights.shape[0]

    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == HIDDEN_SIZE, f"hidden_size must be {HIDDEN_SIZE}"
    assert I == INTERMEDIATE_SIZE, f"intermediate_size must be {INTERMEDIATE_SIZE}"
    assert E_global == NUM_EXPERTS_GLOBAL, f"num_experts must be {NUM_EXPERTS_GLOBAL}"
    assert E_local == NUM_LOCAL_EXPERTS, f"num_local_experts must be {NUM_LOCAL_EXPERTS}"

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_scales.shape == (E_local,)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_scales.shape == (E_local,)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 per-tensor dequantization (TRT-LLM convention): float = fp8 / scale
    # where scale = 448/amax, so dequant = fp8 * (amax/448)
    # hidden_states: [T, H], scale: scalar
    A = hidden_states.to(torch.float32) / hidden_states_scale  # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_scales.to(torch.float32)  # [E_local]
    W13 = W13_fp32 / S13.view(E_local, 1, 1)  # [E, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_scales.to(torch.float32)  # [E_local]
    W2 = W2_fp32 / S2.view(E_local, 1, 1)  # [E, H, I] float32

    # 2) Renormalize routing: TopK -> Softmax
    logits = routing_logits.to(torch.float32)  # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)  # [E_global]

    # First take top-k on raw logits (with bias added)
    topk_logits, topk_idx = torch.topk(
        logits + bias, k=TOP_K, dim=1, largest=True, sorted=False
    )  # [T, K]

    # Then apply softmax on top-k values only
    topk_weights = torch.softmax(topk_logits, dim=-1)  # [T, K]

    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor  # [T, K]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Find tokens that selected this global expert in their top-k
        # topk_idx: [T, K] contains the expert indices
        sel_mask_per_token = (topk_idx == ge).any(dim=1)  # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)  # [Tk, H]
        W13_e = W13[le]  # [2I, H]
        W2_e = W2[le]  # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())  # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]  # [Tk, I]
        X2 = G1[:, I:]  # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))  # [Tk, I]
        C = silu_X2 * X1  # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())  # [Tk, H]

        # Get routing weights for this expert from topk_weights
        # Find which position in top-k this expert is at for each token
        topk_idx_tok = topk_idx.index_select(0, token_idx)  # [Tk, K]
        topk_weights_tok = topk_weights.index_select(0, token_idx)  # [Tk, K]
        expert_mask = topk_idx_tok == ge  # [Tk, K]
        w_tok = (topk_weights_tok * expert_mask.float()).sum(dim=1)  # [Tk]

        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    return output.to(torch.bfloat16)


# -----------------------------
# Helpers: FP8 per-tensor quantization (TRT-LLM/FlashInfer convention)
# -----------------------------
def _fp8_per_tensor_quant(x: torch.Tensor):
    """
    Quantize tensor to FP8 with per-tensor scale (TRT-LLM convention).
    Returns:
      x_fp8: same shape as x (float8_e4m3fn)
      scale: scalar (float32) -- scale = 448/amax, fp8 = float * scale, dequant = fp8 / scale
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max  # 448

    x_f32 = x.to(torch.float32)
    amax = torch.amax(torch.abs(x_f32)).nan_to_num()

    # TRT-LLM convention: scale = max_fp8 / amax
    # Quantize: fp8 = float * scale
    # Dequant: float = fp8 / scale
    scale = (max_fp8 / amax) if amax > 0 else torch.tensor(1.0, device=x.device)
    x_fp8 = (x_f32 * scale).to(torch.float8_e4m3fn)

    return x_fp8, scale.item()


def _fp8_per_tensor_quant_batched(w: torch.Tensor):
    """
    Quantize batched weights [E, ...] to FP8 with per-expert scale (TRT-LLM convention).
    Returns:
      w_fp8: same shape as w (float8_e4m3fn)
      scales: [E] tensor of per-expert scales (float32) -- scale = 448/amax
    """
    E = w.shape[0]
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max  # 448

    w_f32 = w.to(torch.float32)
    w_fp8 = torch.empty_like(w_f32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(E, dtype=torch.float32, device=w.device)

    for i in range(E):
        amax = torch.amax(torch.abs(w_f32[i])).nan_to_num()
        # TRT-LLM convention: scale = max_fp8 / amax
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

    # Inputs for routing (bfloat16 for FlashInfer kernel)
    routing_logits = torch.randn(T, E_global, dtype=torch.bfloat16, device=device)
    if use_bias:
        routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
    else:
        routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device=device)

    # Activations: start from bf16, then FP8 per-tensor quant
    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scale = _fp8_per_tensor_quant(a_bf16)

    # Weights per local expert
    # W13: [E_local, 2I, H], W2: [E_local, H, I]
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


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


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
        f"Testing MoE FP8 Per-Tensor Scale: seq_len={seq_len}, offset={local_expert_offset}, use_bias={use_bias}"
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

    # Run reference (returns bf16)
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

    # Compute output scales for the kernel (TRT-LLM convention)
    # Scales are defined as: scale = 448/amax
    # The kernel expects: scale_c_fc1 = c_global_sf * (1/gemm1_scale) * (1/hidden_scale)
    # This dequantizes the FP8 GEMM output: out_fp8 * (amax_gemm/448) * (amax_hidden/448)
    # For simplicity, we use c_global_sf = 1.0
    c_global_sf = 1.0
    hidden_states_scale = inputs["hidden_states_scale"]
    scale_c_fc1 = c_global_sf * (1.0 / inputs["gemm1_scales"]) * (1.0 / hidden_states_scale)
    scale_gate_fc1 = (1.0 / inputs["gemm1_scales"]) * (1.0 / hidden_states_scale)
    scale_c_fc2 = (1.0 / c_global_sf) * (1.0 / inputs["gemm2_scales"])

    # Run FlashInfer fused kernel
    print("Running FlashInfer kernel...")
    num_tokens = inputs["hidden_states"].shape[0]
    fi_out = trtllm_fp8_per_tensor_scale_moe(
        inputs["routing_logits"],  # already bfloat16
        inputs["routing_bias"],  # bf16
        inputs["hidden_states"],  # fp8
        inputs["gemm1_weights"],  # fp8
        scale_c_fc1,  # [E_local]
        scale_gate_fc1,  # [E_local]
        inputs["gemm2_weights"],  # fp8
        scale_c_fc2,  # [E_local]
        NUM_EXPERTS_GLOBAL,  # num_experts
        TOP_K,  # top_k
        None,  # n_group (None for Renormalize routing)
        None,  # topk_group (None for Renormalize routing)
        INTERMEDIATE_SIZE,  # intermediate_size
        inputs["local_expert_offset"],  # local_expert_offset
        inputs["local_num_experts"],  # local_num_experts
        inputs["routed_scaling_factor"],  # routed_scaling_factor
        False,  # use_routing_scales_on_input
        routing_method_type=RoutingMethodType.Renormalize.value,
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
        # Show top-5 largest absolute errors
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
    print("Testing FP8 Per-Tensor Scale MoE Reference vs FlashInfer")

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
