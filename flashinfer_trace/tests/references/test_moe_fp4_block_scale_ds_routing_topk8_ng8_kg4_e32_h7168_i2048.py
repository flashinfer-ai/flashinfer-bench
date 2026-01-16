"""Reference test for FP4 block scale MoE with DeepSeek V3 routing."""

import torch
from torch.nn import functional as F

try:
    from flashinfer import (
        RoutingMethodType,
        GatedActType,
        e2m1_and_ufp8sf_scale_to_float,
        fp4_quantize,
    )
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import (
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
        _maybe_get_cached_w3_w1_permute_indices,
    )
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

# Fixed geometry for DeepSeek V3
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUPS = 8
TOPK_GROUP = 4
ROUTED_SCALING_FACTOR = 2.5
SF_VEC_SIZE = 16  # NvFP4 block size


def calculate_fp4_global_scale_factor(tensor):
    """
    Calculate FP4 global scale factor for a tensor.
    Formula: (448 * 6) represents max representable value in FP4 format.
    """
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def quant_fp4_single(a, a_global_sf, is_sf_swizzled_layout=True):
    """Quantize a single tensor to FP4 with pre-computed global scale factor."""
    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), SF_VEC_SIZE, False, is_sf_swizzled_layout
    )
    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with per-expert global scale factors."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_global_sf = calculate_fp4_global_scale_factor(a[i])
        a_fp4, a_sf, _ = quant_fp4_single(a[i], a_global_sf, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_fp4_batches_fixed_scale(a, num_experts, fixed_global_scale, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with fixed global scale factor for all experts."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_fp4, a_sf, _ = quant_fp4_single(a[i], fixed_global_scale, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(fixed_global_scale)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
):
    """Batch FP4 dequantization helper."""
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            SF_VEC_SIZE,
            1,  # ufp8_type for NvFP4
            is_sf_swizzled_layout,
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)
    return result


@torch.no_grad()
def run_reference(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states_dequant: torch.Tensor,
    gemm1_weights_dequant: torch.Tensor,
    gemm2_weights_dequant: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    FP4 block scale MoE reference with DeepSeek V3 no-aux routing.
    Uses dequantized tensors for computation.

    DeepSeek V3 no-aux routing:
    1. s = sigmoid(logits)
    2. s_with_bias = s + bias
    3. Group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
    4. On the kept groups, take global top_k=8 experts
    5. Combine with weights derived from s (without bias), normalized and
       scaled by routed_scaling_factor
    """
    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    E_local = gemm1_weights_dequant.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    device = hidden_states_dequant.device

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
    M.scatter_(1, topk_idx, 1.0)  # 0/1 mask
    topk_weights = s * M  # [T, E_global]
    weights_sum = topk_weights.sum(dim=1, keepdim=True) + 1e-20
    topk_weights = (topk_weights / weights_sum) * routed_scaling_factor

    # Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Find tokens that selected this global expert
        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)

        # Gather inputs and weights
        A_e = hidden_states_dequant.index_select(0, token_idx)  # [Tk, H]
        W13_e = gemm1_weights_dequant[le]  # [2I, H]
        W2_e = gemm2_weights_dequant[le]  # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU: split and apply silu
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = F.silu(X2) * X1

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())

        # Get routing weights for this expert
        # topk_weights is [T, E_global] with weights at the selected expert positions
        w_tok = topk_weights.index_select(0, token_idx)[:, ge]  # [Tk]

        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


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
    """Generate random inputs for MoE testing."""
    T, H, I = seq_len, hidden_size, intermediate_size
    E_global, E_local = num_experts_global, num_local_experts

    # Routing inputs - DeepSeekV3 routing uses float32 for logits
    routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)
    # Boost logits for local expert range to ensure they get selected
    # This ensures test coverage for the local computation path
    local_end = min(local_expert_offset + E_local, E_global)
    routing_logits[:, local_expert_offset:local_end] += 3.0  # Boost by 3 to favor local experts
    if use_bias:
        routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
    else:
        routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device=device)

    # Use fixed global scale factor (448.0 * 6.0) for all quantization
    # This matches the official FlashInfer test approach
    fixed_global_scale = torch.tensor([448.0 * 6.0], device=device)

    # Hidden states: generate bf16, then quantize to FP4
    # Note: hidden_states use non-swizzled layout for scales (is_sf_swizzled_layout=False)
    # Scale data to fit within FP4 range (values should be < 6.0 after scaling)
    hidden_states_bf16 = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1
    hidden_states_fp4, hidden_states_scale, _ = quant_fp4_single(
        hidden_states_bf16, fixed_global_scale, is_sf_swizzled_layout=False
    )

    # Weights per local expert (scaled to fit FP4 range)
    w13_bf16 = torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.1
    w2_bf16 = torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device) * 0.1

    # Quantize weights with swizzled layout for kernel (using fixed global scale)
    w13_fp4_swizzled, w13_scales_swizzled, w13_scales_global = quant_fp4_batches_fixed_scale(
        w13_bf16, E_local, fixed_global_scale, is_sf_swizzled_layout=True
    )
    w2_fp4_swizzled, w2_scales_swizzled, w2_scales_global = quant_fp4_batches_fixed_scale(
        w2_bf16, E_local, fixed_global_scale, is_sf_swizzled_layout=True
    )

    # Also quantize with linear layout for reference dequantization
    w13_fp4_linear, w13_scales_linear, _ = quant_fp4_batches_fixed_scale(
        w13_bf16, E_local, fixed_global_scale, is_sf_swizzled_layout=False
    )
    w2_fp4_linear, w2_scales_linear, _ = quant_fp4_batches_fixed_scale(
        w2_bf16, E_local, fixed_global_scale, is_sf_swizzled_layout=False
    )

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        # FP4 quantized (swizzled for kernel)
        "hidden_states_fp4": hidden_states_fp4,
        "hidden_states_scale": hidden_states_scale,
        "hidden_states_scale_global": fixed_global_scale,
        "gemm1_weights_fp4": w13_fp4_swizzled,
        "gemm1_scales": w13_scales_swizzled,
        "gemm1_scales_global": w13_scales_global,
        "gemm2_weights_fp4": w2_fp4_swizzled,
        "gemm2_scales": w2_scales_swizzled,
        "gemm2_scales_global": w2_scales_global,
        # For reference dequantization
        "gemm1_weights_fp4_linear": w13_fp4_linear,
        "gemm1_scales_linear": w13_scales_linear,
        "gemm2_weights_fp4_linear": w2_fp4_linear,
        "gemm2_scales_linear": w2_scales_linear,
        # Metadata
        "local_expert_offset": int(local_expert_offset),
        "local_num_experts": E_local,
        "routed_scaling_factor": float(routed_scaling_factor),
    }


def prepare_weights_for_kernel(
    inputs: dict,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    cache_permute_indices: dict,
):
    """Prepare shuffled weights for the FP4 MoE kernel."""
    epilogue_tile_m = 128

    # Convert to proper format
    gemm1_weights_fp4 = inputs["gemm1_weights_fp4"].view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    gemm1_scales_linear = inputs["gemm1_scales_linear"].view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // SF_VEC_SIZE
    )

    gemm2_weights_fp4 = inputs["gemm2_weights_fp4"].view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    gemm2_scales_linear = inputs["gemm2_scales_linear"].view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // SF_VEC_SIZE
    )

    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []

    for i in range(num_experts):
        # GEMM1 weights and scales
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm1_weights_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            gemm1_scales_linear[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm1_scales_shuffled.append(
            block_scale_interleave(
                gemm1_scales_linear[i]
                .view(torch.uint8)[permute_sf_indices.to(gemm1_scales_linear.device)]
                .contiguous()
            )
        )

        # GEMM2 weights and scales
        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            gemm2_scales_linear[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_shuffled.append(
            block_scale_interleave(
                gemm2_scales_linear[i]
                .view(torch.uint8)[permute_sf_indices.to(gemm2_scales_linear.device)]
                .contiguous()
            )
        )

    # Stack for all experts
    gemm1_weights_shuffled = torch.stack(gemm1_weights_shuffled)
    gemm1_scales_shuffled = (
        torch.stack(gemm1_scales_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // SF_VEC_SIZE)
    )

    gemm2_weights_shuffled = torch.stack(gemm2_weights_shuffled)
    gemm2_scales_shuffled = (
        torch.stack(gemm2_scales_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // SF_VEC_SIZE)
    )

    return {
        "gemm1_weights_shuffled": gemm1_weights_shuffled,
        "gemm1_scales_shuffled": gemm1_scales_shuffled,
        "gemm2_weights_shuffled": gemm2_weights_shuffled,
        "gemm2_scales_shuffled": gemm2_scales_shuffled,
    }


def test_correctness_moe(
    seq_len: int = 32,
    *,
    local_expert_offset: int = 0,
    use_bias: bool = True,
    atol: float = 0.1,
    rtol: float = 0.85,
    percent: float = 0.925,
):
    """Test FP4 block scale MoE with DeepSeek V3 routing."""
    print("\n" + "=" * 70)
    print(
        f"Testing FP4 Block Scale MoE (DeepSeek V3): seq_len={seq_len}, "
        f"offset={local_expert_offset}, use_bias={use_bias}"
    )
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping test.")
        return True

    if not HAS_FLASHINFER:
        print("WARNING: FlashInfer not available, skipping test.")
        return False

    device = "cuda"
    torch.manual_seed(42)

    # Generate inputs
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

    # Dequantize for reference computation
    print("Dequantizing inputs for reference...")
    hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
        inputs["hidden_states_fp4"].cpu(),
        inputs["hidden_states_scale"].cpu().view(torch.uint8).reshape(-1),
        (1 / inputs["hidden_states_scale_global"]).cpu(),
        SF_VEC_SIZE,
        1,  # ufp8_type for NvFP4
        False,  # is_sf_swizzled_layout - hidden_states use non-swizzled layout
    ).cuda()

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        inputs["gemm1_weights_fp4_linear"],
        inputs["gemm1_scales_linear"],
        1 / inputs["gemm1_scales_global"],
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        inputs["gemm2_weights_fp4_linear"],
        inputs["gemm2_scales_linear"],
        1 / inputs["gemm2_scales_global"],
    ).cuda()

    # Run reference
    print("Running reference...")
    ref_out = run_reference(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states_dequant=hidden_states_dequant,
        gemm1_weights_dequant=gemm1_weights_dequant,
        gemm2_weights_dequant=gemm2_weights_dequant,
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    # Prepare weights for kernel
    print("Preparing weights for kernel...")
    cache_permute_indices = {}
    shuffled_weights = prepare_weights_for_kernel(
        inputs,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_LOCAL_EXPERTS,
        cache_permute_indices,
    )

    # Compute output scales
    # For NvFP4, the dequantization scale is 1.0 / (448.0 * 6.0)
    # Output scale = hidden_states_dequant_scale * weight_dequant_scale
    nvfp4_dequant_scale = 1.0 / 448.0 / 6.0

    # Create per-expert scale tensors (same value for all experts, matching official test)
    scale_c_fc1 = torch.tensor(
        [nvfp4_dequant_scale * nvfp4_dequant_scale] * NUM_LOCAL_EXPERTS,
        device=device,
    )
    scale_gate_fc1 = torch.tensor(
        [nvfp4_dequant_scale * nvfp4_dequant_scale] * NUM_LOCAL_EXPERTS,
        device=device,
    )
    scale_c_fc2 = torch.tensor(
        [nvfp4_dequant_scale * nvfp4_dequant_scale] * NUM_LOCAL_EXPERTS,
        device=device,
    )

    # Run FlashInfer kernel
    print("Running FlashInfer kernel...")
    hidden_states_scale_for_kernel = inputs["hidden_states_scale"].view(
        torch.float8_e4m3fn
    ).reshape(seq_len, -1)

    fi_out_list = trtllm_fp4_block_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states_fp4"],
        hidden_states_scale=hidden_states_scale_for_kernel,
        gemm1_weights=shuffled_weights["gemm1_weights_shuffled"],
        gemm1_weights_scale=shuffled_weights["gemm1_scales_shuffled"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=shuffled_weights["gemm2_weights_shuffled"],
        gemm2_weights_scale=shuffled_weights["gemm2_scales_shuffled"],
        gemm2_bias=None,
        output1_scale_scalar=scale_c_fc1,
        output1_scale_gate_scalar=scale_gate_fc1,
        output2_scale_scalar=scale_c_fc2,
        num_experts=NUM_EXPERTS_GLOBAL,
        top_k=TOP_K,
        n_group=N_GROUPS,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=inputs["local_expert_offset"],
        local_num_experts=inputs["local_num_experts"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
        gated_act_type=GatedActType.SwiGlu.value,
        do_finalize=True,
        tune_max_num_tokens=max(8, min(64, seq_len)),
    )
    fi_out = fi_out_list[0]  # Extract output tensor from list

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

    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


def main():
    print("Testing FP4 Block Scale MoE with DeepSeek V3 Routing")

    configs = [
        # (seq_len, local_expert_offset, use_bias)
        (8, 0, True),
        (16, 64, True),
        (32, 128, True),
        (64, 0, True),
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
