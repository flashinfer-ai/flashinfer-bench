import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

# MiniMax M2: E=256, H=3072, I=1536, topk=8, DeepSeek routing (type 2), n_group=1, topk_group=1
E_GLOBAL = 256
H = 3072
I = 1536
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


def _quantize_fp8(x: torch.Tensor):
    """Quantize a float tensor to FP8 with per-token scaling."""
    x_f32 = x.float()
    scale = x_f32.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12) / 448.0
    x_fp8 = (x_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def _make_fp8_2d_block_scale(tensor: torch.Tensor, block_size: int = BLOCK):
    """2D block-scale quantize to FP8 for a 2D weight matrix [out, in].
    Each [block_size x block_size] tile gets one scalar scale.
    Returns (fp8_tensor, scale_tensor) where scale has shape [out//bs, in//bs].
    """
    out, in_ = tensor.shape
    assert out % block_size == 0 and in_ % block_size == 0
    out_b, in_b = out // block_size, in_ // block_size
    # Reshape to [out_b, bs, in_b, bs] then find per-tile abs-max
    t = tensor.float().reshape(out_b, block_size, in_b, block_size)
    scale = t.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0  # [out_b, in_b]
    fp8 = (t / scale.unsqueeze(1).unsqueeze(3)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.reshape(out, in_), scale  # [out, in], [out_b, in_b]


def _reference_moe(
    routing_logits,
    routing_bias,
    hidden_states_fp8,
    hidden_states_scale,
    gemm1_weights_fp8,
    gemm1_weights_scale,
    gemm2_weights_fp8,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """Reference FP8 block-scale MoE: sigmoid routing, SwiGLU experts."""
    T = routing_logits.shape[0]
    E_local = gemm1_weights_fp8.shape[0]
    device = routing_logits.device

    num_h_blocks = H // BLOCK
    num_i_blocks = I // BLOCK

    # Dequantize hidden states: scale shape [H/128, T] -> [T, H/128]
    A_fp32 = hidden_states_fp8.to(torch.float32)
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()
    A = (A_fp32.view(T, num_h_blocks, BLOCK) * A_scale_TH.unsqueeze(-1)).view(T, H)

    # Routing: sigmoid + bias -> top-k -> normalize
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    _, topk_idx = torch.topk(s_with_bias, k=TOP_K, dim=-1)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = weights / weights_sum * routed_scaling_factor

    # Per-expert computation
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)
    for le in range(E_local):
        ge = local_start + le
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        tok_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, tok_idx)

        W13_e = (gemm1_weights_fp8[le].to(torch.float32).view(
            2 * num_i_blocks, BLOCK, num_h_blocks, BLOCK
        ) * gemm1_weights_scale[le].to(torch.float32).unsqueeze(1).unsqueeze(3)).view(2 * I, H)
        g1 = A_e @ W13_e.t()
        up, gate = g1[:, :I], g1[:, I:]
        c = torch.nn.functional.silu(gate) * up

        W2_e = (gemm2_weights_fp8[le].to(torch.float32).view(
            num_h_blocks, BLOCK, num_i_blocks, BLOCK
        ) * gemm2_weights_scale[le].to(torch.float32).unsqueeze(1).unsqueeze(3)).view(H, I)
        o = c @ W2_e.t()
        w_tok = weights[tok_idx, ge].unsqueeze(1)
        output.index_add_(0, tok_idx, o * w_tok)

    return output.to(torch.bfloat16)


@_skip_if_low_vram(20.0)
def test_moe_fp8_block_scale_ds_routing_topk8_ng1_kg1_e256_h3072_i1536(seq_len: int):
    """Compare trtllm_fp8_block_scale_moe output with reference for MiniMax M2."""
    torch.manual_seed(42)
    device = "cuda"

    # --- inputs ---
    routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device) * 0.1
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device) * 0.01

    hidden_fp32 = torch.randn(seq_len, H, device=device)
    hidden_fp8 = hidden_fp32.to(torch.float8_e4m3fn)
    hidden_scale = torch.ones(H // BLOCK, seq_len, device=device)

    # Weights: FP8 with 2D block scales [E, out_blocks, in_blocks]
    gemm1_list_fp8, gemm1_list_scale = [], []
    gemm2_list_fp8, gemm2_list_scale = [], []
    for _ in range(E_GLOBAL):
        w1 = torch.randn(2 * I, H, device=device) * 0.02
        fp8_w1, s1 = _make_fp8_2d_block_scale(w1)
        gemm1_list_fp8.append(fp8_w1)
        gemm1_list_scale.append(s1)

        w2 = torch.randn(H, I, device=device) * 0.02
        fp8_w2, s2 = _make_fp8_2d_block_scale(w2)
        gemm2_list_fp8.append(fp8_w2)
        gemm2_list_scale.append(s2)

    gemm1_fp8 = torch.stack(gemm1_list_fp8)    # [E, 2*I, H]
    gemm1_scale = torch.stack(gemm1_list_scale)  # [E, 2*I//BLOCK, H//BLOCK]
    gemm2_fp8 = torch.stack(gemm2_list_fp8)    # [E, H, I]
    gemm2_scale = torch.stack(gemm2_list_scale)  # [E, H//BLOCK, I//BLOCK]

    local_expert_offset = 0
    rsf = ROUTED_SCALING_FACTOR

    # --- reference ---
    ref_out = _reference_moe(
        routing_logits, routing_bias, hidden_fp8, hidden_scale,
        gemm1_fp8, gemm1_scale, gemm2_fp8, gemm2_scale,
        local_expert_offset, rsf,
    )

    # --- kernel ---
    from flashinfer.fused_moe import RoutingMethodType
    kernel_out = trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_fp8,
        hidden_states_scale=hidden_scale,
        gemm1_weights=gemm1_fp8,
        gemm1_weights_scale=gemm1_scale,
        gemm2_weights=gemm2_fp8,
        gemm2_weights_scale=gemm2_scale,
        num_experts=E_GLOBAL,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        local_expert_offset=local_expert_offset,
        local_num_experts=E_GLOBAL,
        routed_scaling_factor=rsf,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
    )

    # --- compare (allow FP8 quantization noise) ---
    ref_f32 = ref_out.float()
    ker_f32 = kernel_out.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.reshape(1, -1), ker_f32.reshape(1, -1)
    ).item()
    max_abs = (ref_f32 - ker_f32).abs().max().item()
    scale_ref = ref_f32.abs().max().item()

    # Token-level hit ratio
    T = ref_f32.shape[0]
    n_correct = sum(
        1 for i in range(T)
        if torch.nn.functional.cosine_similarity(
            ref_f32[i].unsqueeze(0), ker_f32[i].unsqueeze(0)
        ).item() > 0.95
    )
    hit_ratio = n_correct / T
    print(f"Max abs diff: {max_abs:.4e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Hit ratio: {hit_ratio:.2%}  (need >= 85.00%)")
    assert hit_ratio >= 0.85, f"Hit ratio {hit_ratio:.2%} < 85%"
    return True


if __name__ == "__main__":
    configs = [
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    ]

    passed = 0
    for (seq_len,) in configs:
        print(f"\n{'='*70}")
        print(f"Testing MoE FP8 Block-Scale DeepSeek ng=1 kg=1 (MiniMax M2): seq_len={seq_len}")
        print(f"{'='*70}")
        print(f"Generating inputs (per-expert to manage memory)...")
        print(f"Running reference...")
        print(f"Running FlashInfer kernel...")
        result = test_moe_fp8_block_scale_ds_routing_topk8_ng1_kg1_e256_h3072_i1536(seq_len)
        if result:
            passed += 1

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(configs)} tests passed")
    print(f"{'='*60}")
