"""
Reference test for FP4 block-scale pre-routed MoE kernel (DeepSeek V3/R1).

This test validates the reference implementation against the FlashInfer
FP4 routed MoE kernel on SM100+ GPUs (Blackwell architecture).

Ground truth source: FlashInfer FP4 block-scale routed MoE API
Reference implementation: Vanilla PyTorch FP4 dequantization + MoE computation

DeepSeek V3/R1 MoE Configuration:
- hidden_size: 7168
- intermediate_size: 2048
- num_experts (global): 256
- num_local_experts (EP=8): 32
- top_k: 8
- n_group: 8
- topk_group: 4
- routed_scaling_factor: 2.5
"""

import json
from pathlib import Path

import pytest
import torch

# DeepSeek V3/R1 MoE constants
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32  # EP=8 (256 / 8 = 32)
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
SF_VEC_SIZE = 16  # NvFP4 scale factor block size

# Derived constants
GEMM1_OUT_SIZE = 2 * INTERMEDIATE_SIZE  # 4096
HIDDEN_SIZE_PACKED = HIDDEN_SIZE // 2  # 3584
INTERMEDIATE_SIZE_PACKED = INTERMEDIATE_SIZE // 2  # 1024
NUM_HIDDEN_SF_BLOCKS = HIDDEN_SIZE // SF_VEC_SIZE  # 448
NUM_GEMM1_OUT_SF_BLOCKS = GEMM1_OUT_SIZE // SF_VEC_SIZE  # 256
NUM_INTERMEDIATE_SF_BLOCKS = INTERMEDIATE_SIZE // SF_VEC_SIZE  # 128

TRACE_ROOT = Path(__file__).resolve().parents[2]
DEFINITION_PATH = (
    TRACE_ROOT
    / "definitions"
    / "moe"
    / "moe_fp4_block_scale_pre_routed_topk8_ng8_kg4_e32_h7168_i2048.json"
)


def skip_if_no_sm100():
    """Skip test if GPU doesn't support SM100+ (required for FP4 kernels)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        from flashinfer.utils import get_compute_capability

        cc = get_compute_capability(torch.device("cuda"))
        if cc[0] < 10:
            pytest.skip(f"FP4 MoE kernels require SM100+, got SM{cc[0]}{cc[1]}")
    except ImportError:
        pytest.skip("flashinfer.utils not available")


def try_import_flashinfer_fp4():
    """Try to import FlashInfer FP4 MoE functions."""
    try:
        from flashinfer import GatedActType, RoutingMethodType, fp4_quantize
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

        return {
            "fp4_routed_moe": trtllm_fp4_block_scale_routed_moe,
            "fp4_quantize": fp4_quantize,
            "GatedActType": GatedActType,
            "RoutingMethodType": RoutingMethodType,
        }
    except ImportError as e:
        pytest.skip(f"FlashInfer FP4 MoE not available: {e}")


# FP4 E2M1 lookup table (NvFP4 format)
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_nvfp4_activations(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize NvFP4 packed activations.

    Args:
        packed: [T, HIDDEN_SIZE_PACKED] uint8 tensor (2 FP4 values per byte)
        scale: [T, NUM_HIDDEN_SF_BLOCKS] float8_e4m3fn scale factors

    Returns:
        Dequantized [T, HIDDEN_SIZE] float32 tensor
    """
    device = packed.device
    T, packed_size = packed.shape
    H = packed_size * 2

    # Unpack uint8 to two FP4 values (low nibble first, then high)
    low_nibble = (packed & 0x0F).to(torch.int64)
    high_nibble = ((packed >> 4) & 0x0F).to(torch.int64)

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(T, H)

    # Apply FP4 lookup table
    lut = FP4_LUT.to(device)
    unpacked_fp32 = lut[unpacked]

    # Apply block-wise scale factors
    scale_fp32 = scale.to(torch.float32)
    scale_expanded = scale_fp32.unsqueeze(-1).repeat(1, 1, SF_VEC_SIZE).reshape(T, H)

    return unpacked_fp32 * scale_expanded


def dequant_nvfp4_weights(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize NvFP4 packed weights.

    Args:
        packed: [E, out_dim, in_dim/2] uint8 tensor
        scale: [E, out_dim, in_dim/SF_VEC_SIZE] float8_e4m3fn scale factors

    Returns:
        Dequantized [E, out_dim, in_dim] float32 tensor
    """
    device = packed.device
    E, out_dim, in_dim_packed = packed.shape
    in_dim = in_dim_packed * 2

    low_nibble = (packed & 0x0F).to(torch.int64)
    high_nibble = ((packed >> 4) & 0x0F).to(torch.int64)

    unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(E, out_dim, in_dim)

    lut = FP4_LUT.to(device)
    unpacked_fp32 = lut[unpacked]

    scale_fp32 = scale.to(torch.float32)
    scale_expanded = (
        scale_fp32.unsqueeze(-1).repeat(1, 1, 1, SF_VEC_SIZE).reshape(E, out_dim, in_dim)
    )

    return unpacked_fp32 * scale_expanded


@torch.no_grad()
def run(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    local_expert_offset: int,
) -> torch.Tensor:
    """
    Reference implementation for FP4 block-scale pre-routed MoE (DeepSeek V3/R1).

    This is a vanilla PyTorch implementation used to validate the FlashInfer kernel.
    Uses fixed DeepSeek V3/R1 constants for hidden_size, intermediate_size, etc.

    Input format for topk_ids:
    - Upper 16 bits: expert index (unsigned int16)
    - Lower 16 bits: expert weight (bfloat16 viewed as int16)
    """
    device = hidden_states.device
    T = hidden_states.shape[0]
    E_local = gemm1_weights.shape[0]

    # Verify DeepSeek V3 constants
    assert (
        E_local == NUM_LOCAL_EXPERTS
    ), f"Expected {NUM_LOCAL_EXPERTS} local experts, got {E_local}"
    assert topk_ids.shape[1] == TOP_K, f"Expected top_k={TOP_K}, got {topk_ids.shape[1]}"

    # Unpack topk_ids to get expert indices and weights
    # Format: upper 16 bits = expert index, lower 16 bits = weight (bfloat16 as int16)
    expert_indices = ((topk_ids >> 16) & 0xFFFF).to(torch.int32)
    expert_weights_packed = (topk_ids & 0xFFFF).to(torch.int16)
    expert_weights = expert_weights_packed.view(torch.bfloat16).to(torch.float32)

    # Dequantize hidden states
    A = dequant_nvfp4_activations(hidden_states, hidden_states_scale)

    # Dequantize weights
    W13 = dequant_nvfp4_weights(gemm1_weights, gemm1_weights_scale)
    W2 = dequant_nvfp4_weights(gemm2_weights, gemm2_weights_scale)

    # Initialize output
    output = torch.zeros((T, HIDDEN_SIZE), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1 -> SwiGLU -> GEMM2
    for le in range(E_local):
        ge = local_start + le  # Global expert index
        if ge < 0 or ge >= NUM_EXPERTS_GLOBAL:
            continue

        # Find tokens that selected this expert
        sel_mask = (expert_indices == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # Get routing weights for this expert
        expert_mask = expert_indices[token_idx] == ge
        weights = (expert_weights[token_idx] * expert_mask.float()).sum(dim=1)

        # Gather inputs
        A_e = A.index_select(0, token_idx)
        W13_e = W13[le]
        W2_e = W2[le]

        # Apply output scale factors
        scale1 = output1_scale_scalar[le].item()
        scale1_gate = output1_scale_gate_scalar[le].item()
        scale2 = output2_scale_scalar[le].item()

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t()) * scale1

        # SwiGLU: silu(gate) * up
        gate = G1[:, :INTERMEDIATE_SIZE]
        up = G1[:, INTERMEDIATE_SIZE:] * (scale1_gate / scale1)
        silu_gate = gate * torch.sigmoid(gate)
        C = silu_gate * up

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t()) * scale2

        # Accumulate with routing weights
        output.index_add_(0, token_idx, O * weights.unsqueeze(1))

    return output.to(torch.bfloat16)


def generate_random_fp4_inputs(num_tokens: int, local_expert_offset: int = 0, device: str = "cuda"):
    """
    Generate random inputs for FP4 pre-routed MoE testing with DeepSeek V3 configuration.

    Args:
        num_tokens: Number of tokens (seq_len)
        local_expert_offset: Offset for local experts (0, 32, 64, ..., 224 for EP=8)
        device: Device to create tensors on

    Returns:
        Dictionary of input tensors
    """
    fi = try_import_flashinfer_fp4()
    fp4_quantize = fi["fp4_quantize"]

    T = num_tokens

    # Generate random hidden states and quantize
    hidden_states_bf16 = torch.randn(T, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.1
    hidden_states, hidden_states_scale = fp4_quantize(
        hidden_states_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(T, -1)

    # Generate weights for local experts and quantize
    w13_bf16 = (
        torch.randn(
            NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(
            NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    w13, w13_scale = fp4_quantize(
        w13_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, -1)

    w2, w2_scale = fp4_quantize(
        w2_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(NUM_LOCAL_EXPERTS, HIDDEN_SIZE, -1)

    # Global scales
    global_scale = 1.0 / 448.0 / 6.0
    output1_scale_scalar = torch.full(
        (NUM_LOCAL_EXPERTS,), global_scale * global_scale, device=device
    )
    output1_scale_gate_scalar = torch.full(
        (NUM_LOCAL_EXPERTS,), global_scale * global_scale, device=device
    )
    output2_scale_scalar = torch.full(
        (NUM_LOCAL_EXPERTS,), global_scale * global_scale, device=device
    )

    # Generate routing logits over all global experts
    routing_logits = torch.rand(T, NUM_EXPERTS_GLOBAL, device=device, dtype=torch.bfloat16)

    # Simple topk routing (ensure some tokens go to local experts)
    topk_weights, topk_indices = torch.topk(routing_logits.float(), k=TOP_K, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).to(torch.bfloat16)

    # Pack indices and weights: (indices << 16) | weights_as_int16
    packed_tensor = (topk_indices.to(torch.int32) << 16) | (
        topk_weights.view(torch.int16).to(torch.int32) & 0xFFFF
    )

    return {
        "topk_ids": packed_tensor,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13,
        "gemm1_weights_scale": w13_scale,
        "gemm2_weights": w2,
        "gemm2_weights_scale": w2_scale,
        "output1_scale_scalar": output1_scale_scalar,
        "output1_scale_gate_scalar": output1_scale_gate_scalar,
        "output2_scale_scalar": output2_scale_scalar,
        "local_expert_offset": local_expert_offset,
    }


def _compare_reference_vs_kernel(
    inputs: dict, *, local_expert_offset: int, atol: float, rtol: float, percent: float
):
    """Compare reference implementation vs FlashInfer kernel with comprehensive metrics."""
    fi = try_import_flashinfer_fp4()
    fp4_routed_moe = fi["fp4_routed_moe"]
    RoutingMethodType = fi["RoutingMethodType"]
    GatedActType = fi["GatedActType"]

    device = inputs["hidden_states"].device

    try:
        from flashinfer.utils import device_support_pdl

        enable_pdl = device_support_pdl(torch.device(device))
    except ImportError:
        enable_pdl = False

    # Run reference implementation
    print("Running reference...")
    ref_out = run(
        topk_ids=inputs["topk_ids"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        output1_scale_scalar=inputs["output1_scale_scalar"],
        output1_scale_gate_scalar=inputs["output1_scale_gate_scalar"],
        output2_scale_scalar=inputs["output2_scale_scalar"],
        local_expert_offset=local_expert_offset,
    )

    # Run FlashInfer kernel
    # Use keyword arguments for optional parameters to ensure correct mapping
    print("Running FlashInfer kernel...")
    fi_out = fp4_routed_moe(
        inputs["topk_ids"],
        None,  # routing_bias
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        inputs["gemm1_weights"],
        inputs["gemm1_weights_scale"],
        None,  # gemm1_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        inputs["gemm2_weights"],
        inputs["gemm2_weights_scale"],
        None,  # gemm2_bias
        inputs["output1_scale_scalar"],
        inputs["output1_scale_gate_scalar"],
        inputs["output2_scale_scalar"],
        NUM_EXPERTS_GLOBAL,  # num_experts
        TOP_K,  # top_k
        None,  # n_group (None for simple TopK routing)
        None,  # topk_group (None for simple TopK routing)
        INTERMEDIATE_SIZE,  # intermediate_size
        local_expert_offset,  # local_expert_offset
        NUM_LOCAL_EXPERTS,  # local_num_experts
        None,  # routed_scaling_factor (None for simple TopK routing)
        routing_method_type=RoutingMethodType.TopK.value,
        do_finalize=True,
        enable_pdl=enable_pdl,
        gated_act_type=int(GatedActType.SwiGlu),
        output=None,
    )[0]

    # Compare outputs
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

    # Hit ratio check
    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


# Test with DeepSeek V3 fixed configuration, varying seq_len and local_expert_offset
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 256])
@pytest.mark.parametrize("local_expert_offset", [0, 32, 64, 128])  # Different EP ranks
def test_moe_fp4_block_scale_pre_routed_deepseek_v3(num_tokens: int, local_expert_offset: int):
    """
    Test FP4 pre-routed MoE kernel against reference implementation with DeepSeek V3 config.

    This test validates that the FlashInfer FP4 pre-routed MoE kernel produces
    outputs matching the reference PyTorch implementation within acceptable tolerance.

    Fixed DeepSeek V3/R1 configuration:
    - hidden_size: 7168
    - intermediate_size: 2048
    - num_experts (global): 256
    - num_local_experts: 32 (EP=8)
    - top_k: 8
    """
    skip_if_no_sm100()

    print("\n" + "=" * 70)
    print(f"Testing FP4 MoE: num_tokens={num_tokens}, local_expert_offset={local_expert_offset}")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"

    # Generate inputs with DeepSeek V3 configuration
    inputs = generate_random_fp4_inputs(
        num_tokens=num_tokens, local_expert_offset=local_expert_offset, device=device
    )

    # FP4 quantization has higher error than FP8, use relaxed tolerance
    atol = 1e-1
    rtol = 2e-1
    percent = 0.85  # Allow up to 15% mismatch due to FP4 quantization errors

    ok = _compare_reference_vs_kernel(
        inputs, local_expert_offset=local_expert_offset, atol=atol, rtol=rtol, percent=percent
    )

    assert ok, (
        f"FlashInfer output mismatched reference.\n"
        f"Config: T={num_tokens}, local_expert_offset={local_expert_offset}\n"
        f"DeepSeek V3: H={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, "
        f"E_global={NUM_EXPERTS_GLOBAL}, E_local={NUM_LOCAL_EXPERTS}, top_k={TOP_K}"
    )


def test_load_definition():
    """Test that the definition JSON can be loaded and parsed correctly."""
    assert DEFINITION_PATH.exists(), f"Definition file not found: {DEFINITION_PATH}"

    with open(DEFINITION_PATH, "r") as f:
        definition = json.load(f)

    # Verify required fields
    assert "name" in definition
    assert "op_type" in definition
    assert definition["op_type"] == "moe"
    assert "axes" in definition
    assert "inputs" in definition
    assert "outputs" in definition
    assert "reference" in definition

    # Verify name matches new naming convention
    assert definition["name"] == "moe_fp4_block_scale_pre_routed_topk8_ng8_kg4_e32_h7168_i2048"

    # Verify DeepSeek V3 constants in definition
    axes = definition["axes"]
    assert axes["hidden_size"]["value"] == HIDDEN_SIZE
    assert axes["intermediate_size"]["value"] == INTERMEDIATE_SIZE
    assert axes["num_experts"]["value"] == NUM_EXPERTS_GLOBAL
    assert axes["num_local_experts"]["value"] == NUM_LOCAL_EXPERTS
    assert axes["top_k"]["value"] == TOP_K
    assert axes["n_group"]["value"] == N_GROUP
    assert axes["topk_group"]["value"] == TOPK_GROUP

    # Verify model tags
    assert "model:deepseek-v3" in definition["tags"]
    assert "model:deepseek-r1" in definition["tags"]

    # Verify reference can be compiled
    ref_code = definition["reference"]
    assert "def run(" in ref_code
    assert "HIDDEN_SIZE = 7168" in ref_code
    assert "NUM_EXPERTS_GLOBAL = 256" in ref_code


def main():
    """Run tests manually for debugging."""
    print("Testing FP4 Block-Scale Pre-Routed MoE (DeepSeek V3/R1)")

    print("\n" + "=" * 70)
    print("Testing definition loading...")
    print("=" * 70)
    test_load_definition()
    print("Definition loaded successfully!")

    print(f"\nDeepSeek V3/R1 Configuration:")
    print(f"  hidden_size: {HIDDEN_SIZE}")
    print(f"  intermediate_size: {INTERMEDIATE_SIZE}")
    print(f"  num_experts (global): {NUM_EXPERTS_GLOBAL}")
    print(f"  num_local_experts (EP=8): {NUM_LOCAL_EXPERTS}")
    print(f"  top_k: {TOP_K}")
    print(f"  n_group: {N_GROUP}")
    print(f"  topk_group: {TOPK_GROUP}")

    configs = [
        # (num_tokens, local_expert_offset)
        (1, 0),
        (8, 0),
        (8, 64),
        (64, 32),
        (256, 128),
    ]

    passed = 0
    for T, offset in configs:
        try:
            test_moe_fp4_block_scale_pre_routed_deepseek_v3(
                num_tokens=T, local_expert_offset=offset
            )
            passed += 1
            print("Test passed!")
        except pytest.skip.Exception as e:
            print(f"Test skipped: {e}")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{len(configs)} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
