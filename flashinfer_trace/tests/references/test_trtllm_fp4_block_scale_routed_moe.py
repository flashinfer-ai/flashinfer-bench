"""
Reference test for trtllm_fp4_block_scale_routed_moe kernel.

This test validates the reference implementation against the FlashInfer
FP4 routed MoE kernel on SM100+ GPUs (Blackwell architecture).

Ground truth source: FlashInfer trtllm_fp4_block_scale_routed_moe API
Reference implementation: Vanilla PyTorch FP4 dequantization + MoE computation
"""

import json
from pathlib import Path

import pytest
import torch

TRACE_ROOT = Path(__file__).resolve().parents[2]
DEFINITION_PATH = (
    TRACE_ROOT
    / "definitions"
    / "moe"
    / "trtllm_fp4_block_scale_routed_moe_topk8_e128_h4096_i2048.json"
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
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer import fp4_quantize, GatedActType, RoutingMethodType

        return {
            "trtllm_fp4_block_scale_routed_moe": trtllm_fp4_block_scale_routed_moe,
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


def dequant_nvfp4_activations(
    packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int = 16
) -> torch.Tensor:
    """
    Dequantize NvFP4 packed activations.

    Args:
        packed: [T, H/2] uint8 tensor (2 FP4 values per byte)
        scale: [T, H/sf_vec_size] float8_e4m3fn scale factors
        sf_vec_size: Scale factor block size (16 for NvFP4)

    Returns:
        Dequantized [T, H] float32 tensor
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
    scale_expanded = scale_fp32.unsqueeze(-1).repeat(1, 1, sf_vec_size).reshape(T, H)

    return unpacked_fp32 * scale_expanded


def dequant_nvfp4_weights(
    packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int = 16
) -> torch.Tensor:
    """
    Dequantize NvFP4 packed weights.

    Args:
        packed: [E, out_dim, in_dim/2] uint8 tensor
        scale: [E, out_dim, in_dim/sf_vec_size] float8_e4m3fn scale factors
        sf_vec_size: Scale factor block size (16 for NvFP4)

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
    scale_expanded = scale_fp32.unsqueeze(-1).repeat(1, 1, 1, sf_vec_size).reshape(
        E, out_dim, in_dim
    )

    return unpacked_fp32 * scale_expanded


@torch.no_grad()
def reference_fp4_routed_moe(
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
    intermediate_size: int,
    top_k: int,
) -> torch.Tensor:
    """
    Reference implementation for FP4 block-scale routed MoE.

    This is a vanilla PyTorch implementation used to validate the FlashInfer kernel.
    """
    device = hidden_states.device
    T = hidden_states.shape[0]
    H = gemm2_weights.shape[1] * 2  # W2 shape is [E, H, I/2]
    I = intermediate_size
    E_local = gemm1_weights.shape[0]
    SF_VEC_SIZE = 16

    # Unpack topk_ids to get expert indices and weights
    expert_indices = (topk_ids & 0xFFFF).to(torch.int32)
    expert_weights_packed = (topk_ids >> 16).to(torch.int16)
    expert_weights = expert_weights_packed.view(torch.bfloat16).to(torch.float32)

    # Dequantize hidden states
    A = dequant_nvfp4_activations(hidden_states, hidden_states_scale, SF_VEC_SIZE)

    # Dequantize weights
    W13 = dequant_nvfp4_weights(gemm1_weights, gemm1_weights_scale, SF_VEC_SIZE)
    W2 = dequant_nvfp4_weights(gemm2_weights, gemm2_weights_scale, SF_VEC_SIZE)

    # Initialize output
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1 -> SwiGLU -> GEMM2
    for le in range(E_local):
        ge = local_start + le

        # Find tokens that selected this expert
        sel_mask = (expert_indices == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        Tk = token_idx.numel()

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
        gate = G1[:, :I]
        up = G1[:, I:] * (scale1_gate / scale1)
        silu_gate = gate * torch.sigmoid(gate)
        C = silu_gate * up

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t()) * scale2

        # Accumulate with routing weights
        output.index_add_(0, token_idx, O * weights.unsqueeze(1))

    return output.to(torch.bfloat16)


def generate_random_fp4_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
):
    """Generate random inputs for FP4 routed MoE testing."""
    fi = try_import_flashinfer_fp4()
    fp4_quantize = fi["fp4_quantize"]
    RoutingMethodType = fi["RoutingMethodType"]

    T, H, I, E = num_tokens, hidden_size, intermediate_size, num_experts
    SF_VEC_SIZE = 16

    # Generate random hidden states and quantize
    hidden_states_bf16 = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
    hidden_states, hidden_states_scale = fp4_quantize(
        hidden_states_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(T, -1)

    # Generate weights and quantize
    w13_bf16 = torch.randn(E, I * 2, H, device=device, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.1

    w13, w13_scale = fp4_quantize(
        w13_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(E, I * 2, -1)

    w2, w2_scale = fp4_quantize(
        w2_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(E, H, -1)

    # Global scales
    global_scale = 1.0 / 448.0 / 6.0
    output1_scale_scalar = torch.full((E,), global_scale * global_scale, device=device)
    output1_scale_gate_scalar = torch.full((E,), global_scale * global_scale, device=device)
    output2_scale_scalar = torch.full((E,), global_scale * global_scale, device=device)

    # Generate routing logits and compute topk indices
    routing_logits = torch.rand(T, E, device=device, dtype=torch.bfloat16)

    # Simple topk routing
    topk_weights, topk_indices = torch.topk(routing_logits.float(), k=top_k, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).to(torch.bfloat16)

    # Pack indices and weights: (indices << 16) | weights_as_int16
    packed_tensor = (topk_indices.to(torch.int32) << 16) | topk_weights.view(torch.int16).to(
        torch.int32
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
        "routing_logits": routing_logits,
        "topk_weights": topk_weights,
        "topk_indices": topk_indices,
    }


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("top_k", [4, 8])
def test_trtllm_fp4_block_scale_routed_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
):
    """
    Test FP4 routed MoE kernel against reference implementation.

    This test validates that the FlashInfer trtllm_fp4_block_scale_routed_moe
    kernel produces outputs matching the reference PyTorch implementation
    within acceptable tolerance.
    """
    skip_if_no_sm100()
    fi = try_import_flashinfer_fp4()

    trtllm_fp4_block_scale_routed_moe = fi["trtllm_fp4_block_scale_routed_moe"]
    RoutingMethodType = fi["RoutingMethodType"]
    GatedActType = fi["GatedActType"]

    torch.manual_seed(42)
    device = "cuda"

    try:
        from flashinfer.utils import device_support_pdl

        enable_pdl = device_support_pdl(torch.device(device))
    except ImportError:
        enable_pdl = False

    # Generate inputs
    inputs = generate_random_fp4_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
    )

    local_expert_offset = 0

    # Run reference implementation
    ref_output = reference_fp4_routed_moe(
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
        intermediate_size=intermediate_size,
        top_k=top_k,
    )

    # Run FlashInfer kernel
    fi_output = trtllm_fp4_block_scale_routed_moe(
        inputs["topk_ids"],
        None,  # routing_bias
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        inputs["gemm1_weights"],
        inputs["gemm1_weights_scale"],
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        inputs["gemm2_weights"],
        inputs["gemm2_weights_scale"],
        None,  # w2_bias
        inputs["output1_scale_scalar"],
        inputs["output1_scale_gate_scalar"],
        inputs["output2_scale_scalar"],
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        local_expert_offset,
        num_experts,  # local_num_experts
        None,  # routed_scaling_factor
        RoutingMethodType.TopK.value,
        True,  # do_finalize
        enable_pdl,
        GatedActType.SwiGlu.value,
        None,
    )[0]

    # Compare outputs
    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    # Use relaxed tolerance for FP4 quantization
    mask = torch.isclose(ref_f32, fi_f32, rtol=1e-2, atol=1e-2)
    mismatch_pct = (~mask).float().mean().item() * 100

    # Allow up to 10% mismatch due to FP4 quantization errors
    assert mismatch_pct < 10, (
        f"Mismatch percentage is {mismatch_pct:.2f}% (expected < 10%)\n"
        f"Config: T={num_tokens}, H={hidden_size}, I={intermediate_size}, "
        f"E={num_experts}, top_k={top_k}"
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

    # Verify key axes
    assert "hidden_size" in definition["axes"]
    assert "intermediate_size" in definition["axes"]
    assert "num_experts" in definition["axes"]
    assert "top_k" in definition["axes"]

    # Verify reference can be compiled
    ref_code = definition["reference"]
    assert "def run(" in ref_code
    assert "torch" in ref_code


if __name__ == "__main__":
    # Run basic tests
    print("Testing definition loading...")
    test_load_definition()
    print("Definition loaded successfully!")

    print("\nTesting FP4 routed MoE (requires SM100+ GPU)...")
    try:
        test_trtllm_fp4_block_scale_routed_moe(
            num_tokens=8,
            hidden_size=1024,
            intermediate_size=1024,
            num_experts=64,
            top_k=4,
        )
        print("Test passed!")
    except Exception as e:
        print(f"Test skipped or failed: {e}")
