"""Reference test for trtllm_fp8_per_tensor_scale_moe_topk1_e16_h5120_i8192."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.data import Definition, load_json_file

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"
DEFINITION_NAME = "trtllm_fp8_per_tensor_scale_moe_topk1_e16_h5120_i8192"

# ── Fixed geometry ────────────────────────────────────────────────────────────
num_experts = 16
top_k = 1
hidden_size = 5120
intermediate_size = 8192

device = "cuda"


def load_definition(name: str) -> Definition:
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math, "F": F}
    exec(reference_code, namespace)
    return namespace["run"]


def _quantize_fp8_per_tensor(x: torch.Tensor):
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    amax = x.float().abs().max().clamp(min=1e-6)
    scale = amax / FP8_MAX
    fp8 = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8, scale.to(torch.float32)


def generate_random_inputs(seq_len: int, dev: str = "cuda"):
    E, H, I = num_experts, hidden_size, intermediate_size
    routing_logits = torch.randn(seq_len, E, dtype=torch.bfloat16, device=dev)
    routing_bias = torch.randn(E, dtype=torch.bfloat16, device=dev) * 0.1
    hidden_f32 = torch.randn(seq_len, H, dtype=torch.float32, device=dev) * 0.1
    hidden_fp8, hs_scale = _quantize_fp8_per_tensor(hidden_f32)
    w1_fp8_list, w2_fp8_list, out1_scales, out2_scales = [], [], [], []
    for e in range(E):
        w1_f32 = torch.randn(2 * I, H, dtype=torch.float32, device=dev) * 0.1
        w2_f32 = torch.randn(H, I, dtype=torch.float32, device=dev) * 0.1
        w1_fp8, w1_s = _quantize_fp8_per_tensor(w1_f32)
        w2_fp8, w2_s = _quantize_fp8_per_tensor(w2_f32)
        w1_fp8_list.append(w1_fp8)
        w2_fp8_list.append(w2_fp8)
        out1_scales.append(hs_scale * w1_s)
        out2_scales.append(w2_s)
    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_fp8,
        "gemm1_weights": torch.stack(w1_fp8_list),
        "gemm2_weights": torch.stack(w2_fp8_list),
        "output1_scales_scalar": torch.tensor(out1_scales, dtype=torch.float32, device=dev),
        "output1_scales_gate_scalar": torch.tensor(out1_scales, dtype=torch.float32, device=dev),
        "output2_scales_scalar": torch.tensor(out2_scales, dtype=torch.float32, device=dev),
    }


def _shuffle_weights(gemm1_fp8: torch.Tensor, gemm2_fp8: torch.Tensor):
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    E = gemm1_fp8.shape[0]
    TILE_M = 128
    cache = {}
    g1_shuffled, g2_shuffled = [], []
    for e in range(E):
        g1_u8 = gemm1_fp8[e].view(torch.uint8)
        perm1 = _maybe_get_cached_w3_w1_permute_indices(cache, g1_u8, TILE_M)
        g1_shuffled.append(g1_u8[perm1.to(device)].contiguous())
        g2_u8 = gemm2_fp8[e].view(torch.uint8)
        perm2 = get_w2_permute_indices_with_cache(cache, g2_u8, TILE_M)
        g2_shuffled.append(g2_u8[perm2.to(device)].contiguous())
    return (
        torch.stack(g1_shuffled).view(torch.float8_e4m3fn),
        torch.stack(g2_shuffled).view(torch.float8_e4m3fn),
    )


def run_kernel(inputs: dict) -> torch.Tensor:
    from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe

    T = inputs["routing_logits"].shape[0]
    g1s, g2s = _shuffle_weights(inputs["gemm1_weights"], inputs["gemm2_weights"])
    result = trtllm_fp8_per_tensor_scale_moe(
        routing_logits=inputs["routing_logits"].to(torch.bfloat16),
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        gemm1_weights=g1s,
        output1_scales_scalar=inputs["output1_scales_scalar"],
        output1_scales_gate_scalar=inputs["output1_scales_gate_scalar"],
        gemm2_weights=g2s,
        output2_scales_scalar=inputs["output2_scales_scalar"],
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=2.5,
        use_routing_scales_on_input=True,
        routing_method_type=3,
        tune_max_num_tokens=max(8, T * top_k),
    )
    return result[0].to(torch.bfloat16) if isinstance(result, list) else result.to(torch.bfloat16)


@pytest.mark.parametrize("seq_len", [1, 4, 8, 16, 32, 64])
def test_fp8_per_tensor_scale_moe_topk1_e16_h5120_i8192(seq_len):
    torch.manual_seed(seq_len)
    definition = load_definition(DEFINITION_NAME)
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(seq_len, device)
    ref = run(
        inputs["routing_logits"],
        inputs["routing_bias"],
        inputs["hidden_states"],
        inputs["gemm1_weights"],
        inputs["output1_scales_scalar"],
        inputs["output1_scales_gate_scalar"],
        inputs["gemm2_weights"],
        inputs["output2_scales_scalar"],
    )
    kernel_out = run_kernel(inputs)
    ref_f = ref.to(torch.float32)
    ker_f = kernel_out.to(torch.float32)
    assert not ker_f.isnan().any(), f"Kernel output has NaN (seq_len={seq_len})"
    cosine = F.cosine_similarity(ref_f.reshape(1, -1), ker_f.reshape(1, -1)).item()
    diff = (ref_f - ker_f).abs()
    hit_ratio = (diff <= 0.1 + 0.85 * ref_f.abs()).float().mean().item()
    print(f"seq_len={seq_len}: cosine={cosine:.4f}, hit_ratio={hit_ratio * 100:.1f}%")
    assert cosine > 0.9, f"Cosine similarity too low: {cosine:.4f}"
    assert hit_ratio >= 0.9, f"Hit ratio too low: {hit_ratio * 100:.1f}%"


if __name__ == "__main__":
    print(f"Testing {DEFINITION_NAME}")
    definition = load_definition(DEFINITION_NAME)
    run = compile_reference(definition.reference)
    for seq_len in [1, 4, 8, 16, 32, 64]:
        torch.manual_seed(seq_len)
        inputs = generate_random_inputs(seq_len, device)
        ref = run(
            inputs["routing_logits"],
            inputs["routing_bias"],
            inputs["hidden_states"],
            inputs["gemm1_weights"],
            inputs["output1_scales_scalar"],
            inputs["output1_scales_gate_scalar"],
            inputs["gemm2_weights"],
            inputs["output2_scales_scalar"],
        )
        kernel_out = run_kernel(inputs)
        ref_f = ref.to(torch.float32)
        ker_f = kernel_out.to(torch.float32)
        cosine = F.cosine_similarity(ref_f.reshape(1, -1), ker_f.reshape(1, -1)).item()
        diff = (ref_f - ker_f).abs()
        hit_ratio = (diff <= 0.1 + 0.85 * ref_f.abs()).float().mean().item()
        status = "PASS" if cosine > 0.9 and hit_ratio >= 0.9 else "FAIL"
        print(f"  seq_len={seq_len:3d}: cosine={cosine:.4f}, hit={hit_ratio * 100:.1f}% {status}")
