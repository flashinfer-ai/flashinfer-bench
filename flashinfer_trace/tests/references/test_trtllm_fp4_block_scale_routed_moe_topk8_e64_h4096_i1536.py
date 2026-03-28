"""Reference test for trtllm_fp4_block_scale_routed_moe_topk8_e64_h4096_i1536."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.data import Definition, load_json_file

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"
DEFINITION_NAME = "trtllm_fp4_block_scale_routed_moe_topk8_e64_h4096_i1536"

# ── Fixed geometry ────────────────────────────────────────────────────────────
num_experts = 64
top_k = 8
hidden_size = 4096
intermediate_size = 1536
SF_VEC = 32  # MxFP4 block size
TILE_M = 128  # epilogue tile size

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

def generate_random_inputs(seq_len: int, dev: str = "cuda"):
    E, H, I, K = num_experts, hidden_size, intermediate_size, top_k
    routing_logits = torch.randn(seq_len, E, dtype=torch.bfloat16, device=dev)
    topk_logits, topk_idx = torch.topk(routing_logits.float(), k=K, dim=-1)
    probs_topk = torch.softmax(topk_logits, dim=-1).to(torch.bfloat16)
    packed_topk_ids = (topk_idx.int() << 16) | probs_topk.view(torch.int16).to(torch.int32)
    return {
        "topk_ids": packed_topk_ids,
        "hidden_states": torch.randn(seq_len, H, dtype=torch.bfloat16, device=dev) * 0.1,
        "gemm1_weights": torch.randn(E, 2 * I, H, dtype=torch.float32, device=dev) * 0.01,
        "gemm2_weights": torch.randn(E, H, I, dtype=torch.float32, device=dev) * 0.01,
    }

def _quantize_and_shuffle_weights(gemm1_f32: torch.Tensor, gemm2_f32: torch.Tensor):
    from flashinfer import fp4_quantize
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    E, H, I = gemm1_f32.shape[0], hidden_size, intermediate_size
    cache = {}
    g1_fp4_list, g1_sf_list, g2_fp4_list, g2_sf_list = [], [], [], []
    for e in range(E):
        g_global = torch.tensor(1.0, dtype=torch.float32, device=device)
        g1_fp4, g1_sf = fp4_quantize(gemm1_f32[e].to(torch.bfloat16), g_global, SF_VEC, True, False)
        g2_fp4, g2_sf = fp4_quantize(gemm2_f32[e].to(torch.bfloat16), g_global, SF_VEC, True, False)
        g1_u8, g1_sf_u8 = g1_fp4.view(torch.uint8), g1_sf.view(torch.uint8)
        g2_u8, g2_sf_u8 = g2_fp4.view(torch.uint8), g2_sf.view(torch.uint8)
        p1 = _maybe_get_cached_w3_w1_permute_indices(cache, g1_u8, TILE_M)
        p1s = _maybe_get_cached_w3_w1_permute_indices(cache, g1_sf_u8, TILE_M, num_elts_per_sf=16)
        g1_fp4_list.append(g1_u8[p1.to(device)].contiguous())
        g1_sf_list.append(block_scale_interleave(g1_sf_u8[p1s.to(device)].contiguous()))
        p2 = get_w2_permute_indices_with_cache(cache, g2_u8, TILE_M)
        p2s = get_w2_permute_indices_with_cache(cache, g2_sf_u8, TILE_M, num_elts_per_sf=16)
        g2_fp4_list.append(g2_u8[p2.to(device)].contiguous())
        g2_sf_list.append(block_scale_interleave(g2_sf_u8[p2s.to(device)].contiguous()))
    G1K = torch.stack(g1_fp4_list)
    G1SK = torch.stack(g1_sf_list).view(torch.float8_e4m3fn).reshape(E, 2 * I, H // SF_VEC)
    G2K = torch.stack(g2_fp4_list)
    G2SK = torch.stack(g2_sf_list).view(torch.float8_e4m3fn).reshape(E, H, I // SF_VEC)
    ones = torch.ones(E, dtype=torch.float32, device=device)
    return G1K, G1SK, G2K, G2SK, ones

def run_kernel(inputs: dict) -> torch.Tensor:
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    T = inputs["topk_ids"].shape[0]
    G1K, G1SK, G2K, G2SK, ones = _quantize_and_shuffle_weights(
        inputs["gemm1_weights"], inputs["gemm2_weights"]
    )
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=inputs["topk_ids"],
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=None,
        gemm1_weights=G1K,
        gemm1_weights_scale=G1SK,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=G2K,
        gemm2_weights_scale=G2SK,
        gemm2_bias=None,
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,
        tune_max_num_tokens=max(8, T * top_k),
    )
    return result[0].to(torch.bfloat16) if isinstance(result, list) else result.to(torch.bfloat16)

@pytest.mark.parametrize("seq_len", [1, 4, 8, 16, 32, 64])
def test_fp4_block_scale_routed_moe_topk8_e64_h4096_i1536(seq_len):
    torch.manual_seed(seq_len)
    definition = load_definition(DEFINITION_NAME)
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(seq_len, device)
    ref = run(
        inputs["topk_ids"],
        inputs["hidden_states"],
        inputs["gemm1_weights"],
        inputs["gemm2_weights"],
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
        inputs["topk_ids"],
        inputs["hidden_states"],
        inputs["gemm1_weights"],
        inputs["gemm2_weights"],
        )
        kernel_out = run_kernel(inputs)
        ref_f = ref.to(torch.float32)
        ker_f = kernel_out.to(torch.float32)
        cosine = F.cosine_similarity(ref_f.reshape(1, -1), ker_f.reshape(1, -1)).item()
        diff = (ref_f - ker_f).abs()
        hit_ratio = (diff <= 0.1 + 0.85 * ref_f.abs()).float().mean().item()
        status = "PASS" if cosine > 0.9 and hit_ratio >= 0.9 else "FAIL"
        print(f"  seq_len={seq_len:3d}: cosine={cosine:.4f}, hit={hit_ratio * 100:.1f}% {status}")
