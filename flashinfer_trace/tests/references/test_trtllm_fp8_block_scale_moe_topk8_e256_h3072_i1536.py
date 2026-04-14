"""Reference test for trtllm_fp8_block_scale_moe_topk8_e256_h3072_i1536.

MiniMax M2 (~230B): 256 experts, top-8, hidden=3072, intermediate=1536.
FP8 block-scale (block=128), sigmoid routing + renormalize.
routing_method_type=0 (no-aux sigmoid topk), routed_scaling_factor=1.0.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.data import Definition, load_json_file

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"
DEFINITION_NAME = "trtllm_fp8_block_scale_moe_topk8_e256_h3072_i1536"

# ── Fixed geometry ────────────────────────────────────────────────────────────
num_experts = 256
top_k = 8
hidden_size = 3072
intermediate_size = 1536
BLOCK = 128

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


def _fp8_block_quant_1d(x: torch.Tensor, block: int = 128):
    """Quantize [T, H] activations to FP8 with per-(token, block) scales.

    Returns:
        x_fp8:  [T, H]       float8_e4m3fn
        scales: [T, H/block] float32
    """
    assert x.dim() == 2
    T, Hx = x.shape
    assert Hx % block == 0
    nb = Hx // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    x_f32 = x.to(torch.float32)
    x_blocked = x_f32.view(T, nb, block)
    amax = torch.amax(x_blocked.abs(), dim=2)
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    x_fp8 = (x_blocked / scales.unsqueeze(2)).view(T, Hx).to(torch.float8_e4m3fn)
    return x_fp8, scales


def _fp8_block_quant_2d(w: torch.Tensor, block: int = 128):
    """Quantize weights [*, R, C] to FP8 with per-block scales [*, R/block, C/block]."""
    assert w.dim() >= 2
    *prefix, R, C = w.shape
    assert R % block == 0 and C % block == 0
    nb_r, nb_c = R // block, C // block
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    w_f32 = w.to(torch.float32).contiguous()
    w_blocked = w_f32.view(*prefix, nb_r, block, nb_c, block)
    amax = torch.amax(w_blocked.abs(), dim=(-3, -1))
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
    w_fp8 = (w_blocked / scales.unsqueeze(-2).unsqueeze(-1)).view(*prefix, R, C).to(
        torch.float8_e4m3fn
    )
    return w_fp8, scales


def generate_random_inputs(seq_len: int, dev: str = "cuda"):
    E, H, I = num_experts, hidden_size, intermediate_size
    T = seq_len

    routing_logits = torch.randn(T, E, dtype=torch.float32, device=dev) * 0.5
    routing_bias = torch.randn(E, dtype=torch.bfloat16, device=dev) * 0.1

    a_bf16 = 0.5 * torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    # Scale layout expected by kernel: [H/128, T] (transposed)
    hidden_states_scale = a_scales.transpose(0, 1).contiguous()  # [H/128, T]

    w13_bf16 = 0.1 * torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=dev)
    w2_bf16 = 0.1 * torch.randn(E, H, I, dtype=torch.bfloat16, device=dev)
    w13_fp8, w13_scales = _fp8_block_quant_2d(w13_bf16)
    w2_fp8, w2_scales = _fp8_block_quant_2d(w2_bf16)

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": a_fp8,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
        "local_expert_offset": 0,
        "routed_scaling_factor": 1.0,
    }


def run_kernel(inputs: dict) -> torch.Tensor:
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    T = inputs["routing_logits"].shape[0]
    return trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"].to(torch.float32),
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"].to(torch.float32),
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,       # no grouped routing for MiniMax M2
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=inputs["local_expert_offset"],
        local_num_experts=num_experts,
        routed_scaling_factor=inputs["routed_scaling_factor"],
        routing_method_type=0,  # no-aux sigmoid topk
        use_shuffled_weight=False,
        tune_max_num_tokens=max(8, T * top_k),
    )


@pytest.mark.parametrize("seq_len", [1, 4, 8, 16, 32, 64])
def test_fp8_block_scale_moe_topk8_e256_h3072_i1536(seq_len):
    torch.manual_seed(seq_len)
    definition = load_definition(DEFINITION_NAME)
    run_ref = compile_reference(definition.reference)
    inputs = generate_random_inputs(seq_len, device)

    ref = run_ref(
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
    run_ref = compile_reference(definition.reference)
    for seq_len in [1, 4, 8, 16, 32, 64]:
        torch.manual_seed(seq_len)
        inputs = generate_random_inputs(seq_len, device)
        ref = run_ref(
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
        kernel_out = run_kernel(inputs)
        ref_f = ref.to(torch.float32)
        ker_f = kernel_out.to(torch.float32)
        cosine = F.cosine_similarity(ref_f.reshape(1, -1), ker_f.reshape(1, -1)).item()
        diff = (ref_f - ker_f).abs()
        hit_ratio = (diff <= 0.1 + 0.85 * ref_f.abs()).float().mean().item()
        status = "PASS" if cosine > 0.9 and hit_ratio >= 0.9 else "FAIL"
        print(f"  seq_len={seq_len:3d}: cosine={cosine:.4f}, hit={hit_ratio * 100:.1f}% {status}")
