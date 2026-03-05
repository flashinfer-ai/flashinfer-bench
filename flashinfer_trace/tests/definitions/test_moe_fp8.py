"""Tests for MoE FP8 block scale definitions.

This test requires workload data from the workloads directory.
If workloads are not available, tests using workload data will be skipped.
"""

import sys

import numpy as np
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest
from flashinfer_bench.testing.comparators import HitRatioComparator

try:
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    FLASHINFER_MOE_AVAILABLE = True
except ImportError:
    FLASHINFER_MOE_AVAILABLE = False

# Constants
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_EXPERTS_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK_SIZE = 128


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def get_tile_tokens_dim(num_tokens, top_k, num_experts):
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


def _fp8_block_quant_1d(x_bf16: torch.Tensor, block: int = 128):
    """Quantize [T, H] activations into FP8 with per-block scales."""
    assert x_bf16.dim() == 2
    T, H = x_bf16.shape
    assert H % block == 0
    nb = H // block

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    x_f32 = x_bf16.to(torch.float32)
    x_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x_bf16.device)
    scales = torch.empty((T, nb), dtype=torch.float32, device=x_bf16.device)

    for j in range(nb):
        sl = slice(j * block, (j + 1) * block)
        blk = x_f32[:, sl]
        amax = torch.amax(torch.abs(blk), dim=1)
        s = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
        q = (blk / s.unsqueeze(1)).to(torch.float8_e4m3fn)
        x_fp8[:, sl] = q
        scales[:, j] = s
    return x_fp8, scales


def _fp8_block_quant_2d(w_bf16: torch.Tensor, block: int = 128):
    """Quantize weights with 2D block scales."""
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r = R // block
    nb_c = C // block

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    w_f32 = w_bf16.to(torch.float32).contiguous()
    w_fp8 = torch.empty_like(w_f32, dtype=torch.float8_e4m3fn)
    scales = torch.empty((*prefix, nb_r, nb_c), dtype=torch.float32, device=w_bf16.device)

    it = np.ndindex(*prefix) if prefix else [()]
    for idx in it:
        sel = idx if isinstance(idx, tuple) else (idx,)
        for i in range(nb_r):
            rs = slice(i * block, (i + 1) * block)
            for j in range(nb_c):
                cs = slice(j * block, (j + 1) * block)
                blk = w_f32[(*sel, rs, cs)]
                amax = torch.amax(torch.abs(blk))
                s = (amax / max_fp8) if amax > 0 else torch.tensor(1.0, device=w_bf16.device)
                q = (blk / s).to(torch.float8_e4m3fn)
                w_fp8[(*sel, rs, cs)] = q
                scales[(*sel, i, j)] = s
    return w_fp8, scales


@torch.no_grad()
def generate_moe_fp8_inputs(
    seq_len: int,
    num_experts_global: int = NUM_EXPERTS_GLOBAL,
    num_local_experts: int = NUM_EXPERTS_LOCAL,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = INTERMEDIATE_SIZE,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 2.5,
    device: str = "cuda",
):
    """Generate random inputs for MoE FP8 testing."""
    T, H, I = seq_len, hidden_size, intermediate_size
    E_global, E_local = num_experts_global, num_local_experts

    # Routing inputs
    routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)

    # Hidden states with FP8 quantization
    hidden_bf16 = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    hidden_states, hidden_states_scale = _fp8_block_quant_1d(hidden_bf16)
    hidden_states_scale = hidden_states_scale.permute(1, 0).contiguous()

    # Expert weights with FP8 quantization
    w13_bf16 = torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device)

    gemm1_weights, gemm1_weights_scale = _fp8_block_quant_2d(w13_bf16)
    gemm2_weights, gemm2_weights_scale = _fp8_block_quant_2d(w2_bf16)

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "local_expert_offset": local_expert_offset,
        "routed_scaling_factor": routed_scaling_factor,
    }


@pytest.mark.skipif(not FLASHINFER_MOE_AVAILABLE, reason="FlashInfer MoE not available")
class TestMoEFP8BlockScale(DefinitionTest):
    """Test MoE FP8 block scale with DeepSeek routing."""

    definition_path = (
        "definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json"
    )
    configs = [{"seq_len": 64}, {"seq_len": 128}, {"seq_len": 256}]
    atol = 1e-1
    rtol = 2e-1
    comparator = HitRatioComparator(atol=1e-1, rtol=2e-1, min_hit_ratio=0.85)

    @staticmethod
    def input_generator(**config):
        return generate_moe_fp8_inputs(seq_len=config["seq_len"])

    def baseline_fn(
        self,
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
    ):
        """FlashInfer TRT-LLM FP8 MoE baseline implementation."""
        seq_len = hidden_states.shape[0]
        local_num_experts = gemm1_weights.shape[0]
        tile_tokens_dim = get_tile_tokens_dim(seq_len, TOP_K, NUM_EXPERTS_GLOBAL)

        return trtllm_fp8_block_scale_moe(
            routing_logits.to(torch.float32),
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale.to(torch.float32),
            gemm2_weights,
            gemm2_weights_scale.to(torch.float32),
            NUM_EXPERTS_GLOBAL,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            INTERMEDIATE_SIZE,
            int(local_expert_offset),
            local_num_experts,
            float(routed_scaling_factor),
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=2,
            use_shuffled_weight=False,
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
