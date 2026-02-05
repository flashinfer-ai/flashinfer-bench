#!/usr/bin/env python3
"""
Script to generate the corrected flashinfer_moe solution for flashinfer 0.6.2 compatibility.

This script creates the flashinfer_moe solution JSON file with the updated parameter:
- tile_tokens_dim (removed in 0.6.2) → tune_max_num_tokens (current API)

The solution calls flashinfer.fused_moe.trtllm_fp8_block_scale_moe with the correct parameters.
"""

import json
from pathlib import Path

# Define the corrected solution source code
SOLUTION_SOURCE = '''import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    FlashInfer MoE FP8 block-scale wrapper for DeepSeek-V3/R1 routing.

    Fixed for flashinfer-python 0.6.2 compatibility:
    - Changed tile_tokens_dim → tune_max_num_tokens
    """
    # Constants for DeepSeek-V3/R1
    NUM_EXPERTS = 256
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    INTERMEDIATE_SIZE = 2048
    LOCAL_NUM_EXPERTS = 32

    # Get sequence length
    seq_len = routing_logits.shape[0]

    # Call flashinfer kernel with corrected parameter name
    return trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits.to(torch.float32),
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale.to(torch.float32),
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale.to(torch.float32),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=local_expert_offset,
        local_num_experts=LOCAL_NUM_EXPERTS,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=2,  # DeepSeek-V3 routing
        use_shuffled_weight=False,
        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),  # FIXED: was tile_tokens_dim
    )
'''

# Define the solution metadata
SOLUTION_JSON = {
    "name": "flashinfer_moe",
    "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "description": "FlashInfer MoE FP8 block-scale implementation with DeepSeek-V3 routing. Updated for flashinfer-python 0.6.2 API compatibility.",
    "author": "flashinfer-ai",
    "spec": {
        "language": "python",
        "target_hardware": ["cuda"],
        "entry_point": "main.py::run",
        "dependencies": ["flashinfer-python>=0.6.2", "torch"],
        "destination_passing_style": False
    },
    "sources": [
        {
            "path": "main.py",
            "content": SOLUTION_SOURCE
        }
    ]
}


def main():
    """Generate the corrected flashinfer_moe solution JSON file."""
    # Determine output path
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "flashinfer_trace" / "solutions" / "moe"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "flashinfer_moe.json"

    # Write the solution JSON
    with open(output_file, "w") as f:
        json.dump(SOLUTION_JSON, f, indent=2)

    print(f"✓ Generated corrected flashinfer_moe solution: {output_file}")
    print("\nChanges:")
    print("  - Replaced deprecated 'tile_tokens_dim' with 'tune_max_num_tokens'")
    print("  - Updated for flashinfer-python 0.6.2 API")
    print("\nTo use this solution, ensure flashinfer_trace/solutions is in your dataset path.")


if __name__ == "__main__":
    main()
