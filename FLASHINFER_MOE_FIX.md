# FlashInfer MoE Solution Fix for flashinfer-python 0.6.2

## Issue

The `flashinfer_moe` solution in the HuggingFace dataset (`flashinfer-ai/flashinfer-trace`) uses the deprecated parameter `tile_tokens_dim` which was removed in `flashinfer-python 0.6.2` and replaced with `tune_max_num_tokens`.

## Fix

The corrected solution JSON is provided below. To apply this fix:

### Option 1: Create the solution file locally

Create the file `flashinfer_trace/solutions/moe/flashinfer_moe.json` with the following content:

```json
{
  "name": "flashinfer_moe",
  "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
  "description": "FlashInfer MoE FP8 block-scale implementation with DeepSeek-V3 routing. Updated for flashinfer-python 0.6.2 API compatibility.",
  "author": "flashinfer-ai",
  "spec": {
    "language": "python",
    "target_hardware": [
      "cuda"
    ],
    "entry_point": "main.py::run",
    "dependencies": [
      "flashinfer-python>=0.6.2",
      "torch"
    ],
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nfrom flashinfer.fused_moe import trtllm_fp8_block_scale_moe\n\n\n@torch.no_grad()\ndef run(\n    routing_logits: torch.Tensor,\n    routing_bias: torch.Tensor,\n    hidden_states: torch.Tensor,\n    hidden_states_scale: torch.Tensor,\n    gemm1_weights: torch.Tensor,\n    gemm1_weights_scale: torch.Tensor,\n    gemm2_weights: torch.Tensor,\n    gemm2_weights_scale: torch.Tensor,\n    local_expert_offset: int,\n    routed_scaling_factor: float,\n):\n    \"\"\"\n    FlashInfer MoE FP8 block-scale wrapper for DeepSeek-V3/R1 routing.\n\n    Fixed for flashinfer-python 0.6.2 compatibility:\n    - Changed tile_tokens_dim → tune_max_num_tokens\n    \"\"\"\n    # Constants for DeepSeek-V3/R1\n    NUM_EXPERTS = 256\n    TOP_K = 8\n    N_GROUP = 8\n    TOPK_GROUP = 4\n    INTERMEDIATE_SIZE = 2048\n    LOCAL_NUM_EXPERTS = 32\n\n    # Get sequence length\n    seq_len = routing_logits.shape[0]\n\n    # Call flashinfer kernel with corrected parameter name\n    return trtllm_fp8_block_scale_moe(\n        routing_logits=routing_logits.to(torch.float32),\n        routing_bias=routing_bias,\n        hidden_states=hidden_states,\n        hidden_states_scale=hidden_states_scale,\n        gemm1_weights=gemm1_weights,\n        gemm1_weights_scale=gemm1_weights_scale.to(torch.float32),\n        gemm2_weights=gemm2_weights,\n        gemm2_weights_scale=gemm2_weights_scale.to(torch.float32),\n        num_experts=NUM_EXPERTS,\n        top_k=TOP_K,\n        n_group=N_GROUP,\n        topk_group=TOPK_GROUP,\n        intermediate_size=INTERMEDIATE_SIZE,\n        local_expert_offset=local_expert_offset,\n        local_num_experts=LOCAL_NUM_EXPERTS,\n        routed_scaling_factor=routed_scaling_factor,\n        routing_method_type=2,\n        use_shuffled_weight=False,\n        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),\n    )\n"
    }
  ]
}
```

### Option 2: Use the provided script

Run the provided script to automatically generate the solution file:

```bash
python3 scripts/fix_flashinfer_moe_solution.py
```

### Option 3: Update HuggingFace dataset

The maintainers should update the `flashinfer_moe.json` file in the HuggingFace dataset at:
`https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace`

## Changes Made

The key change is on line 58 of the solution source code:

```python
# OLD (flashinfer < 0.6.2):
tile_tokens_dim=tile_tokens_dim,

# NEW (flashinfer >= 0.6.2):
tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
```

## Testing

After applying the fix, verify it works:

```bash
flashinfer-bench run --local ./flashinfer-trace \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions flashinfer_moe
```

All 19 workloads should now pass instead of failing with `RUNTIME_ERROR`.

## References

- Issue: #177
- FlashInfer 0.6.2 API changes: Parameter `tile_tokens_dim` removed, replaced with `tune_max_num_tokens`
- Test file showing correct usage: `flashinfer_trace/tests/references/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py:418,620`
