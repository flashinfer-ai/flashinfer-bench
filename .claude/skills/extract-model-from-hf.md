# Extract Model from HuggingFace

Extract model configuration and architecture information from HuggingFace Hub.

## Description

This skill downloads and analyzes a model's configuration from HuggingFace, extracting key architectural information needed for FlashInfer-Bench integration.

## Parameters

- `model_id` (required): HuggingFace model repository ID (e.g., "moonshot-ai/kimi-k2")
- `output_dir` (optional): Directory to save output files (default: "./model_analysis")

## Usage

```bash
/extract-model-from-hf --model-id moonshot-ai/kimi-k2
/extract-model-from-hf --model-id meta-llama/Llama-3.3-70B-Instruct --output-dir ./models/llama33
```

## What This Skill Does

1. Downloads `config.json` from the specified HuggingFace repository
2. Parses key architectural parameters:
   - Model architecture type (e.g., LlamaForCausalLM, DeepseekV3ForCausalLM)
   - Number of layers
   - Hidden dimensions
   - Attention configuration (heads, KV heads, head dimension)
   - MLP configuration (intermediate size, activation)
   - Special features (MoE, MLA, etc.)
3. Extracts module structure by analyzing:
   - Layer normalization type and dimensions
   - Attention mechanism (GQA, MHA, MLA)
   - MLP structure (dense or MoE)
   - Positional encoding type
4. Generates output files:
   - `model_config.json`: Raw configuration from HuggingFace
   - `model_architecture.json`: Parsed architecture information
   - `module_mapping.json`: Suggested mapping to FlashInfer Definitions

## Output Format

### model_architecture.json

```json
{
  "model_id": "moonshot-ai/kimi-k2",
  "architecture_type": "KimiForCausalLM",
  "num_layers": 32,
  "hidden_size": 4096,
  "attention": {
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "type": "gqa"
  },
  "mlp": {
    "intermediate_size": 14336,
    "activation": "silu",
    "type": "dense"
  },
  "normalization": {
    "type": "rmsnorm",
    "epsilon": 1e-6
  },
  "special_features": []
}
```

### module_mapping.json

```json
{
  "input_layernorm": ["rmsnorm_h4096", "fused_add_rmsnorm_h4096"],
  "qkv_proj": ["gemm_n_6144_k_4096"],
  "attn": [
    "gqa_paged_decode_h32_kv8_d128_ps1",
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
    "gqa_ragged_prefill_causal_h32_kv8_d128"
  ],
  "o_proj": ["gemm_n_4096_k_4096"],
  "post_attention_layernorm": ["fused_add_rmsnorm_h4096"],
  "gate_up_proj": ["gemm_n_28672_k_4096"],
  "down_proj": ["gemm_n_4096_k_14336"]
}
```

## Requirements

- Python packages: `huggingface_hub`
- Network access to HuggingFace Hub
- Optional: HuggingFace token for gated models

## Implementation

When executed, this skill will:

1. Check if `huggingface_hub` is installed, install if needed
2. Use `hf_hub_download` to fetch `config.json`
3. Parse the config to identify:
   - Model type from `architectures` field
   - Layer count from `num_hidden_layers`
   - Hidden dimensions from `hidden_size`
   - Attention config from `num_attention_heads`, `num_key_value_heads`
   - MLP config from `intermediate_size`
4. Infer module structure based on model type:
   - Standard Transformer (Llama-style)
   - MoE Transformer (Qwen/DeepSeek-style)
   - MLA Transformer (DeepSeek V3-style)
5. Generate Definition name suggestions based on dimensions
6. Save all outputs to specified directory

## Notes

- For gated models, set `HF_TOKEN` environment variable
- The skill attempts to infer the module structure based on common patterns
- Review generated mappings as they may need manual adjustment for custom architectures
- Use output files as input for `generate-model-definition` skill
