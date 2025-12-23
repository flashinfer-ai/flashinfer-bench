# Generate Model Definition

Generate TypeScript model definition file for FlashInfer-Bench web interface.

## Description

This skill takes model architecture information and SGLang baseline implementation data to generate a complete TypeScript model definition for `web/apps/web/data/models.ts`.

## Parameters

- `config` (required): Path to model architecture JSON (from `extract-model-from-hf`)
- `sglang_impl` (optional): Path to SGLang implementation JSON (from `find-sglang-baseline`)
- `model_name` (required): Model identifier for the definition (e.g., "kimi-k2")
- `output` (optional): Output path (default: "web/apps/web/data/models.ts")
- `append` (optional): Whether to append to existing file (default: true)

## Usage

```bash
/generate-model-definition \
  --config model_analysis/model_architecture.json \
  --model-name kimi-k2

/generate-model-definition \
  --config model_analysis/model_architecture.json \
  --sglang-impl model_analysis/sglang_implementation.json \
  --model-name kimi-k2 \
  --output web/apps/web/data/models.ts
```

## What This Skill Does

1. Loads model architecture information from JSON
2. Optionally loads SGLang baseline implementation details
3. Determines model architecture pattern:
   - Standard Transformer (Llama-style)
   - MoE Transformer (Qwen/DeepSeek-style)
   - MLA Transformer (DeepSeek V3-style)
   - Custom hybrid architectures
4. Generates hierarchical module structure:
   - Top-level model block
   - Embedding layer
   - Decoder layers with proper parent references
   - Individual sublayers (normalization, attention, MLP)
   - Final norm and lm_head
5. Maps each module to appropriate Definitions:
   - Normalization: `rmsnorm_h{size}`, `fused_add_rmsnorm_h{size}`
   - GEMM: `gemm_n_{out}_k_{in}`
   - Attention: `gqa_*`, `mla_*` based on architecture
   - MoE: `moe_*` with routing parameters
6. Generates TypeScript code conforming to Model interface
7. Appends to existing models.ts or creates new file

## Output Format

The generated TypeScript will follow this structure:

```typescript
{
  id: "kimi-k2",
  name: "Kimi K2",
  description: "Moonshot AI Kimi K2 model",
  modules: {
    KimiForCausalLM: {
      count: 1,
      type: "block",
      definitions: [],
    },
    KimiModel: {
      count: 1,
      parent: "KimiForCausalLM",
      type: "block",
      definitions: [],
    },
    embed_tokens: {
      count: 1,
      parent: "KimiModel",
      type: "layer",
      definitions: [],
    },
    layers: {
      count: 32,
      parent: "KimiModel",
      type: "block",
      definitions: [],
    },
    KimiDecoderLayer: {
      count: 32,
      parent: "layers",
      type: "block",
      definitions: [],
    },
    input_layernorm: {
      count: 32,
      parent: "KimiDecoderLayer",
      type: "layer",
      definitions: ["rmsnorm_h4096", "fused_add_rmsnorm_h4096"],
    },
    self_attn: {
      count: 32,
      parent: "KimiDecoderLayer",
      type: "block",
      definitions: [],
    },
    qkv_proj: {
      count: 32,
      parent: "self_attn",
      type: "layer",
      definitions: ["gemm_n_6144_k_4096"],
    },
    rotary_emb: {
      count: 32,
      parent: "self_attn",
      type: "layer",
      definitions: [],
    },
    attn: {
      count: 32,
      parent: "self_attn",
      type: "layer",
      definitions: [
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
        "gqa_paged_decode_h32_kv8_d128_ps1",
        "gqa_ragged_prefill_causal_h32_kv8_d128",
      ],
    },
    o_proj: {
      count: 32,
      parent: "self_attn",
      type: "layer",
      definitions: ["gemm_n_4096_k_4096"],
    },
    post_attention_layernorm: {
      count: 32,
      parent: "KimiDecoderLayer",
      type: "layer",
      definitions: ["fused_add_rmsnorm_h4096"],
    },
    mlp: {
      count: 32,
      parent: "KimiDecoderLayer",
      type: "block",
      definitions: [],
    },
    gate_up_proj: {
      count: 32,
      parent: "mlp",
      type: "layer",
      definitions: ["gemm_n_28672_k_4096"],
    },
    act_fn: {
      count: 32,
      parent: "mlp",
      type: "layer",
      definitions: [],
    },
    down_proj: {
      count: 32,
      parent: "mlp",
      type: "layer",
      definitions: ["gemm_n_4096_k_14336"],
    },
    norm: {
      count: 1,
      parent: "KimiModel",
      type: "layer",
      definitions: ["fused_add_rmsnorm_h4096"],
    },
    lm_head: {
      count: 1,
      parent: "KimiForCausalLM",
      type: "layer",
      definitions: [],
    },
  },
}
```

## Definition Naming Conventions

The skill follows these conventions for generating Definition names:

### RMSNorm
- Basic: `rmsnorm_h{hidden_size}`
- Fused with residual: `fused_add_rmsnorm_h{hidden_size}`

### GEMM
- Format: `gemm_n_{output_dim}_k_{input_dim}`
- Examples:
  - QKV projection: `gemm_n_{3*hidden}_k_{hidden}`
  - O projection: `gemm_n_{hidden}_k_{hidden}`
  - Gate-up: `gemm_n_{2*intermediate}_k_{hidden}`
  - Down: `gemm_n_{hidden}_k_{intermediate}`

### GQA (Group Query Attention)
- Paged decode: `gqa_paged_decode_h{heads}_kv{kv_heads}_d{head_dim}_ps1`
- Paged prefill: `gqa_paged_prefill_causal_h{heads}_kv{kv_heads}_d{head_dim}_ps1`
- Ragged prefill: `gqa_ragged_prefill_causal_h{heads}_kv{kv_heads}_d{head_dim}`

### MLA (Multi-Head Latent Attention)
- Paged decode: `mla_paged_decode_h{heads}_ckv{ckv_dim}_kpe{kpe_dim}_ps1`
- Paged prefill: `mla_paged_prefill_causal_h{heads}_ckv{ckv_dim}_kpe{kpe_dim}_ps1`
- Ragged prefill: `mla_ragged_prefill_causal_h{heads}_qk{qk_dim}_vo{vo_dim}`

### MoE
- Format: `moe_fp8_block_scale_ds_routing_topk{topk}_ng{num_groups}_kg{group_size}_e{experts}_h{hidden}_i{intermediate}`

## Architecture Pattern Detection

The skill automatically detects architecture patterns:

1. **Standard Transformer (Llama-style)**:
   - Indicators: Standard attention with `num_key_value_heads`
   - Modules: Standard self_attn, mlp blocks
   - Template: Uses Llama structure

2. **MoE Transformer**:
   - Indicators: `num_experts` in config, MoE in model class name
   - Modules: Adds moe_gate, moe_topk, moe_experts
   - Template: Uses Qwen3/DeepSeek structure

3. **MLA Transformer (DeepSeek V3)**:
   - Indicators: `kv_lora_rank`, `qk_nope_head_dim` in config
   - Modules: Complex attention with a_proj, b_proj, layernorms
   - Template: Uses DeepSeek V3 structure

4. **Custom**:
   - Falls back to generic transformer structure
   - Uses information from SGLang implementation if available

## Requirements

- Node.js and TypeScript (for validation)
- Python packages: `json` (standard library)

## Implementation

When executed, this skill will:

1. **Load Input Data**:
   - Parse model_architecture.json
   - Parse sglang_implementation.json (if provided)
   - Validate required fields

2. **Detect Architecture Pattern**:
   - Check config for MoE indicators
   - Check for MLA-specific parameters
   - Default to standard transformer

3. **Generate Module Hierarchy**:
   - Start with top-level model class
   - Add embedding layer
   - Generate decoder layer structure based on pattern
   - Add sublayers with appropriate parents
   - Add final norm and lm_head

4. **Calculate Dimensions**:
   - QKV dimension: `(num_heads * head_dim + 2 * num_kv_heads * head_dim)`
   - Gate-up dimension: `2 * intermediate_size`
   - Head dimension: `hidden_size / num_heads`

5. **Map to Definitions**:
   - Use dimension information to generate names
   - Apply naming conventions
   - Add all relevant variants (paged/ragged, decode/prefill)

6. **Generate TypeScript Code**:
   - Format according to Model interface
   - Ensure proper indentation and structure
   - Add comments with source information

7. **Update models.ts**:
   - Read existing file
   - Parse to find insertion point
   - Append new model definition
   - Preserve existing models
   - Format and validate TypeScript

## Validation

The skill performs these validations:

- All parent references exist
- No circular dependencies
- Count values are positive integers
- Type is either "block" or "layer"
- Definitions array contains valid strings
- TypeScript syntax is valid

## Notes

- Review generated definitions before use
- Some custom architectures may need manual adjustment
- Definition names must match existing definitions in the dataset
- For new operations, create corresponding Definition JSONs first
- The skill preserves existing models when appending

## Example Workflow

Complete workflow with all skills:

```bash
# Step 1: Extract from HuggingFace
/extract-model-from-hf --model-id moonshot-ai/kimi-k2

# Step 2: Find SGLang baseline
/find-sglang-baseline --model-name kimi

# Step 3: Generate model definition
/generate-model-definition \
  --config model_analysis/model_architecture.json \
  --sglang-impl model_analysis/sglang_implementation.json \
  --model-name kimi-k2

# Step 4: Verify in web UI
cd web/apps/web && pnpm dev
```
