# Extract Kernel Definitions

Extract kernel schemas and definitions from SGLang model implementations, with deduplication, and add them to flashinfer-trace dataset with vanilla Python reference implementations.

## Description

This skill analyzes SGLang model implementations to extract the complete set of GPU kernels used during inference. It identifies kernel types (MLA, MOE, GQA, RMSNorm, GEMM), extracts their parameters, generates Definition JSON schemas with Python reference implementations, and handles deduplication across multiple models.

## Usage

```bash
# Extract kernels from DeepSeek V3 model (MLA + MoE)
/extract-kernel-definitions --model-name deepseek_v3

# Extract from Llama (GQA)
/extract-kernel-definitions --model-name llama

# Extract from multiple models with automatic deduplication
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe

# Extract only prefill or decode kernels
/extract-kernel-definitions --model-name llama --execution-modes prefill
/extract-kernel-definitions --model-name llama --execution-modes decode
```

## Parameters

- `model_name` (required): Model name to extract kernels from (e.g., "deepseek_v3", "llama", "qwen2_moe")
- `execution_modes` (optional): Inference execution modes to analyze (default: ["prefill", "decode"])
- `include_quantized` (optional): Include quantized kernel variants (default: true)
- `deduplicate` (optional): Check for and skip existing definitions (default: true)

## Prerequisites

Run `/clone-repos` first to set up the `third_party/` directory with SGLang and flashinfer-trace.

## What This Skill Does

### Phase 1: Model Analysis

1. **Locate Model Implementation**:
   - Search `third_party/sglang/python/sglang/srt/models/{model_name}.py`
   - Identify model class (e.g., `DeepseekV3ForCausalLM`, `LlamaForCausalLM`)
   - Parse model architecture from config

2. **Identify Layer Components**:
   - Attention mechanism (GQA, MHA, MLA)
   - MLP/FFN structure (Dense, MoE)
   - Normalization layers (RMSNorm, LayerNorm)
   - Embedding and output projections

3. **Extract Execution Paths**:
   - Prefill path (batch processing, variable sequence length)
   - Decode path (single token, paged KV cache)

### Phase 2: Kernel Extraction

For each layer component, extract:

#### Attention Kernels
- **GQA**: `gqa_paged_decode`, `gqa_paged_prefill`, `gqa_ragged_prefill`
- **MLA**: `mla_paged_decode`, `mla_paged_prefill`
- **Parameters**: num_heads, num_kv_heads, head_dim, page_size, ckv_dim, kpe_dim

#### MoE Kernels
- Expert routing and selection
- Expert execution (FP8 quantized, block-scaled)
- **Parameters**: num_experts, topk, hidden_size, intermediate_size, group_size

#### Normalization Kernels
- `rmsnorm_h{hidden_size}`
- `fused_add_rmsnorm_h{hidden_size}`
- **Parameters**: hidden_size, epsilon

#### GEMM Kernels
- QKV projection, O projection
- Gate/Up projection, Down projection
- **Parameters**: M (variable), N (output dim), K (input dim)

### Phase 3: Deduplication

1. **Load Existing Definitions**:
   - Scan `third_party/flashinfer-trace/definitions/` for existing JSONs
   - Build index of definition names and signatures

2. **Compare Extracted Kernels**:
   - Check if kernel with same name exists
   - Verify parameter compatibility
   - Skip existing definitions, report what was skipped

3. **Handle Shared Kernels**:
   - Kernels like `rmsnorm_h4096` may be used by multiple models
   - Only create once, add model tags to existing definitions

### Phase 4: Definition Generation

For each new kernel, generate a Definition JSON:

1. **Generate Name**: Follow naming convention `{op_type}_{variant}_{params}`
   - Example: `mla_paged_decode_h16_ckv512_kpe64_ps1`
   - Example: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

2. **Define Axes**:
   - Constant axes: fixed at compile time (e.g., hidden_size, num_heads)
   - Variable axes: determined at runtime (e.g., batch_size, seq_len)

3. **Specify Inputs/Outputs**:
   - Tensor shapes using axis names
   - Data types (float16, bfloat16, float8_e4m3fn, etc.)

4. **Write Reference Implementation**:
   - Plain PyTorch implementation
   - Step-by-step computation (no high-level APIs)
   - Serves as mathematical specification

### Phase 5: Save Definitions

Output to `third_party/flashinfer-trace/definitions/{op_type}/{definition_name}.json`

## Output Format

### Definition JSON Example (MLA Decode)

```json
{
  "name": "mla_paged_decode_h16_ckv512_kpe64_ps1",
  "op_type": "mla_paged",
  "description": "Multi-head Latent Attention decode with paged KV cache",
  "tags": ["stage:decode", "model:deepseek-v3"],
  "axes": {
    "batch_size": { "type": "var" },
    "num_pages": { "type": "var" },
    "num_kv_indices": { "type": "var" },
    "num_qo_heads": { "type": "const", "value": 16 },
    "head_dim_ckv": { "type": "const", "value": 512 },
    "head_dim_kpe": { "type": "const", "value": 64 },
    "page_size": { "type": "const", "value": 1 }
  },
  "inputs": {
    "q_nope": { "shape": ["batch_size", "num_qo_heads", "head_dim_ckv"], "dtype": "float16" },
    "q_pe": { "shape": ["batch_size", "num_qo_heads", "head_dim_kpe"], "dtype": "float16" },
    "ckv_cache": { "shape": ["num_pages", "page_size", "head_dim_ckv"], "dtype": "float16" },
    "kpe_cache": { "shape": ["num_pages", "page_size", "head_dim_kpe"], "dtype": "float16" },
    "kv_indptr": { "shape": ["batch_size"], "dtype": "int32" },
    "kv_indices": { "shape": ["num_kv_indices"], "dtype": "int32" },
    "sm_scale": { "shape": null, "dtype": "float32" }
  },
  "outputs": {
    "output": { "shape": ["batch_size", "num_qo_heads", "head_dim_ckv"], "dtype": "float16" },
    "lse": { "shape": ["batch_size", "num_qo_heads"], "dtype": "float32" }
  },
  "reference": "import torch\nimport math\n\ndef run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):\n    ..."
}
```

### Definition JSON Example (MoE)

```json
{
  "name": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
  "op_type": "moe",
  "description": "DeepSeek-style MoE with FP8 block-scaled weights and top-8 expert routing",
  "tags": ["model:deepseek-v3", "quantization:float8_e4m3fn"],
  "axes": {
    "seq_len": { "type": "var" },
    "num_experts": { "type": "const", "value": 32 },
    "topk": { "type": "const", "value": 8 },
    "num_groups": { "type": "const", "value": 8 },
    "group_size": { "type": "const", "value": 4 },
    "hidden_size": { "type": "const", "value": 7168 },
    "intermediate_size": { "type": "const", "value": 2048 }
  },
  "inputs": {
    "hidden_states": { "shape": ["seq_len", "hidden_size"], "dtype": "float8_e4m3fn" },
    "hidden_states_scale": { "shape": ["seq_len"], "dtype": "float32" },
    "routing_logits": { "shape": ["seq_len", "num_experts"], "dtype": "float32" },
    "expert_weights": { "shape": ["num_experts", "intermediate_size", "hidden_size"], "dtype": "float8_e4m3fn" },
    "expert_weights_scale": { "shape": ["num_experts", "intermediate_size"], "dtype": "float32" }
  },
  "outputs": {
    "output": { "shape": ["seq_len", "hidden_size"], "dtype": "bfloat16" }
  },
  "reference": "..."
}
```

## Implementation Steps

When executing this skill:

1. **Locate model file**:
   ```bash
   ls third_party/sglang/python/sglang/srt/models/ | grep -i {model_name}
   ```

2. **Read model implementation**:
   - Parse the Python file
   - Identify forward() methods
   - Extract kernel calls (attention, MoE, norm, GEMM)

3. **Extract kernel parameters from model config**:
   - Look for config class (e.g., `DeepseekV3Config`)
   - Extract: num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, etc.

4. **Check existing definitions**:
   ```bash
   ls third_party/flashinfer-trace/definitions/*/
   ```

5. **Generate new definition JSONs**:
   - Create directory if needed: `mkdir -p third_party/flashinfer-trace/definitions/{op_type}/`
   - Write JSON file with reference implementation

6. **Report results**:
   - List new definitions created
   - List existing definitions skipped (deduplication)
   - List kernels shared across models

## SGLang Code Patterns to Look For

### Attention Kernel Calls

```python
# GQA decode (paged)
flashinfer.batch_decode_with_paged_kv_cache(...)

# MLA decode
from sglang.srt.layers.attention.mla_decode import mla_decode_attention
```

### MoE Kernel Calls

```python
# DeepSeek MoE
from sglang.srt.layers.moe.fused_moe import fused_moe
fused_moe(hidden_states, w1, w2, w3, topk_weights, topk_ids, ...)
```

### Normalization

```python
# RMSNorm
from sglang.srt.layers.layernorm import RMSNorm
self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
```

## Kernel Type to Model Mapping

| Model | Attention | MLP | Normalization |
|-------|-----------|-----|---------------|
| DeepSeek V3/R1 | MLA | MoE | RMSNorm |
| Llama 3.x | GQA | Dense | RMSNorm |
| Qwen2 MoE | GQA | MoE | RMSNorm |
| Mixtral | GQA | MoE | RMSNorm |

## Error Handling

### Model Not Found
- **Error**: Model file doesn't exist in SGLang
- **Handling**: List available models, suggest closest match

### Unsupported Kernel Type
- **Error**: Unknown kernel pattern in model
- **Handling**: Log warning, skip kernel, report for manual review

### Duplicate with Conflict
- **Error**: Definition exists with different parameters
- **Handling**: Create new versioned definition, flag for review

## Integration with Other Skills

```bash
# Complete workflow
/clone-repos

# Extract from multiple models
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe

# Add tests for new definitions
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe
```

## Notes

- Each kernel may have multiple variants for different execution modes (prefill/decode)
- Quantized kernels are treated as separate definitions
- Reference implementations prioritize clarity over performance
- Model tags enable tracing which models use each kernel
- Deduplication is essential for maintaining a clean dataset

## See Also

- [clone-repos](./clone-repos.md)
- [add-reference-tests](./add-reference-tests.md)
- [workflow](./workflow.md)
