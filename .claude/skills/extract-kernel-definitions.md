# Extract Kernel Definitions

Extract kernel schemas and definitions from SGLang model implementations, with deduplication, and add them to flashinfer-trace dataset with vanilla Python reference implementations.

## Description

This skill analyzes SGLang model implementations to extract the complete set of GPU kernels used during inference. It identifies kernel types (MLA, MOE, GQA, RMSNorm, GEMM), extracts their parameters, generates Definition JSON schemas with Python reference implementations, and handles deduplication across multiple models.

## Parameters

- `model_name` (required): Model name to extract kernels from (e.g., "deepseek_v3", "llama", "qwen2_moe")
- `repos_config` (optional): Path to repos_config.json from clone-repos skill (default: "./repos/repos_config.json")
- `sglang_path` (optional): Direct path to SGLang repository (overrides repos_config)
- `output_dir` (optional): Output directory for extracted definitions (default: uses flashinfer-trace from repos_config)
- `execution_modes` (optional): Inference execution modes to analyze (default: ["prefill", "decode"])
- `include_quantized` (optional): Include quantized kernel variants (default: true)
- `deduplicate` (optional): Check for and skip existing definitions (default: true)

## Usage

```bash
# Extract kernels from DeepSeek V3 model
/extract-kernel-definitions --model-name deepseek_v3

# Extract from multiple models with deduplication
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe

# Extract with custom paths
/extract-kernel-definitions \
  --model-name kimi \
  --sglang-path ~/repos/sglang \
  --output-dir ~/repos/flashinfer-trace/definitions

# Extract only prefill kernels
/extract-kernel-definitions \
  --model-name llama \
  --execution-modes prefill
```

## What This Skill Does

### Phase 1: Model Analysis

1. **Locate Model Implementation**:
   - Search `{sglang_path}/python/sglang/srt/models/{model_name}.py`
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
   - Speculative decoding paths (if applicable)

### Phase 2: Kernel Extraction

For each layer component, extract:

1. **Attention Kernels**:
   - GQA: `gqa_paged_decode`, `gqa_paged_prefill`, `gqa_ragged_prefill`
   - MLA: `mla_paged_decode`, `mla_paged_prefill`, `mla_ragged_prefill`
   - Parameters: num_heads, num_kv_heads, head_dim, page_size

2. **MoE Kernels**:
   - Expert routing and selection
   - Expert execution (FP8 quantized, block-scaled)
   - Parameters: num_experts, topk, hidden_size, intermediate_size, group_size

3. **Normalization Kernels**:
   - RMSNorm: `rmsnorm_h{hidden_size}`
   - Fused Add RMSNorm: `fused_add_rmsnorm_h{hidden_size}`
   - Parameters: hidden_size, epsilon

4. **GEMM Kernels**:
   - QKV projection, O projection
   - Gate/Up projection, Down projection
   - Parameters: M (variable), N (output dim), K (input dim)

### Phase 3: Deduplication

1. **Load Existing Definitions**:
   - Scan `{flashinfer_trace_path}/definitions/` for existing JSONs
   - Build index of definition names and signatures

2. **Compare Extracted Kernels**:
   - Check if kernel with same name exists
   - Verify parameter compatibility
   - Flag conflicts if parameters differ

3. **Generate Dedup Report**:
   ```json
   {
     "new_definitions": ["mla_paged_decode_h16_ckv512_kpe64_ps1", ...],
     "existing_definitions": ["rmsnorm_h4096", ...],
     "conflicts": [],
     "shared_across_models": {"rmsnorm_h4096": ["llama", "qwen"]}
   }
   ```

### Phase 4: Definition Generation

For each new kernel, generate a Definition JSON:

1. **Generate Name**:
   - Follow naming convention: `{op_type}_{variant}_{params}`
   - Example: `mla_paged_decode_h16_ckv512_kpe64_ps1`

2. **Define Axes**:
   - Constant axes: fixed at compile time (e.g., hidden_size, num_heads)
   - Variable axes: determined at runtime (e.g., batch_size, seq_len)

3. **Specify Inputs/Outputs**:
   - Tensor shapes using axis names
   - Data types (float16, bfloat16, float8_e4m3fn, etc.)
   - Scalar parameters (epsilon, scale factors)

4. **Write Reference Implementation**:
   - Plain PyTorch implementation
   - Step-by-step computation (no high-level APIs)
   - Serves as mathematical specification

### Phase 5: Save Definitions

1. **Create Directory Structure**:
   ```
   definitions/
   ├── {op_type}/
   │   ├── {definition_name}.json
   ```

2. **Write JSON Files**:
   - Formatted with proper indentation
   - Include model tags for traceability

3. **Generate Summary Report**:
   - List of created definitions
   - Model-to-definition mapping
   - Suggestions for missing definitions

## Output Format

### Definition JSON Example (MLA Decode)

```json
{
  "name": "mla_paged_decode_h16_ckv512_kpe64_ps1",
  "op_type": "mla_paged",
  "description": "Multi-head Latent Attention decode with paged KV cache",
  "tags": [
    "stage:decode",
    "model:deepseek-v3"
  ],
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
  "reference": "import torch\nimport math\n\ndef run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):\n    batch_size = q_nope.shape[0]\n    num_heads = q_nope.shape[1]\n    head_dim_ckv = q_nope.shape[2]\n    head_dim_kpe = q_pe.shape[2]\n    \n    outputs = []\n    lses = []\n    \n    for b in range(batch_size):\n        # Get KV indices for this batch\n        start_idx = kv_indptr[b] if b > 0 else 0\n        end_idx = kv_indptr[b + 1] if b < batch_size - 1 else len(kv_indices)\n        seq_indices = kv_indices[start_idx:end_idx]\n        \n        # Gather KV cache\n        k_nope = ckv_cache[seq_indices, 0, :]  # [seq_len, ckv_dim]\n        k_pe = kpe_cache[seq_indices, 0, :]    # [seq_len, kpe_dim]\n        \n        # Compute attention scores\n        # q: [num_heads, head_dim], k: [seq_len, head_dim]\n        q_full = torch.cat([q_nope[b], q_pe[b]], dim=-1)  # [num_heads, ckv+kpe]\n        k_full = torch.cat([k_nope, k_pe], dim=-1)         # [seq_len, ckv+kpe]\n        \n        scores = torch.matmul(q_full, k_full.T) * sm_scale  # [num_heads, seq_len]\n        \n        # Softmax\n        max_scores = scores.max(dim=-1, keepdim=True)[0]\n        exp_scores = torch.exp(scores - max_scores)\n        sum_exp = exp_scores.sum(dim=-1, keepdim=True)\n        attn_weights = exp_scores / sum_exp\n        \n        # Compute output (using ckv only)\n        output = torch.matmul(attn_weights, k_nope)  # [num_heads, ckv_dim]\n        lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))\n        \n        outputs.append(output)\n        lses.append(lse)\n    \n    return torch.stack(outputs), torch.stack(lses)"
}
```

### Definition JSON Example (MoE)

```json
{
  "name": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
  "op_type": "moe",
  "description": "DeepSeek-style MoE with FP8 block-scaled weights and top-8 expert routing",
  "tags": [
    "model:deepseek-v3",
    "quantization:float8_e4m3fn"
  ],
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

## Kernel Type Reference

### MLA (Multi-head Latent Attention)

Found in DeepSeek V3/R1 models:
- Decomposes KV into compressed (CKV) and positional encoding (KPE) components
- Variants: `mla_paged_decode`, `mla_paged_prefill`, `mla_ragged_prefill`
- Key dimensions: num_qo_heads, head_dim_ckv, head_dim_kpe, page_size

### MoE (Mixture of Experts)

Found in DeepSeek, Qwen, Mixtral models:
- Sparse expert selection with top-k routing
- Block-scaled FP8 quantization for efficiency
- Key dimensions: num_experts, topk, hidden_size, intermediate_size

### GQA (Grouped-Query Attention)

Found in Llama 3.x, Qwen, and other models:
- Multiple query heads share fewer KV heads
- Variants: `gqa_paged_decode`, `gqa_paged_prefill`, `gqa_ragged_prefill`
- Key dimensions: num_heads, num_kv_heads, head_dim, page_size

### RMSNorm

Universal across models:
- Root Mean Square Layer Normalization
- Variants: basic and fused with residual add
- Key dimensions: hidden_size

### GEMM

Universal across models:
- Linear projections throughout the model
- Quantized variants (FP8, INT8)
- Key dimensions: M (variable), N (output), K (input)

## SGLang Code Pattern Reference

### Attention Kernel Calls

```python
# GQA decode (paged)
flashinfer.batch_decode_with_shared_prefix_paged_kv_cache(
    q, kv_cache, kv_indices, kv_indptr, num_kv_heads, head_dim
)

# MLA decode
mla_decode_attention(
    q_nope, q_pe, ckv_cache, kpe_cache, ...
)
```

### MoE Kernel Calls

```python
# DeepSeek MoE
fused_moe(
    hidden_states, w1, w2, w3,
    topk_weights, topk_ids,
    inplace=True
)
```

### Normalization

```python
# RMSNorm
input_layernorm(hidden_states, residual)
# or
rmsnorm_forward(hidden_states, weight, eps)
```

## Requirements

- Python packages:
  - `ast` (standard library)
  - `json` (standard library)
  - `torch` (for reference implementation verification)
- Access to SGLang repository
- Write access to flashinfer-trace definitions directory

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

### Reference Implementation Error
- **Error**: Generated reference has syntax errors
- **Handling**: Validate with ast.parse, fix or report

## Integration with Other Skills

```bash
# Complete workflow
/clone-repos --target-dir ./repos

# Extract from multiple models
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe

# Add tests for new definitions
/add-reference-tests --definitions-dir ./repos/flashinfer-trace/definitions
```

## Notes

- Each kernel may have multiple variants for different execution modes
- Quantized kernels are treated as separate definitions
- Reference implementations prioritize clarity over performance
- Model tags enable tracing which models use each kernel
- Deduplication is essential for maintaining a clean dataset

## See Also

- [clone-repos](./clone-repos.md)
- [add-reference-tests](./add-reference-tests.md)
- [Definition Schema](../../docs/flashinfer_trace/definition.md)
- [Op Type Schemas](../../docs/op_type_schema/)
