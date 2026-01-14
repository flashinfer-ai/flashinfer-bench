---
name: extract-kernel-definitions
description: Extract kernel schemas and definitions from SGLang model implementations with deduplication. Use when adding a new model, extracting GPU kernels (MLA, MoE, GQA, RMSNorm, GEMM), or generating Definition JSON files for flashinfer_trace.
---

# Extract Kernel Definitions

Extract kernel schemas and definitions from SGLang model implementations, with deduplication, and add them to `./flashinfer_trace/` with vanilla Python reference implementations.

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

Run `/clone-repos` first to set up the `third_party/` directory with SGLang and FlashInfer. The `flashinfer_trace/` directory is already included in this project.

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
   - Scan `flashinfer_trace/definitions/` for existing JSONs
   - Build index of definition names and signatures

2. **Compare Extracted Kernels**:
   - Check if kernel with same name exists
   - Verify parameter compatibility
   - Skip existing definitions, report what was skipped

3. **Handle Shared Kernels**:
   - Kernels like `rmsnorm_h4096` may be used by multiple models
   - Only create once, add model tags to existing definitions

### Phase 4: Definition Generation

For each new kernel, generate a Definition JSON following the standards below.

## Definition JSON Standards

### Naming Convention

Follow the pattern: `{op_type}_{variant}_{key_params}`

**Parameter Abbreviations:**
- `h` = num_heads or hidden_size (context-dependent)
- `kv` = num_kv_heads
- `d` = head_dim
- `ps` = page_size
- `ckv` = compressed_kv_dim
- `kpe` = key_positional_encoding_dim
- `e` = num_experts
- `i` = intermediate_size
- `topk` = top_k_experts
- `ng` = n_group
- `kg` = topk_group
- `v` = vocab_size

**Examples by op_type:**
- Attention: `gqa_paged_decode_h32_kv8_d128_ps1`, `mla_paged_prefill_h16_ckv512_kpe64_ps1`
- MoE: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Normalization: `rmsnorm_h4096`, `fused_add_rmsnorm_h7168`
- Sampling: `top_k_sampling_from_probs_v129280`

### Tag Patterns

Tags follow the pattern `{category}:{value}`:

| Category | Values | Description |
|----------|--------|-------------|
| `status` | `verified`, `unverified` | Whether reference implementation is validated |
| `stage` | `decode`, `prefill` | Inference execution mode |
| `model` | `deepseek-v3`, `deepseek-r1`, `llama-3.1-8b`, etc. | Associated model(s) |
| `quantization` | `float8_e4m3fn`, `nvfp4`, `int8`, `int4` | Quantization format |
| `routing` | `pre-computed`, `on-the-fly` | For MoE routing type |

**Example tags array:**
```json
"tags": [
  "stage:decode",
  "status:verified",
  "model:deepseek-v3",
  "model:deepseek-r1",
  "quantization:float8_e4m3fn"
]
```

### Axes Structure

```json
"axes": {
  "batch_size": {
    "type": "var",
    "description": "Batch size (number of sequences)"
  },
  "num_qo_heads": {
    "type": "const",
    "value": 16,
    "description": "Number of query heads after tensor parallel split (128/8=16)."
  }
}
```

**Rules:**
- Variable axes (`type: "var"`): runtime dimensions like batch_size, seq_len, num_pages
- Constant axes (`type: "const"`): model-specific values with `value` field
- Always include `description` for complex or model-specific axes

### Constraints Field

Add constraints for input validation:

```json
"constraints": [
  "len_indptr == batch_size + 1",
  "num_kv_indices == kv_indptr[-1].item()"
]
```

### Inputs/Outputs Structure

```json
"inputs": {
  "tensor_name": {
    "shape": ["axis1", "axis2", "axis3"],
    "dtype": "bfloat16",
    "description": "Description of the tensor"
  },
  "scalar_input": {
    "shape": null,
    "dtype": "float32",
    "description": "Scalar parameter"
  }
}
```

**dtype values:** `float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`, `int32`, `int64`

### Reference Implementation

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
   - See "Reference Implementation Sources" section below

### Phase 5: Save Definitions

Output to `flashinfer_trace/definitions/{op_type}/{definition_name}.json`

## Output Format

### Definition JSON Example (MLA Decode)

```json
{
  "name": "mla_paged_decode_h16_ckv512_kpe64_ps1",
  "description": "Batched Multi-head Latent Attention decode with a paged KV cache. Captured from DeepSeek-V3 with tensor parallel size 8.",
  "op_type": "mla_paged",
  "tags": [
    "stage:decode",
    "status:verified",
    "model:deepseek-v3",
    "model:deepseek-r1"
  ],
  "axes": {
    "batch_size": { "type": "var" },
    "num_qo_heads": {
      "type": "const",
      "value": 16,
      "description": "Number of query heads after tensor parallel split (128/8=16)."
    },
    "head_dim_ckv": { "type": "const", "value": 512 },
    "head_dim_kpe": { "type": "const", "value": 64 },
    "page_size": { "type": "const", "value": 1 },
    "num_pages": { "type": "var", "description": "Total number of allocated pages in the KV cache." },
    "len_indptr": { "type": "var", "description": "Length of kv_indptr array." },
    "num_kv_indices": { "type": "var", "description": "Total number of KV page indices." }
  },
  "constraints": [
    "len_indptr == batch_size + 1",
    "num_kv_indices == kv_indptr[-1].item()"
  ],
  "inputs": {
    "q_nope": {
      "shape": ["batch_size", "num_qo_heads", "head_dim_ckv"],
      "dtype": "bfloat16",
      "description": "Query tensor without positional encoding component."
    },
    "q_pe": {
      "shape": ["batch_size", "num_qo_heads", "head_dim_kpe"],
      "dtype": "bfloat16",
      "description": "Query positional encoding component."
    },
    "ckv_cache": {
      "shape": ["num_pages", "page_size", "head_dim_ckv"],
      "dtype": "bfloat16",
      "description": "Compressed key-value cache."
    },
    "kpe_cache": {
      "shape": ["num_pages", "page_size", "head_dim_kpe"],
      "dtype": "bfloat16",
      "description": "Key positional encoding cache."
    },
    "kv_indptr": {
      "shape": ["len_indptr"],
      "dtype": "int32",
      "description": "KV page offsets for each sequence."
    },
    "kv_indices": {
      "shape": ["num_kv_indices"],
      "dtype": "int32",
      "description": "Page indices for KV cache lookups."
    },
    "sm_scale": {
      "shape": null,
      "dtype": "float32",
      "description": "Softmax scale. Default is (1/sqrt(128 + 64) = 1/sqrt(192))."
    }
  },
  "outputs": {
    "output": { "shape": ["batch_size", "num_qo_heads", "head_dim_ckv"], "dtype": "bfloat16" },
    "lse": {
      "shape": ["batch_size", "num_qo_heads"],
      "dtype": "float32",
      "description": "The 2-based log-sum-exp of attention logits."
    }
  },
  "reference": "import math\nimport torch\n\n@torch.no_grad()\ndef run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):\n    # Check constants\n    assert num_qo_heads == 16\n    assert head_dim_ckv == 512\n    ..."
}
```

### Definition JSON Example (MoE)

```json
{
  "name": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
  "description": "FP8 block scale MoE operation. Routing and two grouped-GEMM included.",
  "op_type": "moe",
  "tags": [
    "status:verified",
    "model:deepseek-v3",
    "model:deepseek-r1",
    "quantization:float8_e4m3fn"
  ],
  "axes": {
    "seq_len": { "type": "var", "description": "Sequence length (number of tokens)" },
    "num_experts": {
      "type": "const",
      "value": 256,
      "description": "Total number of experts."
    },
    "num_local_experts": {
      "type": "const",
      "value": 32,
      "description": "Number of local experts with EP size 8."
    },
    "hidden_size": {
      "type": "const",
      "value": 7168,
      "description": "Hidden dimension size."
    },
    "intermediate_size": {
      "type": "const",
      "value": 2048,
      "description": "MoE intermediate layer size."
    }
  },
  "inputs": {
    "routing_logits": {
      "shape": ["seq_len", "num_experts"],
      "dtype": "float32",
      "description": "Tensor of routing logits for expert selection"
    },
    "hidden_states": {
      "shape": ["seq_len", "hidden_size"],
      "dtype": "float8_e4m3fn",
      "description": "Input hidden states tensor (FP8 quantized)"
    },
    "local_expert_offset": {
      "shape": null,
      "dtype": "int32",
      "description": "Offset of local experts in global expert space."
    },
    "routed_scaling_factor": {
      "shape": null,
      "dtype": "float32",
      "description": "Scaling factor for routing weights."
    }
  },
  "outputs": {
    "output": {
      "shape": ["seq_len", "hidden_size"],
      "dtype": "bfloat16",
      "description": "Final MoE output tensor"
    }
  },
  "reference": "import torch\n\n@torch.no_grad()\ndef run(routing_logits, hidden_states, ...):\n    # Check constants\n    assert H == 7168, 'hidden_size must be 7168'\n    ..."
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
   ls flashinfer_trace/definitions/*/
   ```

5. **Generate new definition JSONs**:
   - Create directory if needed: `mkdir -p flashinfer_trace/definitions/{op_type}/`
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

## Ground Truth Hierarchy

When extracting kernel definitions, use the following priority order to determine ground truth:

### Priority 1: SGLang Model Config / Vanilla Implementation
- **When**: Model has a vanilla (non-optimized) kernel implementation in SGLang
- **Location**: `third_party/sglang/python/sglang/srt/layers/`
- **Examples**:
  - Vanilla attention in `layers/attention/`
  - Vanilla MoE in `layers/moe/`
  - RMSNorm in `layers/layernorm.py`
- **Use for**: Understanding the mathematical specification and tensor shapes

### Priority 2: SGLang FlashInfer API Integration
- **When**: SGLang integrates FlashInfer APIs for optimized execution
- **Location**: Look for FlashInfer imports and calls in SGLang model files
- **Examples**:
  ```python
  # In SGLang model file
  import flashinfer
  flashinfer.batch_decode_with_paged_kv_cache(...)
  flashinfer.batch_prefill_with_paged_kv_cache(...)
  ```
- **Use for**: Determining the exact FlashInfer API signature and parameters

### Priority 3: FlashInfer API Directly
- **When**: Kernel is a pure FlashInfer API without SGLang wrapper
- **Location**: `third_party/flashinfer/python/flashinfer/`
- **Examples**:
  - `flashinfer.attention.batch_decode_with_paged_kv_cache`
  - `flashinfer.norm.rmsnorm`
  - `flashinfer.moe.moe_align_block_size`
- **Use for**: Ground truth correctness validation

## Reference Implementation Sources

The `reference` field in Definition JSON contains a `run()` function. Source this implementation from:

### Option 1: FlashInfer Test Implementation (Preferred)
- **Location**: `third_party/flashinfer/tests/`
- **Why**: Tests contain vanilla PyTorch implementations used to validate FlashInfer kernels
- **Examples**:
  ```
  tests/test_batch_decode.py      # GQA decode reference
  tests/test_batch_prefill.py     # GQA prefill reference
  tests/test_norm.py              # RMSNorm reference
  tests/test_mla.py               # MLA reference
  ```
- **Pattern**: Look for functions like `ref_attention()`, `ref_rmsnorm()`, etc.

### Option 2: SGLang Vanilla Implementation (Fallback)
- **Location**: `third_party/sglang/python/sglang/srt/layers/`
- **Why**: When FlashInfer tests don't cover the kernel, SGLang vanilla implementations provide the reference
- **Examples**:
  ```
  layers/moe/fused_moe.py         # MoE vanilla forward
  layers/attention/triton_ops/    # Attention vanilla implementations
  layers/layernorm.py             # Normalization vanilla
  ```

### Reference Implementation Guidelines

1. **Pure PyTorch**: Use only `torch` operations, no external kernels
2. **Step-by-step**: Break down computation into clear steps
3. **Match signatures**: Input/output names must match definition schema
4. **Include all outputs**: Return all outputs specified in definition (e.g., both `output` and `lse` for attention)

Example reference implementation pattern:
```python
import torch
import math

def run(q, k, v, ...):
    # Step 1: Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Step 2: Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Step 3: Compute output
    output = torch.matmul(attn_weights, v)

    # Step 4: Compute log-sum-exp (if needed)
    lse = torch.logsumexp(scores, dim=-1)

    return output, lse
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

- [clone-repos](../clone-repos/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
- [workflow](../workflow.md)
