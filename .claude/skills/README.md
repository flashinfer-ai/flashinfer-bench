# FlashInfer-Bench Skills

Automated workflows for adding new models, extracting kernel definitions, and validating implementations in FlashInfer-Bench.

## Quick Start

### Complete Workflow: Add Model with Kernel Definitions and Tests

```bash
# Step 1: Clone all required repositories
/clone-repos --target-dir ./repos

# Step 2: Extract kernel definitions from a model
/extract-kernel-definitions --model-name deepseek_v3

# Step 3: Add reference tests for the definitions
/add-reference-tests --op-type mla_paged

# Step 4: Add model to web interface
/add-new-model --model-name deepseek-v3 --hf-repo-id deepseek-ai/DeepSeek-V3
```

### Quick Model Addition (Web Interface Only)

```bash
/add-new-model --model-name kimi-k2 --hf-repo-id moonshot-ai/kimi-k2
```

## Available Skills

### Repository Setup

#### clone-repos
Clone SGLang, FlashInfer, and flashinfer-trace repositories.
[Documentation](./clone-repos.md)

```bash
/clone-repos --target-dir ./repos
```

### Kernel Definition Extraction

#### extract-kernel-definitions
Extract kernel schemas from SGLang model implementations with deduplication.
[Documentation](./extract-kernel-definitions.md)

```bash
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe
```

### Testing

#### add-reference-tests
Add tests to validate reference implementations against FlashInfer/SGLang ground truth.
[Documentation](./add-reference-tests.md)

```bash
/add-reference-tests --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1
/add-reference-tests --op-type moe
```

### Model Integration

#### add-new-model
Main workflow - automates the complete model addition process for the web interface.
[Documentation](./add-new-model.md)

```bash
/add-new-model --model-name kimi-k2 --hf-repo-id moonshot-ai/kimi-k2
```

#### extract-model-from-hf
Extract model configuration from HuggingFace.
[Documentation](./extract-model-from-hf.md)

```bash
/extract-model-from-hf --model-id moonshot-ai/kimi-k2
```

#### find-sglang-baseline
Find baseline implementation from SGLang codebase.
[Documentation](./find-sglang-baseline.md)

```bash
/find-sglang-baseline --model-name kimi
```

#### generate-model-definition
Generate TypeScript model definition file.
[Documentation](./generate-model-definition.md)

```bash
/generate-model-definition --config model_architecture.json --model-name kimi-k2
```

## Workflow Diagrams

### Full Kernel Extraction Pipeline

```
┌─────────────────────────────────────┐
│         1. Clone Repositories       │
│         /clone-repos                │
├─────────────────────────────────────┤
│  • SGLang (GitHub)                  │
│  • FlashInfer (GitHub)              │
│  • flashinfer-trace (HuggingFace)   │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│    2. Extract Kernel Definitions    │
│    /extract-kernel-definitions      │
├─────────────────────────────────────┤
│  • Analyze SGLang model files       │
│  • Extract kernel parameters        │
│  • Generate Definition JSONs        │
│  • Write reference implementations  │
│  • Deduplicate across models        │
│  Output:                            │
│    → definitions/{op_type}/*.json   │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│      3. Add Reference Tests         │
│      /add-reference-tests           │
├─────────────────────────────────────┤
│  • Find ground truth in FlashInfer  │
│  • Fallback to SGLang if needed     │
│  • Generate pytest test cases       │
│  • Parametrize for multiple sizes   │
│  Output:                            │
│    → tests/references/test_*.py     │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     4. Run Tests & Validate         │
│     pytest tests/references/        │
└─────────────────────────────────────┘
```

### Web Interface Model Addition

```
HuggingFace → extract → architecture.json
                              ↓
SGLang → find-baseline → implementation.json
                              ↓
                    generate-definition
                              ↓
                         models.ts
```

## Kernel Types Supported

| Op Type | Description | Example Definition |
|---------|-------------|-------------------|
| `rmsnorm` | RMS Layer Normalization | `rmsnorm_h4096` |
| `fused_add_rmsnorm` | RMSNorm with residual | `fused_add_rmsnorm_h4096` |
| `gemm` | General Matrix Multiply | `gemm_n_6144_k_4096` |
| `gqa_paged` | Grouped Query Attention (paged) | `gqa_paged_decode_h32_kv8_d128_ps1` |
| `gqa_ragged` | Grouped Query Attention (ragged) | `gqa_ragged_prefill_causal_h32_kv8_d128` |
| `mla_paged` | Multi-head Latent Attention | `mla_paged_decode_h16_ckv512_kpe64_ps1` |
| `moe` | Mixture of Experts | `moe_fp8_block_scale_ds_routing_topk8_...` |

## Examples

### Add DeepSeek V3 Model with MLA and MoE Kernels

```bash
# Setup repositories
/clone-repos --target-dir ./repos

# Extract all kernels (MLA, MoE, RMSNorm, etc.)
/extract-kernel-definitions --model-name deepseek_v3

# Add tests for MLA kernels
/add-reference-tests --op-type mla_paged

# Add tests for MoE kernels
/add-reference-tests --op-type moe

# Add model to web interface
/add-new-model --model-name deepseek-v3 --hf-repo-id deepseek-ai/DeepSeek-V3

# Run tests
cd ./repos/flashinfer-trace && pytest tests/references/ -v
```

### Add Multiple Models with Deduplication

```bash
/clone-repos --target-dir ./repos

# Extract from multiple models - definitions are deduplicated automatically
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe
/extract-kernel-definitions --model-name deepseek_v3

# Add tests for all new definitions
/add-reference-tests --definitions-dir ./repos/flashinfer-trace/definitions
```

### Test Specific Kernel Definition

```bash
/add-reference-tests \
  --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1 \
  --tolerance 1e-4
```

## Output Directories

```
repos/
├── sglang/                    # SGLang source code
├── flashinfer/                # FlashInfer source code
├── flashinfer-trace/          # HuggingFace dataset
│   ├── definitions/           # Kernel definition JSONs
│   │   ├── rmsnorm/
│   │   ├── gemm/
│   │   ├── gqa_paged/
│   │   ├── mla_paged/
│   │   └── moe/
│   ├── solutions/             # Optimized implementations
│   ├── traces/                # Execution traces
│   └── tests/
│       └── references/        # Reference implementation tests
│           ├── conftest.py
│           ├── test_rmsnorm.py
│           ├── test_gqa_paged.py
│           ├── test_mla_paged.py
│           └── test_moe.py
└── repos_config.json          # Configuration for other skills
```

## Requirements

- Git with LFS support
- Python 3.10+
- Python packages:
  - `huggingface_hub`
  - `pytest`
  - `torch` (with CUDA for GPU tests)
  - `flashinfer` (optional, for ground truth)
- Network access to GitHub and HuggingFace

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Complete guide to model addition
- [Definition Schema](../../docs/flashinfer_trace/definition.md)
- [Op Type Schemas](../../docs/op_type_schema/)
