# FlashInfer-Bench Complete Workflow

This document describes the complete workflow for adding new models, extracting kernel definitions, and validating implementations.

## Overview

The workflow consists of three main phases:

1. **Repository Setup**: Clone required repositories from GitHub and HuggingFace
2. **Kernel Extraction**: Extract kernel definitions from SGLang model implementations
3. **Testing**: Add reference tests to validate implementations

## Phase 1: Repository Setup

### Clone All Required Repositories

```bash
/clone-repos --target-dir ./repos
```

This clones:
- **sglang**: `https://github.com/sgl-project/sglang.git`
  - Model implementations in `python/sglang/srt/models/`
  - Layer implementations in `python/sglang/srt/layers/`

- **flashinfer**: `https://github.com/flashinfer-ai/flashinfer.git`
  - Ground truth kernel implementations
  - Python bindings in `python/flashinfer/`

- **flashinfer-trace**: `https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace`
  - Kernel definitions in `definitions/`
  - Reference tests in `tests/references/`

### Output

Creates `repos_config.json`:
```json
{
  "target_dir": "./repos",
  "repositories": {
    "sglang": {"path": "./repos/sglang", "status": "cloned"},
    "flashinfer": {"path": "./repos/flashinfer", "status": "cloned"},
    "flashinfer_trace": {"path": "./repos/flashinfer-trace", "status": "cloned"}
  }
}
```

## Phase 2: Kernel Definition Extraction

### Extract Kernels from Model Implementations

For each model you want to support:

```bash
# DeepSeek V3 (MLA + MoE)
/extract-kernel-definitions --model-name deepseek_v3

# Llama family (GQA)
/extract-kernel-definitions --model-name llama

# Qwen MoE (GQA + MoE)
/extract-kernel-definitions --model-name qwen2_moe
```

### What Gets Extracted

For each model, the skill extracts:

| Model | Kernel Types | Example Definitions |
|-------|--------------|---------------------|
| DeepSeek V3 | MLA, MoE, RMSNorm | `mla_paged_decode_h16_ckv512_kpe64_ps1`, `moe_fp8_block_scale_ds_routing_topk8_...` |
| Llama | GQA, GEMM, RMSNorm | `gqa_paged_decode_h32_kv8_d128_ps1`, `gemm_n_6144_k_4096` |
| Qwen MoE | GQA, MoE, RMSNorm | `gqa_paged_decode_h32_kv4_d128_ps1`, `moe_...` |

### Deduplication

Kernels used by multiple models are automatically deduplicated:

```
Shared kernels:
├── rmsnorm_h4096: [llama, qwen]
├── gqa_paged_decode_h32_kv8_d128_ps1: [llama]
└── mla_paged_decode_h16_ckv512_kpe64_ps1: [deepseek_v3]
```

### Output Structure

```
repos/flashinfer-trace/definitions/
├── rmsnorm/
│   ├── rmsnorm_h4096.json
│   ├── rmsnorm_h7168.json
│   └── fused_add_rmsnorm_h4096.json
├── gemm/
│   ├── gemm_n_6144_k_4096.json
│   └── gemm_n_4096_k_4096.json
├── gqa_paged/
│   ├── gqa_paged_decode_h32_kv8_d128_ps1.json
│   └── gqa_paged_prefill_causal_h32_kv8_d128_ps1.json
├── mla_paged/
│   ├── mla_paged_decode_h16_ckv512_kpe64_ps1.json
│   └── mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.json
└── moe/
    └── moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
```

### Definition JSON Format

Each definition includes:
- **name**: Unique identifier with parameters encoded
- **op_type**: Kernel category (rmsnorm, gemm, gqa_paged, mla_paged, moe)
- **axes**: Constant and variable dimensions
- **inputs/outputs**: Tensor specifications with shapes and dtypes
- **reference**: Vanilla Python/PyTorch implementation

Example:
```json
{
  "name": "mla_paged_decode_h16_ckv512_kpe64_ps1",
  "op_type": "mla_paged",
  "tags": ["stage:decode", "model:deepseek-v3"],
  "axes": {
    "batch_size": {"type": "var"},
    "num_qo_heads": {"type": "const", "value": 16},
    "head_dim_ckv": {"type": "const", "value": 512},
    "head_dim_kpe": {"type": "const", "value": 64},
    "page_size": {"type": "const", "value": 1}
  },
  "inputs": {...},
  "outputs": {...},
  "reference": "import torch\n\ndef run(...):\n    ..."
}
```

## Phase 3: Add Reference Tests

### Add Tests for New Definitions

```bash
# Test all MLA kernels
/add-reference-tests --op-type mla_paged

# Test all MoE kernels
/add-reference-tests --op-type moe

# Test a specific definition
/add-reference-tests --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1

# Test all definitions in a directory
/add-reference-tests --definitions-dir ./repos/flashinfer-trace/definitions
```

### Ground Truth Sources

Tests compare reference implementations against:

1. **FlashInfer** (preferred): Optimized GPU kernels
   - Location: `repos/flashinfer/python/flashinfer/`
   - For: GQA, MLA, RMSNorm, GEMM

2. **SGLang** (fallback): When FlashInfer doesn't have the kernel
   - Location: `repos/sglang/python/sglang/srt/layers/`
   - For: MoE, custom kernels

### Test Output

```
repos/flashinfer-trace/tests/references/
├── conftest.py              # Shared fixtures and utilities
├── test_rmsnorm.py          # RMSNorm tests
├── test_gqa_paged.py        # GQA paged tests
├── test_mla_paged.py        # MLA paged tests
├── test_moe.py              # MoE tests
└── test_gemm.py             # GEMM tests
```

### Running Tests

```bash
cd repos/flashinfer-trace

# Run all reference tests
pytest tests/references/ -v

# Run specific test file
pytest tests/references/test_mla_paged.py -v

# Run with GPU
pytest tests/references/ -v --device cuda
```

## Optional: Add Model to Web Interface

If you also want to add the model to the FlashInfer-Bench web interface:

```bash
/add-new-model --model-name deepseek-v3 --hf-repo-id deepseek-ai/DeepSeek-V3
```

This updates `web/apps/web/data/models.ts` with the model's module hierarchy.

## Complete Example: Adding DeepSeek V3

```bash
# Step 1: Clone repositories
/clone-repos --target-dir ./repos

# Step 2: Extract kernel definitions
/extract-kernel-definitions --model-name deepseek_v3

# This extracts:
# - mla_paged_decode_h16_ckv512_kpe64_ps1
# - mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
# - moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
# - rmsnorm_h7168, rmsnorm_h1536, rmsnorm_h512
# - fused_add_rmsnorm_h7168

# Step 3: Add tests for MLA kernels
/add-reference-tests --op-type mla_paged

# Step 4: Add tests for MoE kernels
/add-reference-tests --op-type moe

# Step 5: Add tests for normalization
/add-reference-tests --op-type rmsnorm

# Step 6: Run tests to validate
cd repos/flashinfer-trace
pytest tests/references/ -v

# Step 7 (optional): Add to web interface
/add-new-model --model-name deepseek-v3 --hf-repo-id deepseek-ai/DeepSeek-V3
```

## Complete Example: Adding Multiple Models

```bash
# Setup
/clone-repos --target-dir ./repos

# Extract from all target models (deduplication handled automatically)
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe
/extract-kernel-definitions --model-name deepseek_v3
/extract-kernel-definitions --model-name mixtral

# Add tests for all kernel types
/add-reference-tests --op-type gqa_paged
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe
/add-reference-tests --op-type rmsnorm
/add-reference-tests --op-type gemm

# Validate
cd repos/flashinfer-trace
pytest tests/references/ -v --tb=short
```

## Troubleshooting

### Issue: Kernel not found in SGLang

```bash
# List available models
ls repos/sglang/python/sglang/srt/models/

# Search for specific kernel
grep -r "batch_decode" repos/sglang/python/sglang/srt/
```

### Issue: Ground truth not available

When FlashInfer doesn't have the kernel:
- Tests will use SGLang as fallback
- If neither has it, tests are marked as `skip`

### Issue: Numerical tolerance failures

```bash
# Run with looser tolerance
/add-reference-tests --definition-name xxx --tolerance 1e-2
```

### Issue: Definition already exists

Deduplication is automatic. If a conflict is detected:
- New definition saved with version suffix
- Conflict report generated for manual review

## Summary

| Phase | Skill | Output |
|-------|-------|--------|
| 1. Setup | `/clone-repos` | `repos/` directory with all repos |
| 2. Extract | `/extract-kernel-definitions` | `definitions/{op_type}/*.json` |
| 3. Test | `/add-reference-tests` | `tests/references/test_*.py` |
| 4. Web (optional) | `/add-new-model` | Updated `models.ts` |

## Next Steps After Workflow

1. **Review definitions**: Check generated JSON files for accuracy
2. **Run tests**: Ensure all tests pass
3. **Submit to dataset**: Push to flashinfer-trace HuggingFace repo
4. **Create optimized solutions**: Use kernel_generator for Triton/CUDA implementations
5. **Run benchmarks**: `flashinfer-bench run --local ./data`
