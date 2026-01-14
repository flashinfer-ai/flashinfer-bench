# FlashInfer-Bench Complete Workflow

This document describes the complete workflow for adding new models, extracting kernel definitions, and validating implementations.

## Overview

The workflow consists of three main phases:

1. **Repository Setup**: Clone required repositories to `third_party/`
2. **Kernel Extraction**: Extract kernel definitions from SGLang model implementations
3. **Testing**: Add reference tests to validate implementations against ground truth

## Phase 1: Repository Setup

### Clone Required Repositories

```bash
/clone-repos
```

This clones to `third_party/`:
- **sglang**: `https://github.com/sgl-project/sglang.git`
  - Model implementations in `python/sglang/srt/models/`
  - Layer implementations in `python/sglang/srt/layers/`

- **flashinfer**: `https://github.com/flashinfer-ai/flashinfer.git`
  - Ground truth kernel implementations
  - Python bindings in `python/flashinfer/`

**Note**: The `flashinfer_trace/` directory is already included in this project (no cloning needed):
- Kernel definitions in `definitions/`
- Workloads in `workloads/`
- Reference tests in `tests/references/`

### Directory Structure

```
flashinfer-bench/
├── flashinfer_trace/          # Local (already in project)
│   ├── definitions/           # Kernel definition JSONs (output)
│   ├── workloads/             # Workload configurations
│   └── tests/references/      # Reference tests (output)
└── third_party/               # Cloned repositories
    ├── sglang/
    │   └── python/sglang/srt/
    │       ├── models/        # Model implementations (kernel extraction source)
    │       └── layers/        # Layer implementations (ground truth fallback)
    └── flashinfer/
        └── python/flashinfer/ # Ground truth implementations
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
flashinfer_trace/definitions/
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

## Phase 3: Add Reference Tests

### Add Tests for New Definitions

```bash
# Test all MLA kernels
/add-reference-tests --op-type mla_paged

# Test all MoE kernels
/add-reference-tests --op-type moe

# Test a specific definition
/add-reference-tests --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1

# Test all definitions
/add-reference-tests --all
```

### Ground Truth Sources

Tests compare reference implementations against:

1. **FlashInfer** (preferred): Optimized GPU kernels
   - Location: `third_party/flashinfer/python/flashinfer/`
   - For: GQA, MLA, RMSNorm, GEMM

2. **SGLang** (fallback): When FlashInfer doesn't have the kernel
   - Location: `third_party/sglang/python/sglang/srt/layers/`
   - For: MoE, custom kernels

### Ground Truth Mapping

| Op Type | Primary (FlashInfer) | Fallback (SGLang) |
|---------|---------------------|-------------------|
| `rmsnorm` | `norm/rmsnorm.py` | `layers/layernorm.py` |
| `gqa_paged` | `attention/decode.py` | `layers/attention/` |
| `mla_paged` | `attention/mla.py` | `layers/attention/mla_decode.py` |
| `moe` | `moe/` | `layers/moe/fused_moe.py` |
| `gemm` | `gemm/` | `torch.nn.functional.linear` |

### Test Output

```
flashinfer_trace/tests/references/
├── conftest.py              # Shared fixtures and utilities
├── test_rmsnorm.py          # RMSNorm tests
├── test_gqa_paged.py        # GQA paged tests
├── test_mla_paged.py        # MLA paged tests
├── test_moe.py              # MoE tests
└── test_gemm.py             # GEMM tests
```

### Running Tests

Run from the project root:

```bash
# Run all reference tests
pytest flashinfer_trace/tests/references/ -v

# Run specific test file
pytest flashinfer_trace/tests/references/test_mla_paged.py -v

# Run with GPU
pytest flashinfer_trace/tests/references/ -v --device cuda
```

## Complete Example: Adding DeepSeek V3

```bash
# Step 1: Clone repositories
/clone-repos

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

# Step 6: Run tests to validate (from project root)
pytest flashinfer_trace/tests/references/ -v
```

## Complete Example: Adding Multiple Models

```bash
# Setup
/clone-repos

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

# Validate (from project root)
pytest flashinfer_trace/tests/references/ -v --tb=short
```

## Troubleshooting

### Issue: Kernel not found in SGLang

```bash
# List available models
ls third_party/sglang/python/sglang/srt/models/

# Search for specific kernel
grep -r "batch_decode" third_party/sglang/python/sglang/srt/
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

| Phase | Skill | Input | Output |
|-------|-------|-------|--------|
| 1. Setup | `/clone-repos` | None | `third_party/` with SGLang and FlashInfer |
| 2. Extract | `/extract-kernel-definitions` | SGLang model files | `flashinfer_trace/definitions/{op_type}/*.json` |
| 3. Test | `/add-reference-tests` | Definition JSONs | `flashinfer_trace/tests/references/test_*.py` |

## Workflow Diagram

```
┌─────────────────────────────────────┐
│         1. Clone Repositories       │
│         /clone-repos                │
├─────────────────────────────────────┤
│  third_party/                       │
│  ├── sglang/      (GitHub)          │
│  └── flashinfer/  (GitHub)          │
│                                     │
│  flashinfer_trace/ (already local)  │
│  ├── definitions/                   │
│  ├── workloads/                     │
│  └── tests/references/              │
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
│    → flashinfer_trace/definitions/  │
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
│    → flashinfer_trace/tests/        │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     4. Run Tests & Validate         │
│  pytest flashinfer_trace/tests/     │
└─────────────────────────────────────┘
```

## Next Steps After Workflow

1. **Review definitions**: Check generated JSON files in `flashinfer_trace/definitions/`
2. **Run tests**: Ensure all tests pass
3. **Commit changes**: Commit new definitions and tests to the repository
4. **Create optimized solutions**: Use kernel_generator for Triton/CUDA implementations
5. **Run benchmarks**: `flashinfer-bench run --local ./data`

## See Also

- [clone-repos](./clone-repos/SKILL.md)
- [extract-kernel-definitions](./extract-kernel-definitions/SKILL.md)
- [add-reference-tests](./add-reference-tests/SKILL.md)
- [CLAUDE.md](../../CLAUDE.md)
