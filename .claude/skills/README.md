# FlashInfer-Bench Skills

Automated workflows for adding new models, extracting kernel definitions, and validating implementations in FlashInfer-Bench.

## Quick Start

### Complete Workflow: Add Model with Kernel Definitions and Tests

```bash
# Step 1: Clone SGLang and FlashInfer repositories to third_party/
/clone-repos

# Step 2: Extract kernel definitions from a model (e.g., deepseek_v3)
/extract-kernel-definitions --model-name deepseek_v3

# Step 3: Add reference tests for the definitions
/add-reference-tests --op-type mla_paged
```

## Available Skills

| Skill | Description |
|-------|-------------|
| [clone-repos](./clone-repos/SKILL.md) | Clone SGLang, FlashInfer from GitHub to `third_party/` |
| [extract-kernel-definitions](./extract-kernel-definitions/SKILL.md) | Extract kernel schemas from SGLang with deduplication |
| [add-reference-tests](./add-reference-tests/SKILL.md) | Add tests to validate reference implementations |
| [workflow](./workflow.md) | Complete workflow documentation |

## Directory Structure

```
flashinfer-bench/
├── flashinfer_trace/              # Local (included in project)
│   ├── definitions/               # Kernel definition JSONs
│   │   ├── rmsnorm/
│   │   ├── gemm/
│   │   ├── gqa_paged/
│   │   ├── mla_paged/
│   │   └── moe/
│   ├── workloads/                 # Workload configurations
│   └── tests/
│       └── references/            # Reference implementation tests
└── third_party/                   # Cloned repositories
    ├── sglang/                    # SGLang source code (GitHub)
    │   └── python/sglang/srt/
    │       ├── models/            # Model implementations (kernel calls)
    │       └── layers/            # Layer implementations
    └── flashinfer/                # FlashInfer source code (GitHub)
        └── python/flashinfer/     # Ground truth implementations
```

## Kernel Types Supported

| Op Type | Description | Models |
|---------|-------------|--------|
| `rmsnorm` | RMS Layer Normalization | All models |
| `gemm` | General Matrix Multiply | All models |
| `gqa_paged` | Grouped Query Attention (paged) | Llama, Qwen |
| `gqa_ragged` | Grouped Query Attention (ragged) | Llama, Qwen |
| `mla_paged` | Multi-head Latent Attention | DeepSeek V3/R1 |
| `moe` | Mixture of Experts | DeepSeek, Qwen MoE |

## Example: Adding DeepSeek V3

```bash
# 1. Setup repositories
/clone-repos

# 2. Extract all kernels (MLA, MoE, RMSNorm, etc.)
/extract-kernel-definitions --model-name deepseek_v3

# 3. Add tests for MLA and MoE kernels
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe

# 4. Run tests to validate (from project root)
pytest flashinfer_trace/tests/references/ -v
```

## Example: Adding Multiple Models with Deduplication

```bash
/clone-repos

# Extract from multiple models - definitions are deduplicated automatically
/extract-kernel-definitions --model-name llama
/extract-kernel-definitions --model-name qwen2_moe
/extract-kernel-definitions --model-name deepseek_v3

# Add tests for all kernel types
/add-reference-tests --op-type gqa_paged
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe
/add-reference-tests --op-type rmsnorm
```

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Complete guide to model addition
- [Definition Schema](../../docs/flashinfer_trace/definition.md)
- [Op Type Schemas](../../docs/op_type_schema/)
