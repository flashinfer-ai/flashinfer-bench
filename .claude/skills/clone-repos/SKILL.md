---
name: clone-repos
description: Clone SGLang, FlashInfer repositories from GitHub to third_party/. Use when setting up the project, preparing for kernel extraction, or when the user needs the source repositories.
---

# Clone Repositories

Clone SGLang, FlashInfer repositories from GitHub to the `third_party/` directory.

## Description

This skill sets up the required repositories for kernel extraction and testing workflows. It clones two repositories:
- **SGLang**: Inference engine with model implementations and kernel calls
- **FlashInfer**: GPU kernel library with optimized implementations (ground truth)

**Note**: The `flashinfer_trace` directory containing definitions, workloads, and tests is already included in the flashinfer-bench project at `./flashinfer_trace/`. No cloning required.

## Usage

```bash
# Clone all repos to ./third_party directory
/clone-repos

# Force update if repos already exist
/clone-repos --force-update

# Clone specific branches
/clone-repos --sglang-branch v0.4.0 --flashinfer-branch v0.2.0
```

## Parameters

- `force_update` (optional): Force pull latest changes if repos exist (default: false)
- `sglang_branch` (optional): SGLang branch to checkout (default: "main")
- `flashinfer_branch` (optional): FlashInfer branch to checkout (default: "main")

## What This Skill Does

### Step 1: Create third_party Directory

```bash
mkdir -p third_party
```

### Step 2: Clone/Update SGLang Repository

1. Clone from `https://github.com/sgl-project/sglang.git`
2. Checkout specified branch
3. Key directories for kernel extraction:
   - `python/sglang/srt/models/` - Model implementations
   - `python/sglang/srt/layers/` - Layer implementations (attention, MLP, norms)
   - `python/sglang/srt/layers/moe/` - MoE kernel implementations
   - `python/sglang/srt/layers/attention/` - Attention kernel implementations

### Step 3: Clone/Update FlashInfer Repository

1. Clone from `https://github.com/flashinfer-ai/flashinfer.git`
2. Checkout specified branch
3. Key directories for ground truth:
   - `python/flashinfer/` - Python bindings
   - `include/flashinfer/` - C++ headers with kernel implementations
   - `csrc/` - CUDA source files
   - `tests/` - Test implementations

### Step 4: Verification

1. Verify all repositories cloned successfully
2. Check required directories exist
3. Verify local `flashinfer_trace/` directory exists with definitions and tests
4. Report repository status

## Implementation Steps

When executing this skill:

1. **Check if third_party exists and has repos**:
   ```bash
   ls -la third_party/
   ```

2. **Clone SGLang** (if not exists):
   ```bash
   git clone --depth 1 https://github.com/sgl-project/sglang.git third_party/sglang
   ```

3. **Clone FlashInfer** (if not exists):
   ```bash
   git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git third_party/flashinfer
   ```

4. **Update if force_update=true**:
   ```bash
   cd third_party/sglang && git fetch origin && git pull origin main
   cd third_party/flashinfer && git fetch origin && git pull origin main
   ```

5. **Verify structure**:
   ```bash
   ls third_party/sglang/python/sglang/srt/models/
   ls third_party/flashinfer/python/flashinfer/
   ls flashinfer_trace/definitions/
   ls flashinfer_trace/tests/references/
   ```

## Output Directory Structure

```
flashinfer-bench/
├── flashinfer_trace/                 # Local (already in project)
│   ├── definitions/                  # Kernel definitions
│   │   ├── rmsnorm/
│   │   ├── gemm/
│   │   ├── gqa_paged/
│   │   ├── mla_paged/
│   │   └── moe/
│   ├── workloads/                    # Workload configurations
│   └── tests/
│       └── references/               # Reference tests
└── third_party/                      # Cloned repositories
    ├── sglang/                       # SGLang repository
    │   └── python/sglang/srt/
    │       ├── models/               # Model implementations
    │       │   ├── llama.py
    │       │   ├── deepseek_v3.py
    │       │   ├── qwen2_moe.py
    │       │   └── ...
    │       └── layers/               # Layer implementations
    │           ├── attention/
    │           ├── moe/
    │           └── layernorm.py
    └── flashinfer/                   # FlashInfer repository
        ├── python/flashinfer/        # Python bindings (ground truth)
        │   ├── attention/
        │   ├── norm/
        │   └── moe/
        └── tests/                    # Reference tests
```

## Requirements

- Git
- Network access to GitHub (for sglang, flashinfer)
- Sufficient disk space (~4GB total)

## Error Handling

### Network Errors
- **Error**: Cannot reach GitHub
- **Handling**: Retry with exponential backoff, report specific endpoint failure

### Disk Space Errors
- **Error**: Insufficient disk space
- **Handling**: Report space requirements, suggest cleanup

### Missing Local flashinfer_trace
- **Error**: `flashinfer_trace/` directory not found in project root
- **Handling**: Report error - this directory should be part of the flashinfer-bench repository

## Integration with Other Skills

This skill provides the foundation for:

1. **extract-kernel-definitions**: Uses SGLang model files to extract kernels, outputs to `./flashinfer_trace/definitions/`
2. **add-reference-tests**: Uses FlashInfer for ground truth, outputs tests to `./flashinfer_trace/tests/references/`

Example workflow:

```bash
# Step 1: Clone SGLang and FlashInfer repositories
/clone-repos

# Step 2: Extract kernel definitions from a model
/extract-kernel-definitions --model-name deepseek_v3

# Step 3: Add reference tests
/add-reference-tests --op-type mla_paged
```

## Notes

- Uses shallow clones (--depth 1) by default to save disk space
- SGLang and FlashInfer are actively developed; pin versions for reproducibility
- The `flashinfer_trace/` directory is part of this repository; no external cloning needed
- All output paths are relative to project root

## See Also

- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
- [workflow](../workflow.md)
