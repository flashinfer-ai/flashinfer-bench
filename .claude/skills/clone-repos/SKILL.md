---
name: clone-repos
description: Clone SGLang, FlashInfer repositories from GitHub to tmp/. Use when setting up the project, preparing for kernel extraction, or when the user needs the source repositories.
---

# Clone Repositories

Clone SGLang, FlashInfer repositories from GitHub to the `tmp/` directory.

## Description

This skill sets up the required repositories for kernel extraction and testing workflows. It:
1. Clones SGLang and FlashInfer repositories to `tmp/` directory (if not already present)
2. Updates repositories by pulling latest changes from remote (if repos already exist)
3. Installs both packages from source in the current environment

**Repositories:**
- **SGLang**: Inference engine with model implementations and kernel calls
- **FlashInfer**: GPU kernel library with optimized implementations (ground truth)

**Note**: The `flashinfer_trace/` directory is part of this repository and requires no cloning.

## Usage

```bash
# Clone all repos to ./tmp directory, update if exists, and install from source
/clone-repos

# Clone specific branches
/clone-repos --sglang-branch v0.4.0 --flashinfer-branch v0.2.0
```

## Parameters

- `sglang_branch` (optional): SGLang branch to checkout (default: "main")
- `flashinfer_branch` (optional): FlashInfer branch to checkout (default: "main")

## What This Skill Does

### Step 1: Create tmp Directory

```bash
mkdir -p tmp
```

### Step 2: Clone/Update SGLang Repository

1. **If repository doesn't exist**: Clone from `https://github.com/sgl-project/sglang.git`
2. **If repository exists**: Pull latest changes from remote origin
3. Checkout specified branch (default: main)
4. Install from source: `pip install -e tmp/sglang`
5. Key directories for kernel extraction:
   - `python/sglang/srt/models/` - Model implementations
   - `python/sglang/srt/layers/` - Layer implementations (attention, MLP, norms)
   - `python/sglang/srt/layers/moe/` - MoE kernel implementations
   - `python/sglang/srt/layers/attention/` - Attention kernel implementations

### Step 3: Clone/Update FlashInfer Repository

1. **If repository doesn't exist**: Clone from `https://github.com/flashinfer-ai/flashinfer.git`
2. **If repository exists**: Pull latest changes from remote origin
3. Checkout specified branch (default: main)
4. Install from source: `pip install -e tmp/flashinfer/python`
5. Key directories for ground truth:
   - `python/flashinfer/` - Python bindings
   - `include/flashinfer/` - C++ headers with kernel implementations
   - `csrc/` - CUDA source files
   - `tests/` - Test implementations with reference functions

### Step 4: Verification

1. Verify all repositories cloned/updated successfully
2. Check required directories exist
3. Verify packages installed correctly
4. Verify local `flashinfer_trace/` directory exists with definitions and tests
5. Report repository status

## Implementation Steps

When executing this skill:

1. **Create tmp directory if needed**:
   ```bash
   mkdir -p tmp
   ```

2. **Handle SGLang repository**:
   ```bash
   # Check if repo exists
   if [ -d "tmp/sglang/.git" ]; then
       echo "SGLang exists, pulling latest changes..."
       cd tmp/sglang && git fetch origin && git pull origin main && cd ../..
   else
       echo "Cloning SGLang..."
       git clone https://github.com/sgl-project/sglang.git tmp/sglang
   fi
   ```

3. **Handle FlashInfer repository**:
   ```bash
   # Check if repo exists
   if [ -d "tmp/flashinfer/.git" ]; then
       echo "FlashInfer exists, pulling latest changes..."
       cd tmp/flashinfer && git fetch origin && git pull origin main && cd ../..
   else
       echo "Cloning FlashInfer..."
       git clone https://github.com/flashinfer-ai/flashinfer.git tmp/flashinfer
   fi
   ```

4. **Install packages from source**:
   ```bash
   # Install SGLang
   pip install -e tmp/sglang

   # Install FlashInfer
   pip install -e tmp/flashinfer/python
   ```

5. **Verify structure**:
   ```bash
   ls tmp/sglang/python/sglang/srt/models/
   ls tmp/flashinfer/python/flashinfer/
   ls tmp/flashinfer/tests/
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
└── tmp/                              # Cloned repositories (auto-updated)
    ├── sglang/                       # SGLang repository (installed in current env)
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
    └── flashinfer/                   # FlashInfer repository (installed in current env)
        ├── python/flashinfer/        # Python bindings (ground truth)
        │   ├── attention/
        │   ├── norm/
        │   └── moe/
        └── tests/                    # Reference tests with vanilla implementations
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

### Installation Errors
- **Error**: pip install fails for SGLang or FlashInfer
- **Handling**: Check Python version compatibility, report missing dependencies, suggest manual installation steps

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

- Always pulls latest changes if repositories already exist to keep dependencies up-to-date
- Installs both packages in editable mode (`pip install -e`) for development convenience
- SGLang and FlashInfer are actively developed; use branch parameters to pin specific versions
- Repositories are stored in `tmp/` which can be added to `.gitignore`

## Maintaining This Document

Update this file when changing repository URLs, directory structure, or adding new repositories.

## See Also

- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
- [workflow](../workflow.md)
