---
name: clone-repos
description: Clone SGLang, FlashInfer, sgl-cookbook repositories from GitHub to tmp/. Use when setting up the project, preparing for kernel extraction, or when the user needs the source repositories.
---

# Clone Repositories

Clone SGLang, FlashInfer, and sgl-cookbook repositories from GitHub to the `tmp/` directory.

## Description

This skill sets up the required repositories for kernel extraction and testing workflows. It:
1. Clones SGLang, FlashInfer, and sgl-cookbook repositories to `tmp/` directory (if not already present) with all submodules
2. Updates repositories by pulling latest changes from remote and updating submodules (if repos already exist)
3. Checks out the `main` branch by default (or specified branch)
4. Installs SGLang and FlashInfer packages from source in the current environment

**Repositories:**
- **SGLang**: Inference engine with model implementations and kernel calls
- **FlashInfer**: GPU kernel library with optimized implementations (ground truth)
- **sgl-cookbook**: Best serving configurations for each architecture and model (TP, EP flags)

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
- `cookbook_branch` (optional): sgl-cookbook branch to checkout (default: "main")

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
       (cd tmp/sglang && git fetch origin && git checkout "${sglang_branch:-main}" && git reset --hard "origin/${sglang_branch:-main}" && git submodule update --init --recursive)
   else
       echo "Cloning SGLang with submodules..."
       git clone --recurse-submodules https://github.com/sgl-project/sglang.git tmp/sglang
       (cd tmp/sglang && git checkout "${sglang_branch:-main}")
   fi
   ```

   **Note**: Using `(cd ...)` subshell syntax ensures directory changes are isolated and don't affect subsequent commands.

3. **Handle FlashInfer repository**:
   ```bash
   # Check if repo exists
   if [ -d "tmp/flashinfer/.git" ]; then
       echo "FlashInfer exists, pulling latest changes..."
       (cd tmp/flashinfer && git fetch origin && git checkout "${flashinfer_branch:-main}" && git reset --hard "origin/${flashinfer_branch:-main}" && git submodule update --init --recursive)
   else
       echo "Cloning FlashInfer with submodules..."
       git clone --recurse-submodules https://github.com/flashinfer-ai/flashinfer.git tmp/flashinfer
       (cd tmp/flashinfer && git checkout "${flashinfer_branch:-main}")
   fi
   ```

   **Note**: Using `(cd ...)` subshell syntax ensures directory changes are isolated and don't affect subsequent commands.

4. **Handle sgl-cookbook repository**:
   ```bash
   # Check if repo exists
   if [ -d "tmp/sgl-cookbook/.git" ]; then
       echo "sgl-cookbook exists, pulling latest changes..."
       (cd tmp/sgl-cookbook && git fetch origin && git checkout "${cookbook_branch:-main}" && git reset --hard "origin/${cookbook_branch:-main}")
   else
       echo "Cloning sgl-cookbook..."
       git clone https://github.com/sgl-project/sgl-cookbook.git tmp/sgl-cookbook
       (cd tmp/sgl-cookbook && git checkout "${cookbook_branch:-main}")
   fi
   ```

   **Note**: sgl-cookbook doesn't require submodules or installation. It contains serving configuration files only.

5. **Install packages from source**:
   ```bash
   # Upgrade pip once
   pip install --upgrade pip

   # Install FlashInfer (pyproject.toml in repo root)
   (cd tmp/flashinfer && python -m pip install --no-build-isolation -e . -v)

   # Install SGLang (pyproject.toml in python/ subdirectory)
   (cd tmp/sglang && pip install -e "python")
   ```

   **Note**: Subshell syntax `(cd ... && command)` keeps working directory unchanged.

6. **Verify installations**:
   ```bash
   # Test imports
   python -c "import sglang; print(f'SGLang: {sglang.__version__}')"
   python -c "import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')"

   # Verify directory structure
   ls tmp/sglang/python/sglang/srt/models/
   ls tmp/flashinfer/flashinfer/
   ls tmp/flashinfer/tests/
   ls tmp/sgl-cookbook/
   ls flashinfer_trace/definitions/
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
│   │   ├── gdn/
│   │   ├── moe/
│   │   └── ...
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
    ├── flashinfer/                   # FlashInfer repository (installed in current env)
    │   ├── flashinfer/               # Python package in root (not python/ subdir!)
    │   │   ├── attention.py
    │   │   ├── norm.py
    │   │   ├── moe.py
    │   │   └── ...
    │   ├── tests/                    # Reference tests with vanilla implementations
    │   ├── csrc/                     # CUDA source files
    │   └── include/                  # C++ headers with kernel implementations
    └── sgl-cookbook/                 # Serving configuration repository (NOT installed)
        ├── docs/                     # Model deployment documentation (markdown)
        │   └── autoregressive/
        │       ├── DeepSeek/
        │       │   ├── DeepSeek-V3.md
        │       │   └── DeepSeek-R1.md
        │       ├── Qwen/
        │       │   ├── Qwen3-Next.md
        │       │   └── Qwen3.md
        │       └── ...
        ├── data/
        │   ├── models/               # Model configurations (YAML)
        │   │   └── generated/v0.5.6/
        │   │       ├── qwen3next.yaml
        │   │       ├── deepseek.yaml
        │   │       └── ...
        │   └── optimal-configs/      # Optimal serving configurations
        └── README.md
```

## Requirements

- Git (with submodule support)
- Network access to GitHub (for sglang, flashinfer, sgl-cookbook, and their submodules)
- Sufficient disk space (~6GB total including submodules and serving configs)
- Python development environment for building from source
- CUDA toolkit (for FlashInfer CUDA kernels)

## Common Issues

- **Network errors**: Check GitHub connectivity; repositories with submodules require stable connection
- **Submodule failures**: Retry `git submodule update --init --recursive`
- **Disk space**: Requires ~6GB total for all repositories with submodules
- **Installation failures**: Verify Python ≥3.8, CUDA toolkit installed, and submodules initialized
- **sgl-cookbook not found**: Ensure you have network access to github.com/sgl-project/sgl-cookbook

## Integration with Other Skills

This skill provides the foundation for:

1. **extract-kernel-definitions**: Uses SGLang model files to extract kernels, sgl-cookbook to find serving configurations (TP/EP flags), outputs to `./flashinfer_trace/definitions/`
2. **add-reference-tests**: Uses FlashInfer for ground truth, outputs tests to `./flashinfer_trace/tests/references/`

Example workflow:

```bash
# Step 1: Clone SGLang, FlashInfer, and sgl-cookbook repositories
/clone-repos

# Step 2: Extract kernel definitions from a model (uses sgl-cookbook for TP/EP configs)
/extract-kernel-definitions --model-name deepseek_v3

# Step 3: Add reference tests
/add-reference-tests --op-type mla_paged
```

## Notes

- Updates existing repos or performs full clones with submodules
- Editable installs (`pip install -e`) for development
- FlashInfer package location: `tmp/flashinfer/flashinfer/` (not in `python/` subdirectory)
- sgl-cookbook is NOT installed (configuration files only, no Python package)

## Maintaining This Document

Update this file when changing repository URLs, directory structure, or adding new repositories.

## See Also

- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
