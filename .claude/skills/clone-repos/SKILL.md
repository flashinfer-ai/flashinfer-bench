# Clone Repositories

Clone SGLang, FlashInfer repositories from GitHub and flashinfer-trace dataset from HuggingFace to the `third_party/` directory.

## Description

This skill sets up the required repositories for kernel extraction and testing workflows. It clones three repositories:
- **SGLang**: Inference engine with model implementations and kernel calls
- **FlashInfer**: GPU kernel library with optimized implementations (ground truth)
- **flashinfer-trace**: HuggingFace dataset containing Definition schemas and tests

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
- `flashinfer_trace_revision` (optional): flashinfer-trace dataset revision (default: "main")

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

### Step 4: Clone/Update flashinfer-trace Dataset

1. Clone from HuggingFace: `https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace`
2. Use `huggingface_hub` library or git-lfs
3. Key directories:
   - `definitions/` - Kernel Definition JSON schemas
   - `solutions/` - Optimized kernel implementations
   - `traces/` - Execution traces and benchmarks
   - `tests/references/` - Reference implementation tests

### Step 5: Verification

1. Verify all repositories cloned successfully
2. Check required directories exist
3. Report repository status

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

4. **Clone flashinfer-trace** (if not exists):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="flashinfer-ai/flashinfer-trace",
       repo_type="dataset",
       local_dir="third_party/flashinfer-trace",
       revision="main"
   )
   ```
   Or via git:
   ```bash
   git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace third_party/flashinfer-trace
   ```

5. **Update if force_update=true**:
   ```bash
   cd third_party/sglang && git fetch origin && git pull origin main
   cd third_party/flashinfer && git fetch origin && git pull origin main
   cd third_party/flashinfer-trace && git pull origin main
   ```

6. **Verify structure**:
   ```bash
   ls third_party/sglang/python/sglang/srt/models/
   ls third_party/flashinfer/python/flashinfer/
   ls third_party/flashinfer-trace/
   ```

## Output Directory Structure

```
third_party/
├── sglang/                           # SGLang repository
│   └── python/sglang/srt/
│       ├── models/                   # Model implementations
│       │   ├── llama.py
│       │   ├── deepseek_v3.py
│       │   ├── qwen2_moe.py
│       │   └── ...
│       └── layers/                   # Layer implementations
│           ├── attention/
│           ├── moe/
│           └── layernorm.py
├── flashinfer/                       # FlashInfer repository
│   ├── python/flashinfer/            # Python bindings (ground truth)
│   │   ├── attention/
│   │   ├── norm/
│   │   └── moe/
│   └── tests/                        # Reference tests
└── flashinfer-trace/                 # HuggingFace dataset
    ├── definitions/                  # Kernel definitions (our output)
    │   ├── rmsnorm/
    │   ├── gemm/
    │   ├── gqa_paged/
    │   ├── mla_paged/
    │   └── moe/
    └── tests/
        └── references/               # Reference tests (our output)
```

## Requirements

- Git with LFS support (for flashinfer-trace)
- Python packages:
  - `huggingface_hub` (for dataset cloning)
- Network access to:
  - GitHub (sglang, flashinfer)
  - HuggingFace Hub (flashinfer-trace)
- Sufficient disk space (~5GB total)

## Error Handling

### Network Errors
- **Error**: Cannot reach GitHub/HuggingFace
- **Handling**: Retry with exponential backoff, report specific endpoint failure

### Authentication Errors
- **Error**: Private repository access denied
- **Handling**: Check HF_TOKEN environment variable for HuggingFace

### Disk Space Errors
- **Error**: Insufficient disk space
- **Handling**: Report space requirements, suggest cleanup

### Git LFS Errors
- **Error**: LFS files not downloaded
- **Handling**: Run `git lfs pull`, verify LFS is installed

## Integration with Other Skills

This skill provides the foundation for:

1. **extract-kernel-definitions**: Uses SGLang model files to extract kernels
2. **add-reference-tests**: Uses FlashInfer for ground truth, flashinfer-trace for test location

Example workflow:

```bash
# Step 1: Clone all repositories
/clone-repos

# Step 2: Extract kernel definitions from a model
/extract-kernel-definitions --model-name deepseek_v3

# Step 3: Add reference tests
/add-reference-tests --op-type mla_paged
```

## Notes

- Uses shallow clones (--depth 1) by default to save disk space
- SGLang and FlashInfer are actively developed; pin versions for reproducibility
- flashinfer-trace may contain large trace files; consider partial download
- All output paths are relative to project root

## See Also

- [extract-kernel-definitions](./extract-kernel-definitions.md)
- [add-reference-tests](./add-reference-tests.md)
- [workflow](./workflow.md)
