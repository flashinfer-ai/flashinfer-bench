# Clone Repositories

Clone SGLang, FlashInfer repositories from GitHub and flashinfer-trace dataset from HuggingFace to a specified directory.

## Description

This skill sets up the required repositories for kernel extraction and testing workflows. It clones three repositories:
- **SGLang**: Inference engine with model implementations and kernel calls
- **FlashInfer**: GPU kernel library with optimized implementations (ground truth)
- **flashinfer-trace**: HuggingFace dataset containing Definition schemas and tests

## Parameters

- `target_dir` (required): Base directory to clone repositories into (e.g., "~/repos" or "./external")
- `sglang_branch` (optional): SGLang branch to checkout (default: "main")
- `flashinfer_branch` (optional): FlashInfer branch to checkout (default: "main")
- `flashinfer_trace_revision` (optional): flashinfer-trace dataset revision (default: "main")
- `force_update` (optional): Force pull latest changes if repos exist (default: false)

## Usage

```bash
# Basic usage - clone all repos to ./repos directory
/clone-repos --target-dir ./repos

# Clone to specific directory with force update
/clone-repos --target-dir ~/projects/flashinfer-work --force-update true

# Clone specific branches
/clone-repos \
  --target-dir ./repos \
  --sglang-branch v0.4.0 \
  --flashinfer-branch v0.2.0
```

## What This Skill Does

### Step 1: Validate Target Directory

1. Check if `target_dir` exists, create if not
2. Verify write permissions
3. Report any existing repositories in the directory

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

### Step 5: Verification and Summary

1. Verify all repositories cloned successfully
2. Check required directories exist
3. Generate repository status report
4. Output paths for subsequent skills

## Output Format

The skill outputs a JSON configuration file for use by other skills:

```json
{
  "target_dir": "/path/to/repos",
  "repositories": {
    "sglang": {
      "path": "/path/to/repos/sglang",
      "branch": "main",
      "commit": "abc123...",
      "status": "cloned"
    },
    "flashinfer": {
      "path": "/path/to/repos/flashinfer",
      "branch": "main",
      "commit": "def456...",
      "status": "cloned"
    },
    "flashinfer_trace": {
      "path": "/path/to/repos/flashinfer-trace",
      "revision": "main",
      "status": "cloned"
    }
  },
  "model_files_dir": "/path/to/repos/sglang/python/sglang/srt/models",
  "flashinfer_python_dir": "/path/to/repos/flashinfer/python/flashinfer",
  "definitions_dir": "/path/to/repos/flashinfer-trace/definitions",
  "tests_dir": "/path/to/repos/flashinfer-trace/tests/references"
}
```

## Repository Structure Reference

### SGLang Model Files

Model implementations follow this pattern:
```
python/sglang/srt/models/
├── llama.py              # Llama family models
├── deepseek_v3.py        # DeepSeek V3/R1 with MLA
├── qwen2_moe.py          # Qwen MoE models
├── mixtral.py            # Mixtral MoE
├── gemma.py              # Gemma models
└── ...
```

### FlashInfer Ground Truth

```
flashinfer/
├── python/flashinfer/
│   ├── attention/
│   │   ├── prefill.py
│   │   └── decode.py
│   ├── gemm/
│   ├── norm/
│   └── moe/
├── include/flashinfer/
│   ├── attention/
│   └── ...
└── tests/
```

### flashinfer-trace Dataset

```
flashinfer-trace/
├── definitions/
│   ├── rmsnorm/
│   │   ├── rmsnorm_h4096.json
│   │   └── ...
│   ├── gemm/
│   ├── gqa_paged/
│   ├── mla_paged/
│   └── moe/
├── solutions/
├── traces/
└── tests/
    └── references/
        ├── test_rmsnorm.py
        ├── test_gqa.py
        └── ...
```

## Requirements

- Git with LFS support (for flashinfer-trace)
- Python packages:
  - `huggingface_hub` (for dataset cloning)
- Network access to:
  - GitHub (sglang, flashinfer)
  - HuggingFace Hub (flashinfer-trace)
- Sufficient disk space (~5GB total)

## Implementation

When executed, this skill will:

1. **Create Target Directory**:
   ```bash
   mkdir -p {target_dir}
   ```

2. **Clone SGLang**:
   ```bash
   cd {target_dir}
   git clone --depth 1 --branch {sglang_branch} \
     https://github.com/sgl-project/sglang.git
   ```

3. **Clone FlashInfer**:
   ```bash
   cd {target_dir}
   git clone --depth 1 --branch {flashinfer_branch} \
     https://github.com/flashinfer-ai/flashinfer.git
   ```

4. **Clone flashinfer-trace**:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="flashinfer-ai/flashinfer-trace",
       repo_type="dataset",
       local_dir="{target_dir}/flashinfer-trace",
       revision="{flashinfer_trace_revision}"
   )
   ```

5. **Update if Existing** (when `force_update=true`):
   ```bash
   cd {repo_path}
   git fetch origin
   git checkout {branch}
   git pull origin {branch}
   ```

6. **Generate Configuration**:
   - Write `repos_config.json` with paths and status

## Error Handling

### Network Errors
- **Error**: Cannot reach GitHub/HuggingFace
- **Handling**: Retry with exponential backoff, report specific endpoint failure

### Authentication Errors
- **Error**: Private repository access denied
- **Handling**: Prompt for credentials, check HF_TOKEN environment variable

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
3. **find-sglang-baseline**: Uses cloned SGLang repository

Example workflow:

```bash
# Step 1: Clone all repositories
/clone-repos --target-dir ./repos

# Step 2: Extract kernel definitions from a model
/extract-kernel-definitions \
  --model-name deepseek-v3 \
  --repos-config ./repos/repos_config.json

# Step 3: Add reference tests
/add-reference-tests \
  --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1 \
  --repos-config ./repos/repos_config.json
```

## Notes

- Use shallow clones (--depth 1) by default to save disk space
- SGLang and FlashInfer are actively developed; pin versions for reproducibility
- flashinfer-trace may contain large trace files; consider partial download
- All paths in output config are absolute paths for reliability

## See Also

- [extract-kernel-definitions](./extract-kernel-definitions.md)
- [add-reference-tests](./add-reference-tests.md)
- [find-sglang-baseline](./find-sglang-baseline.md)
