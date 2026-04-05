---
name: collect-workloads
description: Auto-collect workloads from SGLang inference runs using FlashInfer logging API. Dumps tensors, sanitizes them according to kernel definitions, and submits PR to flashinfer-trace workload repo.
---

# Collect Workloads

Automatically collect real-world workloads by running SGLang inference with FlashInfer Level 10 logging, then sanitize and submit to the flashinfer-ai/flashinfer-trace HuggingFace dataset repository.

## Design Principle

**No code changes to SGLang or FlashInfer are required.** Collection works entirely through FlashInfer's built-in logging API:

1. **Run SGLang** with `--attention-backend flashinfer` and FlashInfer Level 10 logging env vars → FlashInfer dumps per-call tensors automatically
2. **Sanitize dumps** with `scripts/sanitize_dumps.py` → converts raw FlashInfer dump directories into flashinfer-trace JSONL + safetensors

Then optionally submit a PR to the flashinfer-trace repo.

## Scripts

### Scripts Overview

| Script | Purpose | When to use |
|--------|---------|-------------|
| `collect_workloads.py` | **Primary entry point.** `sglang` mode: full pipeline (SGLang server + inference + sanitize). | Default for any workload collection |
| `sanitize_dumps.py` | Converts FlashInfer Level 10 per-call dump directories → flashinfer-trace JSONL + safetensors | Called automatically by `collect_workloads.py`; also run manually to re-sanitize existing dumps |

Always provide `--flashinfer-trace-dir` to specify the flashinfer-trace repo location.

### `scripts/collect_workloads.py` ← **primary collection script**

The main entry point. Always use `sglang` mode.

```bash
# SGLang mode: real inference → real structural tensors
CUDA_VISIBLE_DEVICES=0,0 python scripts/collect_workloads.py sglang \
  --model-path ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/<hash> \
  --definitions gqa_paged_prefill_causal_h20_kv4_d128_ps64 \
  --flashinfer-trace-dir tmp/flashinfer-trace \
  --replace \
  --skip-install  # skip if packages already installed
```

**Auto-detection from definition tags:**
- `tp:N` tag → auto-sets `--tp N` (use `CUDA_VISIBLE_DEVICES=0,0` to simulate TP=2 on 1 GPU)
- `page_size` const axis → auto-sets `--page-size N`

### `scripts/sanitize_dumps.py` ← **converts FlashInfer dumps to workload format**

Reads FlashInfer Level 10 per-call dump directories and converts to flashinfer-trace JSONL format. Called automatically by `collect_workloads.py sglang`, but can also be run manually to re-sanitize an existing dump dir.

```bash
python scripts/sanitize_dumps.py \
  --dump-dir ./workload_dumps_20260326_123456 \
  --definitions gqa_paged_prefill_causal_h20_kv4_d128_ps64 \
  --flashinfer-trace-dir tmp/flashinfer-trace \
  --replace
```

**Key flag**: `--skip-const-axis-check` — skip const-axis shape verification when collecting TP=1 dumps for a TP=2 definition (structural tensors like indptrs/indices are identical across TP; only head-count axes differ).

**Output:**
- `{flashinfer_trace_dir}/workloads/{op_type}/{def_name}.jsonl`
- `{flashinfer_trace_dir}/blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors`

## Usage

```bash
# Collect workloads for specific definitions
/collect-workloads --definition-names mla_paged_decode_h16_ckv512_kpe64_ps1 rmsnorm_h7168

# Collect for all definitions of an op_type
/collect-workloads --op-type mla_paged --model-name deepseek-v3

# Collect for all definitions (comprehensive collection)
/collect-workloads --all --model-name llama-3.1-8b

# Collect without submitting PR (local testing)
/collect-workloads --op-type gqa_paged --submit-pr false

# Custom dataset and sample size
/collect-workloads --op-type rmsnorm --dataset /path/to/custom_sharegpt.jsonl --num-samples 500
```

## Parameters

- `definition_names` (optional): List of specific definition names to collect workloads for
- `op_type` (optional): Collect workloads for all definitions of a specific op_type
- `all` (optional): Collect workloads for ALL definitions in definitions directory (default: false)
- `model_name` (required): Model to run inference on (e.g., "deepseek-v3", "llama-3.1-8b")
- `dataset` (optional): Path to ShareGPT-format JSONL dataset (default: download from Hugging Face)
- `num_samples` (optional): Number of inference samples to process (default: 100)
- `submit_pr` (optional): Whether to submit PR to flashinfer-trace repo (default: true)

## Prerequisites

Run `/clone-repos` first to set up the `tmp/` directory with SGLang and FlashInfer.

## What This Skill Does

### Phase 0: Install Latest Packages

**Always** install the latest SGLang and FlashInfer from source before collecting workloads.

```bash
git -C tmp/flashinfer pull
git -C tmp/sglang pull
conda run -n flashinfer_bench pip install -e tmp/flashinfer --no-build-isolation
conda run -n flashinfer_bench pip install -e "tmp/sglang/python[all]"

# Verify
conda run -n flashinfer_bench python -c "import sglang, flashinfer; print(f'SGLang: {sglang.__version__}, FlashInfer: {flashinfer.__version__}')"
```

### Phase 1: Resolve Target Definitions

- `--definitions`: load specific definitions by name (searched across all op_type dirs)
- `--op-type`: load all definitions from `{flashinfer-trace-dir}/definitions/{op_type}/`
- `--all`: scan all definitions in `{flashinfer-trace-dir}/definitions/`

### Phase 2: FlashInfer Logging Configuration

Each definition JSON has `fi_api:<dotted.api.name>` tags identifying which FlashInfer API to capture. `collect_workloads.py` parses these to build a precise `FLASHINFER_DUMP_INCLUDE` filter:

```python
# Class/Wrapper APIs → matched as ClassName.run (and ClassName.plan if def has int32/int64 inputs)
# Plain function APIs → matched as function name
# e.g. fi_api:flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper → "BatchPrefillWithPagedKVCacheWrapper.run"
#      + "BatchPrefillWithPagedKVCacheWrapper.plan" (since it has int32 qo_indptr, kv_indptr, kv_indices)
# e.g. fi_api:flashinfer.norm.rmsnorm → "rmsnorm"
```

**`.plan` inclusion rule**: `ClassName.plan` is added to `FLASHINFER_DUMP_INCLUDE` only if any definition using that API has `int32` or `int64` inputs. This ensures structural tensors (indptrs, indices) from `plan()` are captured without logging unnecessary data.

Environment variables set automatically:

```bash
FLASHINFER_LOGLEVEL=10               # enable Flight Recorder (Metadata + Tensors) mode
FLASHINFER_DUMP_DIR=./workload_dumps_<timestamp>
FLASHINFER_DUMP_SAFETENSORS=1        # save tensors as .safetensors (not .pt)
FLASHINFER_DUMP_MAX_SIZE_GB=50
FLASHINFER_DUMP_MAX_COUNT=10000
FLASHINFER_DUMP_INCLUDE=<fi_api patterns>   # only log matching API calls
FLASHINFER_DUMP_EXCLUDE=*.__init__           # skip constructors (plan() only excluded if not needed)
```

`FLASHINFER_DUMP_INCLUDE` is **critical** — without it, every FlashInfer call gets logged (gigabytes of irrelevant data).

**Reference**: [FlashInfer Logging Documentation](https://docs.flashinfer.ai/logging.html)

### Phase 3: SGLang Inference (sglang mode — DEFAULT)

**Always use this mode.** `collect_workloads.py sglang` handles everything automatically — no changes to SGLang needed:

1. Launch SGLang with `--attention-backend flashinfer --disable-cuda-graph --disable-piecewise-cuda-graph`
2. Wait for server ready (polls `/health`, timeout 30min)
3. Run ShareGPT inference via `/v1/chat/completions`
4. Shut down via SIGTERM

**Both CUDA graph flags are required — and they serve different purposes:**

- `--disable-cuda-graph`: disables regular CUDA graph capture. Without this, `BatchDecodeWithPagedKVCacheWrapper.run()` is called inside a CUDA stream capture context, which causes `cudaErrorStreamCaptureUnsupported` when the FlashInfer logging API tries to copy tensors to CPU. The dump fails silently, meaning only `plan()` tensors (structural int32/int64) are captured — `q`, `k_cache`, `v_cache` are never saved. This makes const-axis validation (e.g. `head_dim`) impossible: since `q` is None, `_verify_constant_axis` skips the check and silently accepts wrong values.

- `--disable-piecewise-cuda-graph`: disables piecewise CUDA graph capture. Even with `--disable-cuda-graph`, SGLang still runs piecewise capture by default (a separate mechanism). Piecewise capture also runs `run()` inside a CUDA stream, producing the same `cudaErrorStreamCaptureUnsupported` failure for `run()` dumps.

**Why this matters for `head_dim` validation specifically:** `head_dim` is a const axis inferred from `q.shape[2]`. Since `q` is only an argument to `run()` (not `plan()`), if `run()` is not captured, `head_dim` can never be verified regardless of what the definition says. Workloads collected without both flags will silently pass validation even if `head_dim` is wrong.

**Consequence of disabling CUDA graph:** batch size diversity comes from actual inference batching (SGLang's continuous batching scheduler), not from CUDA graph capture sequences. Use `--num-batches` and `--batch-sizes` to ensure sufficient coverage.

### Phase 4: Tensor Dump Sanitization

`scripts/sanitize_dumps.py` processes the FlashInfer dump directory:

1. **Match by function name** — reads `function_name` from `metadata.jsonl`, matches to definitions via `fi_api` tags
2. **Skip plan() dumps** — consumed as supplements to paired run() dumps
3. **Pair plan() with run()** — for wrapper class APIs (any `.plan` function), find the most recent preceding `.plan` dump from the same PID (by directory sort order) to get structural tensors
4. **Plan kwarg → definition input mapping**:
   - `paged_kv_indptr` → `kv_indptr`
   - `paged_kv_indices` → `kv_indices`
   - `paged_kv_last_page_len` → `kv_last_page_len`
   - `qo_indptr` → `qo_indptr`
5. **Tensor storage policy** (based on definition input dtype):
   - `int32`/`int64` (structural) → saved to safetensors blob (captured from `plan()`)
   - `float32`/`bfloat16`/`float16` (activations like `q`, `k_cache`, `v_cache`) → `{"type": "random"}` (actual values don't affect benchmarking; shapes are reconstructed from const/var axes at benchmark time)
   - null shape (scalars like `sm_scale`) → `{"type": "scalar", "value": <float>}` (captured from `run()`)

   **Critical**: float activation tensors are stored as `{"type": "random"}`, but they are still captured from `run()` to validate const axes like `head_dim` (via `q.shape[2]`). If `run()` is not captured (e.g. due to CUDA graph capture blocking dumps), const axes cannot be verified and may be silently wrong. Always use `--disable-cuda-graph --disable-piecewise-cuda-graph` to ensure `run()` is logged.
6. **kv_indices trimming**: SGLang KV pool is over-allocated; trim to `kv_indptr[-1]` valid entries
7. **Deduplication**: at most 2 entries per unique axes combination per definition

### Phase 5: Baseline Evaluation (correctness gate before PR submission)

Before creating PR 2, run the baseline solution against the collected workloads to verify
correctness. `collect_workloads.py` does this automatically via `--eval-baseline` (on by default):

```bash
# Automatically run as part of collect_workloads.py (--eval-baseline is the default)
# Runs: flashinfer-bench run --local {trace_dir} --definitions {def_name} --solutions baseline
# Writes: {trace_dir}/traces/{def_name}_baseline.jsonl
# Exits non-zero if any workload status != PASSED
```

All entries in the resulting `traces/{def_name}_baseline.jsonl` must have `evaluation.status == "PASSED"`.
If any fail, do **not** submit PR 2 — investigate and fix the reference implementation first.

### Phase 6: Submit Pull Request (1 HuggingFace PR per definition)

This skill produces **PR 2** in the two-PR workflow. PR 1 (definition JSON + reference tests +
`docs/model_coverage.mdx`) must already be open on GitHub before submitting PR 2.

For each definition, submit **one HuggingFace PR** containing **six items**:

| # | Item | Source |
|---|------|--------|
| 1 | Baseline solution JSON | FlashInfer API wrapper at `solutions/baseline/{op_type}/{def_name}/flashinfer_wrapper_*.json` — calls `flashinfer.BatchDecodeWithPagedKVCacheWrapper` or `flashinfer.BatchPrefillWithPagedKVCacheWrapper`, NOT the `reference_impl` from def JSON |
| 2 | Workload JSONL | `{trace_dir}/workloads/{op_type}/{def_name}.jsonl` |
| 3 | Safetensors blobs | `{trace_dir}/blob/workloads/{op_type}/{def_name}/` |
| 4 | Kernel definition JSON | Copied from flashinfer-bench (same as PR 1) |
| 5 | Reference test | Copied from flashinfer-bench (same as PR 1) |
| 6 | **Baseline eval trace JSONL** | `{trace_dir}/traces/{def_name}_baseline.jsonl` → copied to `traces/{op_type}/{def_name}.jsonl` (all PASSED) |

```bash
cd tmp/worktrees/trace-{definition_name}  # or tmp/flashinfer-trace on its own branch

# 1. Baseline solution — FlashInfer API wrapper JSON (NOT reference_impl from def JSON)
# Copy from the main flashinfer-trace repo (solutions/baseline/{op_type}/{def_name}/)
# The solution must call flashinfer.BatchDecodeWithPagedKVCacheWrapper (decode defs) or
# flashinfer.BatchPrefillWithPagedKVCacheWrapper (prefill defs / non-power-of-2 group sizes)
mkdir -p solutions/baseline/{op_type}/{def_name}
# Write a flashinfer_wrapper_<hash>.json file following the format in:
#   solutions/baseline/{op_type}/existing_def/flashinfer_wrapper_*.json
git add solutions/baseline/{op_type}/{def_name}/

# 2+3. Workload JSONL and blobs
cp {trace_dir}/workloads/{op_type}/{def_name}.jsonl workloads/{op_type}/{def_name}.jsonl
cp -r {trace_dir}/blob/workloads/{op_type}/{def_name}/ blob/workloads/{op_type}/{def_name}/
git add workloads/{op_type}/{def_name}.jsonl blob/workloads/{op_type}/{def_name}/

# 4. Kernel definition JSON
mkdir -p definitions/{op_type}
cp {REPO_ROOT}/flashinfer_trace/definitions/{op_type}/{def_name}.json definitions/{op_type}/{def_name}.json
git add definitions/{op_type}/{def_name}.json

# 5. Reference test
mkdir -p tests/references
cp {REPO_ROOT}/flashinfer_trace/tests/references/test_{def_name}.py tests/references/test_{def_name}.py
git add tests/references/test_{def_name}.py

# 6. Baseline eval trace (must all be PASSED — generated by Phase 5)
mkdir -p traces/{op_type}
cp {trace_dir}/traces/{def_name}_baseline.jsonl traces/{op_type}/{def_name}.jsonl
git add traces/{op_type}/{def_name}.jsonl

git commit -m "Add {def_name}: baseline solution + workloads + blobs + def + tests + eval trace

All {N} workload entries PASSED baseline evaluation.
Model: {hf_repo_id}
GitHub PR: flashinfer-ai/flashinfer-bench#{pr1_number}
"
git push origin {branch}

# Create the HuggingFace PR
python -c "
from huggingface_hub import HfApi
result = HfApi().create_discussion(
    repo_id='flashinfer-ai/flashinfer-trace',
    repo_type='dataset',
    title='Add {def_name}: workloads + solution + eval trace',
    description='''GitHub PR: flashinfer-ai/flashinfer-bench#{pr1_number}

## SGLang Collection Log

\`\`\`
{PASTE FULL STDOUT OF collect_workloads.py sglang HERE}
\`\`\`
''',
    pull_request=True,
)
print(result.url)
"
```

**SGLang log is required in PR2 description.** After creating the HuggingFace PR:
- Capture the full stdout of `collect_workloads.py sglang` (Phase 3–5)
- Paste it under `## SGLang Collection Log` in the PR2 discussion body
- The log must show: model loaded, workloads collected, kernel dump counts, baseline eval PASSED
- Workloads must be real SGLang-collected (diverse batch_size ≤ 64 with realistic kv_length distributions)
- **Anti-pattern**: `batch_size=4096` with tiny KV caches is synthetic, not from real inference

**Rule: one definition = one HuggingFace PR.** Do not batch multiple definitions.
Always wait for PR 1 (GitHub) to be open before submitting PR 2.

### Parallelizing across definitions with git worktrees

When submitting PRs for multiple definitions, use git worktrees so all submissions happen
in parallel — one worktree per definition in each repo, one agent per definition.

```bash
DATE=$(date +%Y%m%d)

# Create trace worktrees up front for all definitions
for DEF in {definition_name_1} {definition_name_2} ...; do
  git -C tmp/flashinfer-trace worktree add \
    ../worktrees/trace-${DEF} \
    -b workloads-${DATE}-${DEF}
done

# Spawn one agent per definition — all run simultaneously
# Each agent: copies solution + workload files into its worktree, commits, pushes, creates HF PR
# Clean up after all agents report their PR URLs:
for DEF in {definition_names}; do
  git -C tmp/flashinfer-trace worktree remove ../worktrees/trace-${DEF}
  git worktree remove tmp/worktrees/bench-${DEF}
done
```

See `onboard-model` SKILL.md Phase 4 for the full agent prompt template and per-step details.

## Output Format

### Workload JSONL

```
{flashinfer_trace_dir}/workloads/{op_type}/{def_name}.jsonl
```

Each line:
```json
{
  "definition": "gqa_paged_prefill_causal_h20_kv4_d128_ps64",
  "solution": null,
  "workload": {
    "uuid": "a1b2c3d4-...",
    "axes": {"len_indptr": 5, "total_q": 1024, "num_kv_indices": 512, "num_pages": 576},
    "inputs": {
      "q": {"type": "random"},
      "k_cache": {"type": "random"},
      "v_cache": {"type": "random"},
      "qo_indptr": {
        "type": "safetensors",
        "path": "./blob/workloads/gqa_paged/gqa_paged_prefill_causal_h20_kv4_d128_ps64/gqa_paged_prefill_causal_h20_kv4_d128_ps64_<uuid>.safetensors",
        "tensor_key": "qo_indptr"
      },
      "kv_indptr": {"type": "safetensors", "path": "...", "tensor_key": "kv_indptr"},
      "kv_indices": {"type": "safetensors", "path": "...", "tensor_key": "kv_indices"},
      "sm_scale": {"type": "scalar", "value": 0.08838834764831843}
    }
  },
  "evaluation": null
}
```

### Safetensors Blobs

```
{flashinfer_trace_dir}/blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors
```

One file per workload entry containing all structural tensors.

## Advanced Usage

### Custom FLASHINFER_DUMP_INCLUDE

The skill auto-builds `FLASHINFER_DUMP_INCLUDE` from `fi_api` tags. For ad-hoc runs:

```bash
export FLASHINFER_DUMP_INCLUDE="*Wrapper.run,*Wrapper.plan"  # all attention wrappers + plan
export FLASHINFER_DUMP_INCLUDE="*decode*"                    # decode kernels only
export FLASHINFER_DUMP_EXCLUDE="*.__init__"                  # always exclude constructors
```

### Cross-TP Collection

Collect TP=1 dumps for a TP=2 definition (structural tensors are identical across TP):

```bash
python scripts/sanitize_dumps.py \
  --dump-dir ./workload_dumps_tp1 \
  --definitions gqa_paged_prefill_causal_h20_kv4_d128_ps64 \
  --flashinfer-trace-dir tmp/flashinfer-trace \
  --skip-const-axis-check \
  --replace
```

## Error Handling

### No Tensor Dumps Generated

- Verify `FLASHINFER_LOGLEVEL=10` is set before any FlashInfer import (subprocess env)
- Check `FLASHINFER_DUMP_INCLUDE` matches the actual API function names
- Ensure SGLang is using FlashInfer backend (`--attention-backend flashinfer`)

### `run()` Not Captured — Silent `head_dim` Mismatch

**Symptom**: `cudaErrorStreamCaptureUnsupported` errors appear in the SGLang log for `BatchDecodeWithPagedKVCacheWrapper.run` or `BatchPrefillWithPagedKVCacheWrapper.run`. Workloads are produced but float tensors (`q`, `k_cache`, `v_cache`) are all `{"type": "random"}` stubs with no shape validation. Const axes like `head_dim` silently pass even if wrong.

**Cause**: `run()` was called inside a CUDA stream capture context (regular or piecewise CUDA graph), which blocks FlashInfer's tensor copy-to-CPU dump code.

**Fix**: Always pass both `--disable-cuda-graph` and `--disable-piecewise-cuda-graph` to the SGLang server. These flags must appear in the model config (`model_configs.json`) or be passed directly to `bench_sharegpt.py`. Never remove them to "get diverse batch sizes" — diversity should come from inference load, not CUDA graph capture.

### Constant Axis Mismatch

- Use `--skip-const-axis-check` when collecting across TP configurations
- May need a new definition variant if model config (num_heads, head_dim) doesn't match

### SGLang Server Fails to Start

- Check GPU memory (`nvidia-smi`)
- Reduce `--tp` or try a smaller model
- Check server log in the dump directory

### PR Submission Fails

- Verify HuggingFace auth: `huggingface-cli login`
- Check write permissions to flashinfer-ai/flashinfer-trace

## Checking SGLang Integration Before Collection

Before running `sglang` mode, verify that SGLang actually routes the target kernel through
FlashInfer. Use the `fi_api` tag from the definition JSON as the search term:

```bash
# e.g. for fi_api:flashinfer.gdn.gated_delta_rule_decode
grep -r "gated_delta_rule_decode" tmp/sglang/python/sglang/srt/ --include="*.py" | grep -v __pycache__
```

If the API is **not found** in SGLang:
1. SGLang needs to be updated to wire in this FlashInfer kernel.
2. The `onboard-model` skill (Phase 3c) handles drafting and submitting the SGLang PR.
3. Wait for the SGLang PR to merge before collecting workloads.

## Integration with Other Skills

```bash
/clone-repos
/extract-kernel-definitions --model-name deepseek_v3
/collect-workloads --op-type mla_paged --model-name deepseek-v3
/add-reference-tests --op-type mla_paged

# Or use the full end-to-end pipeline (handles SGLang integration check + PR automatically)
/onboard-model --model-name qwen3-235b-a22b
```

## References

- [FlashInfer Logging Documentation](https://docs.flashinfer.ai/logging.html)
- [flashinfer-ai/flashinfer-trace Dataset](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
- [SGLang Documentation](https://sgl-project.github.io/)

## See Also

- [clone-repos](../clone-repos/SKILL.md)
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
