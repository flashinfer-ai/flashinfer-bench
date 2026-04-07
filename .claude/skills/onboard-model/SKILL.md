---
name: onboard-model
description: End-to-end pipeline for discovering new LLMs with novel kernels and onboarding them into FlashInfer-Bench. Orchestrates repo updates, model discovery, kernel definition generation, workload collection, and PR submission.
---

# Onboard Model

Full end-to-end pipeline for discovering new LLMs with novel kernels and onboarding them into
FlashInfer-Bench. This skill orchestrates all sub-skills in the correct order and handles
branching logic depending on whether FlashInfer already supports the required kernels and whether
SGLang already integrates them.

## Overview of Phases

```
Phase 0: Update local repos (clone-repos)
Phase 1: Discover model + identify required kernels (track-models + HF)
Phase 2: Kernel definition generation
          ├─ kernel present in FlashInfer → generate def from FlashInfer ground truth
          └─ kernel absent from FlashInfer → generate def from HF config + SGLang,
                                              file GitHub issue in flashinfer-ai/flashinfer
Phase 3: Workload collection (only when FlashInfer has the kernel)
          ├─ kernel integrated into SGLang → collect-workloads (sglang mode)
          └─ kernel NOT integrated into SGLang → draft + submit SGLang PR,
                                                  then collect-workloads (sglang mode)
Phase 4: Submit PRs (per definition, not batched)
          ├─ PR 1 → flashinfer-bench (GitHub): definition JSON + reference tests + model_coverage.mdx
          └─ PR 2 → flashinfer-ai/flashinfer-trace (HuggingFace): baseline solution + workloads + traces + def JSON + reference tests
```

## Usage

```bash
# Discover new models and fully onboard any that are ready
/onboard-model --discover

# Onboard a specific known model end-to-end
/onboard-model --model-name qwen3-235b-a22b --hf-repo-id Qwen/Qwen3-235B-A22B

# Run only specific phases (e.g. skip workload collection for now)
/onboard-model --model-name kimi-k2 --phases 0,1,2

# Dry-run: discover and report what would be done without making changes
/onboard-model --discover --dry-run
```

## Parameters

- `--discover` (optional): Auto-discover new models from SGLang day-0 additions and sgl-cookbook.
  Mutually compatible with `--model-name`.
- `--model-name` (optional): Specific model slug to onboard (e.g. `qwen3-235b-a22b`).
- `--hf-repo-id` (optional): HuggingFace repo ID override (e.g. `Qwen/Qwen3-235B-A22B`).
  Inferred from model name if omitted.
- `--phases` (optional): Comma-separated list of phases to run (default: `0,1,2,3,4`).
- `--dry-run` (optional): Print what would be done without writing files or submitting PRs.
- `--skip-workload` (optional): Skip Phase 3 (workload collection). Useful when no GPU is
  available or kernel support is incomplete.
- `--submit-prs` (optional): Submit PRs at the end of Phase 4 (default: true).

---

## Phase 0: Update All Local Repos

**Goal**: Ensure all cloned repos under `tmp/` are current before any analysis.

### Implementation

Delegate to the `clone-repos` skill. This covers SGLang, FlashInfer, sgl-cookbook, and
flashinfer-trace.

```bash
# If repos already exist they will be pulled; otherwise cloned
/clone-repos
```

After the pull, verify the current commit SHAs:

```bash
git -C tmp/sglang rev-parse --short HEAD
git -C tmp/flashinfer rev-parse --short HEAD
git -C tmp/sgl-cookbook rev-parse --short HEAD
git -C tmp/flashinfer-trace rev-parse --short HEAD
```

Report the SHAs in the Phase 0 summary so the user can reproduce the run.

---

## Phase 1: Model Discovery and Kernel Inventory

**Goal**: Identify the target model(s), extract their required kernel set, and classify each
kernel as *known* (definition already exists in `flashinfer_trace/definitions/`) or *new*
(no definition file found).

### 1a: Discover candidate models (when `--discover` is set)

**Day-0 SGLang additions** (highest priority — these are production-ready models):

```bash
# List files added to SGLang models directory since last 30 days
git -C tmp/sglang log --since="30 days ago" --name-status --diff-filter=A \
    -- "python/sglang/srt/models/*.py" | grep "^A" | awk '{print $2}'
```

Models with a new `.py` file in `python/sglang/srt/models/` within the last 30 days are
day-0 candidates. Parse the model class name to derive a human-readable model slug.

**sgl-cookbook new entries**:

```bash
git -C tmp/sgl-cookbook log --since="30 days ago" --name-status --diff-filter=A \
    -- "data/models/generated/v0.5.6/*.yaml" | grep "^A" | awk '{print $2}'
```

New YAML files in sgl-cookbook indicate models with recommended serving configs.

**Filter already-tracked models**:

Read `docs/model_coverage.mdx` and extract model names from the `## Summary` table. Skip any
candidate already listed.

### 1b: Fetch model config from HuggingFace

For each candidate model (or the specified `--model-name`), fetch `config.json`:

```python
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id=hf_repo_id, filename="config.json")
with open(config_path) as f:
    config = json.load(f)
```

Key fields to extract (see `track-models` SKILL.md for the full table).

### 1c: Determine required kernel definitions

Use the same rules as `track-models` Phase 3a to compute the expected set of definition names
from the model config and sgl-cookbook TP/EP values. See `track-models` SKILL.md for the
complete per-op-type formulas.

### 1d: Classify each kernel

For each expected definition name:

```bash
find flashinfer_trace/definitions/ -name "{definition_name}.json"
```

| Result | Classification |
|--------|---------------|
| File found | **existing** — skip definition generation |
| Not found | **new** — proceed to Phase 2 |

### 1e: Check FlashInfer kernel availability for new definitions

For each *new* definition, determine whether FlashInfer already implements the underlying
kernel (even if no definition JSON exists yet).

**op_type → FlashInfer package path mapping**:

| op_type | Check path in `tmp/flashinfer/` |
|---------|--------------------------------|
| `rmsnorm` | `flashinfer/norm.py` — grep for `rmsnorm` |
| `gqa_paged` | `flashinfer/decode.py`, `flashinfer/prefill.py` |
| `gqa_ragged` | `flashinfer/prefill.py` |
| `mla_paged` | `flashinfer/mla.py` |
| `dsa_paged` | `flashinfer/sparse.py` |
| `gdn` | `flashinfer/gdn.py` or `flashinfer/gdn/` |
| `moe` | `flashinfer/fused_moe/` — check for specific variant |
| `gemm` | Always available via PyTorch |
| `sampling` | `flashinfer/sampling.py` |

Additionally check `tmp/flashinfer/tests/` for a corresponding test file — its presence is a
strong signal that the kernel is implemented and tested.

Classify each new definition as:

- **fi_supported**: FlashInfer has the kernel → Phase 2b (generate def from FlashInfer)
- **fi_missing**: FlashInfer does not have the kernel → Phase 2a (generate def + file issue)

### 1f: Phase 1 report

Print a table:

```
Model: Qwen3-235B-A22B
HF repo: Qwen/Qwen3-235B-A22B
Architecture: 94 layers, GQA + MoE

Kernel inventory:
  EXISTING (skip):
    ✅ rmsnorm_h7168
    ✅ moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  NEW — FlashInfer supported:
    🆕 gqa_paged_decode_h40_kv8_d128_ps1
    🆕 gqa_paged_decode_h40_kv8_d128_ps64
  NEW — FlashInfer MISSING (will file issue):
    ❓ <new_op_type>_<params>
```

---

## Phase 2: Kernel Definition Generation

**Goal**: Create definition JSON files for all *new* kernels.

### 2a: FlashInfer missing → generate definition + file issue

When FlashInfer does not yet implement the kernel:

**2a-i: Generate definition JSON from HF config + SGLang**

Follow the same steps as `extract-kernel-definitions` Phase 4, but using SGLang's vanilla
implementation as the reference `run()` instead of FlashInfer tests. See
`extract-kernel-definitions` SKILL.md for the full JSON structure and naming conventions.

Mark the definition with `"status:unverified"` tag (no FlashInfer ground truth yet).

**2a-ii: File a GitHub issue in `flashinfer-ai/flashinfer`**

After writing the definition JSON, file a GitHub issue requesting the kernel implementation.

```bash
gh issue create \
  --repo flashinfer-ai/flashinfer \
  --title "Kernel request: {op_type} for {model_name}" \
  --label "enhancement,kernel-request" \
  --body "$(cat <<'EOF'
## Kernel Request

**Model**: {model_display_name} ({hf_repo_id})
**Op type**: {op_type}
**Definition name**: {definition_name}

### Motivation

This kernel is required for serving **{model_display_name}** with FlashInfer.
A FlashInfer-Bench definition has been generated at:
`flashinfer_trace/definitions/{op_type}/{definition_name}.json`

### Kernel Parameters

{formatted parameter table from definition axes}

### Reference Implementation

A plain-PyTorch reference `run()` is available in the definition JSON above.
The SGLang implementation is at:
`python/sglang/srt/layers/{layer_path}`

### Requested Work

- [ ] CUDA/Triton kernel implementation matching the definition schema
- [ ] FlashInfer Python API (`flashinfer.{module}.{function}`)
- [ ] Unit test in `tests/test_{op_type}.py`

### Links

- FlashInfer-Bench definition: (link to PR in this repo once merged)
- SGLang model: `tmp/sglang/python/sglang/srt/models/{model_file}`
- HuggingFace model: https://huggingface.co/{hf_repo_id}
EOF
)"
```

Record the issue URL. Add it as a comment to the definition JSON `description` field:
```
"description": "... See flashinfer-ai/flashinfer#<issue_number> for kernel implementation request."
```

**Do not proceed to Phase 3** for fi_missing kernels — workload collection requires the
FlashInfer kernel to exist.

### 2b: FlashInfer present → generate definition from FlashInfer

Delegate to `extract-kernel-definitions` for each new definition. This skill handles:
- Looking up TP/EP configs from sgl-cookbook
- Writing definition JSON with FlashInfer test as reference `run()`
- Deduplication against existing files

```bash
/extract-kernel-definitions --model-name {sglang_model_name}
```

After generation, verify each expected definition now exists:

```bash
find flashinfer_trace/definitions/ -name "{definition_name}.json"
```

---

## Phase 3: Workload Collection

**Goal**: Collect real workload tensors for all *fi_supported* definitions that do not yet
have workloads in `tmp/flashinfer-trace/workloads/`.

Skip this phase entirely for *fi_missing* kernels.

### 3a: Check workload existence

```bash
ls tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl 2>/dev/null
```

If the JSONL already exists and is non-empty, skip collection for that definition.

### 3b: Check SGLang integration

Determine whether SGLang already routes the kernel through the FlashInfer backend.

**How to check:**

```bash
# Check if SGLang's FlashInfer attention backend calls the relevant FlashInfer API
grep -r "flashinfer.{relevant_module}" \
    tmp/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py \
    tmp/sglang/python/sglang/srt/layers/attention/ 2>/dev/null

# Check for the specific wrapper or function name
grep -r "{flashinfer_api_name}" \
    tmp/sglang/python/sglang/srt/ 2>/dev/null | grep -v __pycache__
```

Use the `fi_api` tag from the definition JSON to know which API name to search for.

The fi_api → SGLang integration mapping for common op_types:

| fi_api | SGLang integration file | Search term |
|--------|------------------------|-------------|
| `flashinfer.mla.BatchMLAPagedAttentionWrapper` | `layers/attention/flashinfer_backend.py` | `BatchMLAPagedAttentionWrapper` |
| `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` | `layers/attention/flashinfer_backend.py` | `BatchDecodeWithPagedKVCacheWrapper` |
| `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` | `layers/attention/flashinfer_backend.py` | `BatchPrefillWithPagedKVCacheWrapper` |
| `flashinfer.norm.rmsnorm` | `layers/layernorm.py` | `flashinfer.norm` |
| `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` | `layers/moe/fused_moe.py` | `trtllm_fp8_block_scale_moe` |
| `flashinfer.gdn.gated_delta_rule_decode` | `layers/attention/gdn_backend.py` | `gated_delta_rule_decode` |

Classify each definition's SGLang integration status:

- **sgl_integrated**: SGLang already calls this FlashInfer API
- **sgl_missing**: SGLang does not yet call this FlashInfer API

### 3c: SGLang missing → draft + submit SGLang PR

When the FlashInfer kernel exists but SGLang has not wired it in:

**3c-i: Implement the integration**

Locate the appropriate SGLang layer file and add the FlashInfer call. The changes are
typically small (import + conditional dispatch). Follow existing patterns in the same file.

Common integration patterns:

```python
# In layers/attention/flashinfer_backend.py or equivalent
try:
    from flashinfer.{module} import {KernelClass}
    FLASHINFER_{KERNEL}_AVAILABLE = True
except ImportError:
    FLASHINFER_{KERNEL}_AVAILABLE = False

def forward(...):
    if FLASHINFER_{KERNEL}_AVAILABLE and use_flashinfer:
        return {KernelClass}(...).run(...)
    else:
        return vanilla_forward(...)
```

**3c-ii: Submit PR to SGLang**

```bash
cd tmp/sglang

# Create branch
git checkout -b feat/flashinfer-{op_type}-integration-{model_slug}

# Stage the integration changes
git add python/sglang/srt/layers/...

git commit -m "feat: integrate FlashInfer {op_type} for {model_name}

Wire {fi_api} into SGLang's FlashInfer backend to enable
optimized {op_type} for {model_display_name}.

Needed for: flashinfer-ai/flashinfer-bench (workload collection)
FlashInfer API: {fi_api}
"

pre-commit run --all-files
git push origin HEAD

gh pr create \
  --repo sgl-project/sglang \
  --title "feat: integrate FlashInfer {op_type} for {model_name}" \
  --body "$(cat <<'EOF'
## Summary

- Integrates `{fi_api}` into SGLang's FlashInfer backend
- Enables optimized `{op_type}` kernel for **{model_display_name}**
- Required by flashinfer-ai/flashinfer-bench for workload collection

## Changes

- `python/sglang/srt/layers/{path}`: add FlashInfer dispatch for {op_type}

## Test plan

- [ ] SGLang unit test passes with `--attention-backend flashinfer`
- [ ] Inference output matches non-FlashInfer baseline
- [ ] Memory usage within expected bounds
EOF
)"
```

Record the SGLang PR URL. Wait for it to be merged before running workload collection.

**3c-iii: Wait for SGLang PR**

Pause Phase 3 and note that it must resume after the SGLang PR merges.

### 3d: Run workload collection

For each definition with `sgl_integrated` status (or after SGLang PR merges):

```bash
/collect-workloads \
  --definition-names {definition_name} \
  --model-name {model_name} \
  --submit-pr false   # PR is submitted in Phase 4
```

Verify output files were created:

```bash
ls tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl
ls tmp/flashinfer-trace/blob/workloads/{op_type}/{definition_name}/
```

---

## Phase 4: Submit PRs

**Goal**: Publish collected workloads and kernel definitions, one PR per definition,
with all definitions processed in parallel using git worktrees.

**Rule: one definition = two atomic PRs (opened in sequence per definition):**

| # | Target repo | Content | Trigger |
|---|-------------|---------|---------|
| 1 | `flashinfer-ai/flashinfer-bench` (GitHub) | definition JSON + reference tests + `docs/model_coverage.mdx` | after Phase 2 |
| 2 | `flashinfer-ai/flashinfer-trace` (HuggingFace) | baseline solution + workload JSONL + safetensors traces + def JSON + reference tests | after PR 1 is open |

Do **not** batch multiple definitions into a single PR. Each definition must be independently
reviewable and mergeable.

### 4-setup: Create worktrees before spawning agents

For each new definition, create two worktrees: one in the main repo (for PRs 1 and 3) and one
in the `tmp/flashinfer-trace` clone (for PR 2). All worktrees can be created up front, then
agents run in parallel.

```bash
DATE=$(date +%Y%m%d)

# For each {definition_name} in the new definitions list:

# Worktree 1: flashinfer-bench (this repo) — for definition JSON + coverage update
git worktree add \
  tmp/worktrees/bench-{definition_name} \
  -b feat/def-{definition_name}

# Worktree 2: flashinfer-trace (HuggingFace dataset clone) — for workload files
git -C tmp/flashinfer-trace worktree add \
  ../worktrees/trace-{definition_name} \
  -b workloads-${DATE}-{definition_name}
```

Worktree layout after setup:
```
flashinfer-bench/
└── tmp/
    ├── flashinfer-trace/          # main clone (do not commit here directly)
    └── worktrees/
        ├── bench-{def1}/          # isolated branch for def1 definition JSON + coverage
        ├── bench-{def2}/          # isolated branch for def2
        ├── trace-{def1}/          # isolated branch for def1 workloads (HF repo)
        └── trace-{def2}/          # isolated branch for def2 workloads (HF repo)
```

### 4-spawn: Run one agent per definition in parallel

Spawn all definition agents simultaneously. Each agent receives its worktree paths and
handles all three PRs for its definition independently.

**Agent prompt template (repeat for each definition):**

```
You are handling PR submission for kernel definition: {definition_name}

Your worktrees:
- flashinfer-bench worktree: tmp/worktrees/bench-{definition_name}/
  (branch: feat/def-{definition_name})
- flashinfer-trace worktree: tmp/worktrees/trace-{definition_name}/
  (branch: workloads-{date}-{definition_name})

Definition file already written at:
  flashinfer_trace/definitions/{op_type}/{definition_name}.json

Workload files already collected at:
  tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl
  tmp/flashinfer-trace/blob/workloads/{op_type}/{definition_name}/

Do the following in order:

1. PR 1 — GitHub flashinfer-bench (definition JSON + reference tests + coverage):
   - Copy flashinfer_trace/definitions/{op_type}/{definition_name}.json
     into tmp/worktrees/bench-{definition_name}/flashinfer_trace/definitions/{op_type}/
   - Add or update reference test in tests/ for this definition
   - Update docs/model_coverage.mdx to reflect definition status
   - Commit all three changes together and push
   - Open PR to flashinfer-ai/flashinfer-bench
   - Record the PR number as pr1_number

2. PR 2 — HuggingFace flashinfer-trace (baseline solution + workloads + traces + def + tests):
   - Copy baseline solution (Python reference) into tmp/worktrees/trace-{definition_name}/solutions/{op_type}/
   - Copy workload JSONL into tmp/worktrees/trace-{definition_name}/workloads/{op_type}/
   - Copy safetensors blobs into tmp/worktrees/trace-{definition_name}/blob/workloads/{op_type}/
   - Copy definition JSON (from flashinfer-bench) into tmp/worktrees/trace-{definition_name}/definitions/{op_type}/
   - Copy reference test (from flashinfer-bench) into tmp/worktrees/trace-{definition_name}/tests/{op_type}/
   - Commit all together and push
   - Open HuggingFace PR via huggingface_hub.HfApi().create_pull_request()
   - Record the PR number as pr2_number

Report the two PR URLs when done.
```

### 4a: PR 1 — GitHub flashinfer-bench (definition JSON + reference tests + coverage)

Inside `tmp/worktrees/bench-{definition_name}/`:

```bash
# 1. Copy definition JSON
cp flashinfer_trace/definitions/{op_type}/{definition_name}.json \
   tmp/worktrees/bench-{definition_name}/flashinfer_trace/definitions/{op_type}/

# 2. Add or update reference test
# Write tests/test_{op_type}_{definition_name}.py with pytest test calling
# the reference implementation from the definition JSON

# 3. Update model coverage doc
# Edit docs/model_coverage.mdx: mark {definition_name} row as "definition added"

cd tmp/worktrees/bench-{definition_name}
git add flashinfer_trace/definitions/{op_type}/{definition_name}.json \
        tests/test_{op_type}_{definition_name}.py \
        docs/model_coverage.mdx
git commit -m "feat: add {definition_name}

- Definition JSON for {op_type} ({model_display_name})
- Reference test validating the implementation
- Model coverage doc updated
{If fi_missing: FlashInfer issue: flashinfer-ai/flashinfer#{issue_number}}
"
# Run reference test and capture output for PR description
pytest tests/test_{op_type}_{definition_name}.py -v 2>&1 | tee /tmp/test_{definition_name}.txt

pre-commit run --all-files
git push origin feat/def-{definition_name}
gh pr create \
  --repo flashinfer-ai/flashinfer-bench \
  --title "feat: add {definition_name}" \
  --body "$(cat <<'EOF'
## Summary
- Adds kernel definition for `{definition_name}` ({op_type})
- Model: {model_display_name}
- Reference implementation sourced from: {source}
{If fi_missing: - ⚠️ FlashInfer kernel missing — tracking issue: flashinfer-ai/flashinfer#{issue_number}}

## Reference Test Results
\`\`\`
$(cat /tmp/test_{definition_name}.txt)
\`\`\`

## Files changed
- `flashinfer_trace/definitions/{op_type}/{definition_name}.json`
- `tests/test_{op_type}_{definition_name}.py`
- `docs/model_coverage.mdx`
EOF
)"
```

### 4b: PR 2 — HuggingFace flashinfer-trace (baseline solution + workloads + traces + def + tests)

Inside `tmp/worktrees/trace-{definition_name}/`:

```bash
# 1. Copy baseline solution (extract reference_impl field from definition JSON as standalone script)
cp flashinfer_trace/definitions/{op_type}/{definition_name}.json \
   tmp/worktrees/trace-{definition_name}/solutions/{op_type}/{definition_name}.py
# (extract the reference_impl field from the definition JSON as a standalone script)

# 2. Copy workload JSONL and safetensors blobs
cp -r tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl \
      tmp/worktrees/trace-{definition_name}/workloads/{op_type}/
cp -r tmp/flashinfer-trace/blob/workloads/{op_type}/{definition_name}/ \
      tmp/worktrees/trace-{definition_name}/blob/workloads/{op_type}/

# 3. Copy kernel definition JSON and reference test from flashinfer-bench (this repo)
cp flashinfer_trace/definitions/{op_type}/{definition_name}.json \
   tmp/worktrees/trace-{definition_name}/definitions/{op_type}/
cp tests/test_{op_type}_{definition_name}.py \
   tmp/worktrees/trace-{definition_name}/tests/{op_type}/

cd tmp/worktrees/trace-{definition_name}
git add solutions/{op_type}/{definition_name}.py \
        workloads/{op_type}/{definition_name}.jsonl \
        blob/workloads/{op_type}/{definition_name}/ \
        definitions/{op_type}/{definition_name}.json \
        tests/{op_type}/test_{op_type}_{definition_name}.py
git commit -m "Add {definition_name}: baseline solution + workloads + traces + def + tests

Model: {hf_repo_id}
SGLang: {sglang_commit_sha}
FlashInfer: {flashinfer_commit_sha}
Workload entries: {num_workload_entries}
GitHub PR: flashinfer-ai/flashinfer-bench#{pr1_number}
"
pre-commit run --all-files
git push origin workloads-{date}-{definition_name}
python -c "
from huggingface_hub import HfApi
HfApi().create_pull_request(
    repo_id='flashinfer-ai/flashinfer-trace',
    repo_type='dataset',
    title='Add {definition_name}: baseline solution + workloads + traces + def + tests',
    description='...',
    head='workloads-{date}-{definition_name}',
)
"
```

### 4-cleanup: Remove worktrees after PRs are open

```bash
# After all agents have reported their PR URLs:
for def in {definition_names}; do
  git worktree remove tmp/worktrees/bench-${def}
  git -C tmp/flashinfer-trace worktree remove ../worktrees/trace-${def}
done
```

---

## PR Review Checklist

Run this checklist after both PRs are open for a definition. **Both PRs must pass all items
before the definition is considered complete.** If any item fails, fix and re-push before
requesting merge.

### PR 1 — GitHub flashinfer-bench

1. **Definition JSON**: `flashinfer_trace/definitions/{op_type}/{name}.json` exists in the PR
2. **Reference test**: `flashinfer_trace/tests/references/test_{name}.py` exists in the PR
3. **Coverage**: `docs/model_coverage.mdx` updated — row for this definition shows ✅
4. **Test results**: PR description includes the full stdout of running the reference test
5. **PR2 link**: PR description includes a link to the HuggingFace PR 2 (workload addition)
6. **Tags**: definition JSON has `status:verified` (or `status:unverified` if FlashInfer kernel
   is missing), `fi_api:*`, and `ep:*`/`tp:*` if applicable

### PR 2 — HuggingFace flashinfer-trace

1. **Workloads**: `workloads/{op_type}/{name}.jsonl` exists and is non-empty
2. **Blobs**: `blob/workloads/{op_type}/{name}/*.safetensors` files exist
3. **Baseline solution**: `solutions/baseline/{op_type}/{name}/flashinfer_wrapper_*.json`
   exists — this must be a FlashInfer API wrapper (calls `BatchDecodeWithPagedKVCacheWrapper`
   or `BatchPrefillWithPagedKVCacheWrapper`), **not** a copy of `reference_impl`
4. **Eval trace**: `traces/{op_type}/{name}.jsonl` exists and every entry has
   `evaluation.status == "PASSED"` — no failures allowed
5. **Definition JSON**: `definitions/{op_type}/{name}.json` copied from PR 1
6. **Reference test**: `tests/references/test_{name}.py` copied from PR 1
7. **SGLang log**: PR description contains a `## SGLang Collection Log` section with the
   full stdout from the `collect_workloads.py sglang` run (model loading, workload counts,
   dump dir info). Workloads must be SGLang-collected (not synthetic) — real workloads have
   diverse `(batch_size, kv_length)` pairs drawn from actual inference. A uniform sweep like
   `batch_size=4096` with 1-page contexts is a red flag for synthetic data.

---

## Agent TASK.md Template

When spawning an agent for a definition, write `.claude/TASK.md` in its bench worktree.
Every TASK.md for definition onboarding must include:

```markdown
## Objective
Submit 2 PRs for definition {name}:
- PR 1 (GitHub flashinfer-bench): Definition JSON + reference tests + docs/model_coverage.mdx (✅) + paste reference test stdout in PR description
- PR 2 (HuggingFace flashinfer-trace): Baseline solution + workloads + blobs + def JSON + ref test + baseline eval trace (all entries PASSED)

## PR 1 Contents
- `flashinfer_trace/definitions/{op_type}/{name}.json`
- `flashinfer_trace/tests/references/test_{name}.py`
- `docs/model_coverage.mdx` updated: ❌/🟡 → ✅ for this definition
- PR description must include the full stdout of running the reference test
- PR description must include a link to the HuggingFace PR 2 (workload addition)

## PR 2 Contents
- `solutions/baseline/{op_type}/{name}/flashinfer_wrapper_*.json` (FlashInfer API wrapper — NOT the reference_impl from def JSON; must call flashinfer.BatchDecodeWithPagedKVCacheWrapper or flashinfer.BatchPrefillWithPagedKVCacheWrapper)
- `workloads/{op_type}/{name}.jsonl`
- `blob/workloads/{op_type}/*.safetensors`
- `traces/{op_type}/{name}.jsonl` (all entries must have `evaluation.status == "PASSED"`)
- `definitions/{op_type}/{name}.json` (copied from PR 1)
- `tests/references/test_{name}.py` (copied from PR 1)
- PR description must include the SGLang inference stdout (capture stdout of `collect_workloads.py sglang` and paste in PR body under `## SGLang Collection Log`)

## Progress Reporting
Write .agent-progress.md after every major step:
  Status: in_progress | completed | blocked
  Done: <what's done>
  Current: <what you're doing now>
  Next: <next step>
  Blockers: <if any>
  - PR 1 (def JSON + ref tests + coverage → flashinfer-bench GitHub): <URL or pending>
  - PR 2 (baseline solution + workloads + traces → flashinfer-trace HF): <URL or pending>

## GPU Work
Use tools/gpu-lock before any SGLang workload collection:
  tools/gpu-lock --gpus <N> --exec-timeout 1800 -- python collect_workloads.py ...
Where N matches the TP value (1 GPU for TP=1, 4 GPUs for TP=4, etc.)
```

---

## Decision Tree Summary

```
For each required kernel definition:

  Already exists in flashinfer_trace/definitions/?
  ├── YES → skip (existing)
  └── NO  → Phase 2: generate definition JSON
              FlashInfer has this kernel?
              ├── YES → generate def from FlashInfer tests (status:verified)
              │          SGLang integrates this FlashInfer API?
              │          ├── YES → collect workloads (sglang mode)
              │          └── NO  → submit SGLang PR, wait for merge, then collect workloads
              └── NO  → generate def from HF config + SGLang (status:unverified)
                         file GitHub issue in flashinfer-ai/flashinfer
                         ⛔ skip workload collection (no FlashInfer kernel yet)
```

---

## State Tracking

This skill produces a run manifest file at `tmp/onboard_{model_slug}_{date}.json` that
records state for resumability:

```json
{
  "model_slug": "qwen3-235b-a22b",
  "hf_repo_id": "Qwen/Qwen3-235B-A22B",
  "date": "2026-03-29",
  "repo_shas": {
    "sglang": "abc1234",
    "flashinfer": "def5678",
    "sgl_cookbook": "ghi9012",
    "flashinfer_trace": "jkl3456"
  },
  "kernels": [
    {
      "definition_name": "gqa_paged_decode_h40_kv8_d128_ps1",
      "op_type": "gqa_paged",
      "phase1_status": "new",
      "fi_status": "fi_supported",
      "sgl_status": "sgl_integrated",
      "phase2_status": "done",
      "phase3_status": "done",
      "workload_entries": 8
    },
    {
      "definition_name": "new_op_h512",
      "op_type": "new_op",
      "phase1_status": "new",
      "fi_status": "fi_missing",
      "fi_issue_url": "https://github.com/flashinfer-ai/flashinfer/issues/999",
      "phase2_status": "done",
      "phase3_status": "skipped (fi_missing)"
    }
  ],
  "phase4": {
    "flashinfer_trace_pr": "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/discussions/42",
    "flashinfer_bench_pr": "https://github.com/flashinfer-ai/flashinfer-bench/pull/57"
  }
}
```

If the skill is re-run with the same model slug, load this manifest to skip already-completed
phases and resume from where the previous run stopped.

---

## Error Handling

### HuggingFace config unavailable
- Private model or network error: note model as deferred, continue with other models.

### FlashInfer issue creation fails
- Requires `gh` CLI authenticated with write access to `flashinfer-ai/flashinfer`.
- If not authenticated, print the issue body and prompt the user to file manually.

### SGLang PR creation fails
- Requires `gh` CLI authenticated with write access to `sgl-project/sglang`.
- If not authenticated, print the diff and PR body for manual submission.

### Workload collection fails
- GPU OOM: retry with smaller `--num-samples`, or reduce `--tp`.
- SGLang server startup failure: check `nvidia-smi`, reduce model size or `--tp`.
- Zero tensor dumps: verify `FLASHINFER_DUMP_INCLUDE` matches the actual API name from the
  definition's `fi_api` tag.

### HuggingFace PR creation fails
- Requires `huggingface_hub` authenticated with write access to `flashinfer-ai/flashinfer-trace`.
- Fall back to prompting the user to open the PR manually from `tmp/flashinfer-trace`.

---

## Prerequisites

- `gh` CLI installed and authenticated (`gh auth status`)
- `huggingface_hub` Python package installed and logged in (`huggingface-cli login`)
- GPU available for workload collection (Phase 3)
- `/clone-repos` has been run at least once (Phase 0 will update)

---

## Integration with Other Skills

This skill orchestrates all existing skills:

```
/onboard-model
  └── /clone-repos         (Phase 0)
  └── /track-models        (Phase 1 + Phase 4b)
  └── /extract-kernel-definitions  (Phase 2b)
  └── /collect-workloads   (Phase 3)
```

To run a sub-phase in isolation:

```bash
# Just discover models (Phase 1 only)
/onboard-model --discover --phases 1

# Just generate definitions for a known model (Phase 2 only)
/onboard-model --model-name qwen3-235b-a22b --phases 2

# Just collect workloads for already-generated definitions (Phase 3 only)
/onboard-model --model-name qwen3-235b-a22b --phases 3

# Just submit PRs for already-collected workloads (Phase 4 only)
/onboard-model --model-name qwen3-235b-a22b --phases 4
```

## Maintaining This Document

Update this file when:
- New op_types are added (update Phase 1e and 3b tables)
- FlashInfer issue / SGLang PR templates change
- HuggingFace PR submission method changes
- State manifest schema changes

## See Also

- [clone-repos](../clone-repos/SKILL.md)
- [track-models](../track-models/SKILL.md)
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [collect-workloads](../collect-workloads/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
