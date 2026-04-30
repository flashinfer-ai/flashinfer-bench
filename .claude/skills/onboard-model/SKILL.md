---
name: onboard-model
description: End-to-end pipeline for discovering new LLMs with novel kernels and onboarding them into FlashInfer-Bench. Orchestrates repo updates, model discovery, kernel definition generation, workload collection, and PR submission.
---

# Onboard Model

Thin orchestrator that runs the five-phase pipeline for adding a new LLM to
FlashInfer-Bench. Each phase delegates to a focused skill — this file is the contract that
chains them together via a shared run manifest.

## Overview of phases

| Phase | Skill | Output |
|-------|-------|--------|
| 0 | [`/clone-repos`](../clone-repos/SKILL.md) | `tmp/sglang/`, `tmp/flashinfer/`, `tmp/sgl-cookbook/`, `tmp/flashinfer-trace/` cloned and current |
| 1 | [`/discover-models`](../discover-models/SKILL.md) | manifest `kernels[]` populated with `phase1_status`, `fi_status`, `fi_trace_template`, `sgl_status` |
| 2 | [`/extract-kernel-definitions`](../extract-kernel-definitions/SKILL.md) (+ inline `gh issue create` for `fi_missing`) | definition JSONs in `tmp/flashinfer-trace/definitions/` — auto-dumped via `FLASHINFER_TRACE_DUMP=1` for kernels with `fi_trace_template=true`, otherwise hand-written; manifest `phase2_status=done` (and `fi_issue_url` for fi_missing) |
| 3 | [`/collect-workloads`](../collect-workloads/SKILL.md) (+ inline SGLang PR for `sgl_missing`) | workloads + blobs in `tmp/flashinfer-trace/`; manifest `phase3_status=done` |
| 4 | [`/submit-onboarding-prs`](../submit-onboarding-prs/SKILL.md) | one HF PR + one bench PR per definition; manifest `phase4` populated |

The state contract between skills is the run manifest at
`tmp/onboard_{model_slug}_{date}.json` — see "Run manifest" below.

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
  Compatible with `--model-name`.
- `--model-name` (optional): Specific model slug to onboard (e.g. `qwen3-235b-a22b`).
- `--hf-repo-id` (optional): HuggingFace repo override. Inferred from `--model-name` if omitted.
- `--phases` (optional): Comma-separated list of phases to run (default: `0,1,2,3,4`).
- `--dry-run` (optional): Print what would be done without writing files or submitting PRs.
- `--skip-workload` (optional): Skip Phase 3 (e.g. when no GPU is available).
- `--submit-prs` (optional): Submit Phase 4 PRs (default: true).

---

## Phase 0: Update local repos

Delegate to the [`clone-repos`](../clone-repos/SKILL.md) skill.

```bash
/clone-repos
```

After the pull, capture the current SHAs and write them to the manifest's `repo_shas`:

```bash
git -C tmp/sglang rev-parse --short HEAD
git -C tmp/flashinfer rev-parse --short HEAD
git -C tmp/sgl-cookbook rev-parse --short HEAD
git -C tmp/flashinfer-trace rev-parse --short HEAD
```

Report the SHAs in the Phase 0 summary so the user can reproduce the run.

---

## Phase 1: Discover model + classify kernels

Delegate to the [`discover-models`](../discover-models/SKILL.md) skill, which produces the
manifest's `kernels[]` array — each entry tagged with `phase1_status` (existing/new),
`fi_status` (fi_supported/fi_missing/n-a), and `sgl_status` (sgl_integrated/sgl_missing/n-a).

```bash
# In auto-discover mode
/discover-models --discover --manifest tmp/onboard_{model_slug}_{date}.json

# In single-model mode
/discover-models --model-name {model_slug} --hf-repo-id {hf_repo_id} \
                 --manifest tmp/onboard_{model_slug}_{date}.json
```

Subsequent phases iterate over `kernels` and act based on the per-entry classification.

---

## Phase 2: Generate kernel definitions

For each kernel with `phase1_status=new`, generate a Definition JSON and write it into
`tmp/flashinfer-trace/definitions/{op_type}/`.

### 2b: fi_supported → trace-dump from a short SGLang pass (or manual fallback)

Delegate to [`extract-kernel-definitions`](../extract-kernel-definitions/SKILL.md). When
the kernel's FlashInfer API carries an `@flashinfer_api(trace=...)` decorator (i.e.
`fi_trace_template=true` in the manifest), one short SGLang inference pass with
`FLASHINFER_TRACE_DUMP=1` and `attention_backend="flashinfer"` produces complete Definition
JSONs for every shape it touches — `axes`, `inputs`, `outputs`, `tags` (`fi_api:*`,
`status:verified`), and `reference` are filled in by the dumper. For decorated kernels
that the model didn't exercise (e.g. an unused page-size variant) or for FlashInfer APIs
not yet decorated, the same skill falls back to manual extraction from sgl-cookbook +
HF config.

```bash
/extract-kernel-definitions --model-name {sglang_model_name}
```

Verify each new definition now exists:

```bash
find tmp/flashinfer-trace/definitions/ -name "{definition_name}.json"
```

Update each kernel's `phase2_status=done` in the manifest.

### 2a: fi_missing → manual SGLang-sourced reference + file kernel-request issue

When FlashInfer does not yet implement the kernel, the trace-dump path doesn't apply (no
decorated API to fire on). Generate the definition manually with SGLang's vanilla forward
as the reference (`extract-kernel-definitions` Path B), then file an issue against
`flashinfer-ai/flashinfer`. Mark the definition with the `status:unverified` tag.

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
A FlashInfer-Bench definition has been staged at:
`tmp/flashinfer-trace/definitions/{op_type}/{definition_name}.json`
(landing in the HuggingFace dataset PR for `flashinfer-ai/flashinfer-trace`)

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

- HuggingFace dataset PR: (link once PR 2 is open)
- SGLang model: `tmp/sglang/python/sglang/srt/models/{model_file}`
- HuggingFace model: https://huggingface.co/{hf_repo_id}
EOF
)"
```

Record the issue URL in the manifest as `fi_issue_url` for that kernel and add it to the
definition's `description`:

```
"description": "... See flashinfer-ai/flashinfer#<issue_number> for kernel implementation request."
```

**Do not proceed to Phase 3 for fi_missing kernels** — workload collection requires the
FlashInfer kernel to exist.

---

## Phase 3: Workload collection

Skip entirely for `fi_missing` kernels.

### 3a: Skip if workloads already exist

```bash
ls tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl 2>/dev/null
```

If the JSONL exists and is non-empty, mark `phase3_status=done` and skip.

### 3b: sgl_missing → submit SGLang integration PR first

For kernels classified `sgl_missing`, wire `{fi_api}` into the appropriate SGLang layer file
and open a PR against `sgl-project/sglang`. The change is typically small (import +
conditional dispatch).

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

```bash
cd tmp/sglang
git checkout -b feat/flashinfer-{op_type}-integration-{model_slug}
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

Record the SGLang PR URL on the kernel entry. Pause Phase 3 for that kernel until the PR
merges; resume the run with `--phases 3,4` once it lands.

### 3c: sgl_integrated → run workload collection

Delegate to [`collect-workloads`](../collect-workloads/SKILL.md):

```bash
/collect-workloads \
  --definition-names {definition_name} \
  --model-name {model_name} \
  --submit-pr false   # PRs are submitted in Phase 4
```

Verify outputs and update the manifest:

```bash
ls tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl
ls tmp/flashinfer-trace/blob/workloads/{op_type}/{definition_name}/
```

Mark `phase3_status=done` and record `workload_entries`.

---

## Phase 4: Submit PRs

Delegate to [`submit-onboarding-prs`](../submit-onboarding-prs/SKILL.md). It creates the
per-definition worktrees, spawns one agent per definition, and opens PR 2 (HuggingFace
dataset) followed by PR 1 (flashinfer-bench coverage doc) for each.

```bash
/submit-onboarding-prs --manifest tmp/onboard_{model_slug}_{date}.json
```

The skill writes back to the manifest's `phase4` block with the resulting PR URLs.

The PR Review Checklist and Agent TASK.md template both live inside that skill — refer to
`submit-onboarding-prs/SKILL.md` rather than duplicating them here.

---

## Run manifest

The contract between skills. Stored at `tmp/onboard_{model_slug}_{date}.json`:

```json
{
  "model_slug": "qwen3-235b-a22b",
  "hf_repo_id": "Qwen/Qwen3-235B-A22B",
  "date": "2026-04-27",
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
      "fi_trace_template": true,
      "sgl_status": "sgl_integrated",
      "phase2_status": "done",
      "phase2_method": "trace_dump",
      "phase3_status": "done",
      "workload_entries": 8
    },
    {
      "definition_name": "new_op_h512",
      "op_type": "new_op",
      "phase1_status": "new",
      "fi_status": "fi_missing",
      "sgl_status": "n/a",
      "fi_issue_url": "https://github.com/flashinfer-ai/flashinfer/issues/999",
      "phase2_status": "done",
      "phase3_status": "skipped (fi_missing)"
    }
  ],
  "phase4": {
    "gqa_paged_decode_h40_kv8_d128_ps1": {
      "flashinfer_trace_pr": "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/discussions/42",
      "flashinfer_bench_pr": "https://github.com/flashinfer-ai/flashinfer-bench/pull/57"
    }
  }
}
```

Re-running `/onboard-model` with the same `--model-name` loads the existing manifest and
skips any kernel/phase already marked `done` — phases resume from the first incomplete step.

---

## Error handling

### HuggingFace config unavailable
- Private model or network error: note model as deferred, continue with other models.

### FlashInfer issue creation fails
- Requires `gh` authenticated with write access to `flashinfer-ai/flashinfer`.
- If not authenticated, print the issue body and prompt the user to file manually.

### SGLang PR creation fails
- Requires `gh` authenticated with write access to `sgl-project/sglang`.
- If not authenticated, print the diff and PR body for manual submission.

### Workload collection fails
- GPU OOM: retry with smaller `--num-samples`, or reduce `--tp`.
- SGLang server startup failure: check `nvidia-smi`, reduce model size or `--tp`.
- Zero tensor dumps: verify `FLASHINFER_DUMP_INCLUDE` matches the actual API name from the
  definition's `fi_api` tag.

### Phase 4 PR creation fails
- See [`submit-onboarding-prs`](../submit-onboarding-prs/SKILL.md) "Error Handling".

---

## Prerequisites

- `gh` CLI installed and authenticated (`gh auth status`)
- `huggingface_hub` Python package installed and logged in (`huggingface-cli login`)
- GPU available for workload collection (Phase 3)
- `/clone-repos` has been run at least once (Phase 0 will update)

---

## Maintaining this skill

This file is a thin orchestrator. Update it only when:
- The set of phases changes (a new phase is added or removed).
- The run-manifest contract changes (new top-level field, new per-kernel field).
- Phase 2a's issue template or Phase 3b's SGLang PR template changes.

All phase-specific procedures live in the phase skills linked from the overview table at
the top — keep procedural detail there, not here.

To run a single phase by itself, invoke its skill directly (each is documented in its own
`SKILL.md`); `/onboard-model --phases N` is the same thing routed through the orchestrator
so the manifest stays in sync.

## See Also

- [clone-repos](../clone-repos/SKILL.md)
- [discover-models](../discover-models/SKILL.md)
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [collect-workloads](../collect-workloads/SKILL.md)
- [submit-onboarding-prs](../submit-onboarding-prs/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md) (used inside `submit-onboarding-prs`)
- [track-models](../track-models/SKILL.md) (per-op-type formula reference for Phase 1)
