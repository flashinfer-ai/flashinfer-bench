---
name: submit-onboarding-prs
description: Open the per-definition pair of PRs that publishes a model onboarding — PR 2 to the HuggingFace flashinfer-trace dataset (definition + reference test + baseline solution + workloads + blobs + eval traces) and PR 1 to flashinfer-bench (docs/model_coverage.mdx update only). Use as Phase 4 of /onboard-model.
---

# Submit Onboarding PRs

For each new definition that has reached "ready" state (definition JSON written, workloads
collected, baseline eval passing), open two atomic PRs in sequence:

| # | Target repo | Content | Trigger |
|---|-------------|---------|---------|
| 2 | `flashinfer-ai/flashinfer-trace` (HuggingFace) | definition JSON + reference test + baseline solution + workload JSONL + safetensors blobs + eval traces | after Phase 3 |
| 1 | `flashinfer-ai/flashinfer-bench` (GitHub) | `docs/model_coverage.mdx` update only | after PR 2 is open (so PR 1 can link to it) |

After the trace-dataset refactor, the local `flashinfer_trace/` directory does not exist in
`flashinfer-bench`; everything trace-related lives in the HuggingFace dataset. PR 1 is
**only** the coverage-doc update plus a back-link to PR 2.

**Rule: one definition = one pair of PRs.** Do not batch multiple definitions into one PR —
each must be independently reviewable and mergeable.

## Usage

```bash
# Process all "ready" definitions for a model (per the manifest)
/submit-onboarding-prs --manifest tmp/onboard_qwen3-235b-a22b_20260427.json

# Limit to a specific subset of definitions
/submit-onboarding-prs \
  --manifest tmp/onboard_qwen3-235b-a22b_20260427.json \
  --definitions gqa_paged_decode_h40_kv8_d128_ps1,gqa_paged_decode_h40_kv8_d128_ps64

# Dry-run: print the worktree plan + PR titles without committing or pushing
/submit-onboarding-prs --manifest ... --dry-run
```

## Parameters

- `--manifest` (required): Path to the onboard-model run manifest. Reads `model_slug`,
  `hf_repo_id`, `repo_shas`, and the per-kernel statuses; writes back the `phase4` block
  with PR URLs as it makes progress.
- `--definitions` (optional): Comma-separated subset to process. Default: every kernel with
  `phase2_status=done` and (`phase3_status=done` OR `fi_status=fi_missing`).
- `--dry-run` (optional): Print the worktree layout, agent plan, and PR titles; do not write
  commits or open PRs.

## Prerequisites

- `gh` CLI authenticated for `flashinfer-ai/flashinfer-bench` (PR 1).
- `huggingface_hub` authenticated for `flashinfer-ai/flashinfer-trace` (PR 2).
- For each definition: definition JSON in `tmp/flashinfer-trace/definitions/{op_type}/`,
  workload JSONL + blobs in the corresponding `tmp/flashinfer-trace/workloads/` and
  `tmp/flashinfer-trace/blob/workloads/` paths, baseline solution under
  `tmp/flashinfer-trace/solutions/baseline/{op_type}/{name}/`, and eval traces under
  `tmp/flashinfer-trace/traces/{op_type}/{name}.jsonl` with every entry showing
  `evaluation.status == "PASSED"`. (`fi_missing` definitions skip workloads/baseline/traces.)
- `pre-commit` installed (PR 1 must pass `pre-commit run --all-files`).

---

## Phase 4-setup: Create worktrees before spawning agents

For each definition, create two worktrees up front so agents can run in parallel:

```bash
DATE=$(date +%Y%m%d)

# Worktree 1: flashinfer-bench (this repo) — for docs/model_coverage.mdx update
git worktree add \
  tmp/worktrees/bench-{definition_name} \
  -b feat/def-{definition_name}

# Worktree 2: flashinfer-trace (HuggingFace dataset clone) — for definition JSON,
# reference test, baseline solution, workloads, blobs, eval traces
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
        ├── bench-{def1}/          # isolated branch for def1 model_coverage.mdx update
        ├── bench-{def2}/          # isolated branch for def2 model_coverage.mdx update
        ├── trace-{def1}/          # isolated branch for def1 dataset content (HF repo)
        └── trace-{def2}/          # isolated branch for def2 dataset content (HF repo)
```

## Phase 4-spawn: One agent per definition, in parallel

Spawn all definition agents simultaneously. Each agent owns its two worktrees and runs
**Phase 4a then Phase 4b** (below) end-to-end. Write a `.claude/TASK.md` into each agent's
bench worktree using the [TASK.md template](#agent-taskmd-template) and include the two
worktree paths plus the staging paths for the definition JSON, workloads, and blobs. The
agent reports the two PR URLs when done.

## Phase 4a: PR 2 — HuggingFace flashinfer-trace

PR 2 is opened **first** so PR 1 can link to it.

**Check first** — if a baseline solution already exists in
`tmp/flashinfer-trace/solutions/baseline/{op_type}/{definition_name}/`, skip creating a new
one and skip running `flashinfer-bench run`. Include the existing solution files in the PR
commit as-is; do not regenerate eval traces. Only create a new baseline solution and run
eval when no solution exists yet.

Inside `tmp/worktrees/trace-{definition_name}/`:

```bash
# 1. Copy kernel definition JSON (canonical home — only lives in flashinfer-trace now)
cp tmp/flashinfer-trace/definitions/{op_type}/{definition_name}.json \
   tmp/worktrees/trace-{definition_name}/definitions/{op_type}/

# 2. Write the reference test (use the add-reference-tests skill)
# Output: tmp/worktrees/trace-{definition_name}/tests/references/test_{definition_name}.py
# It must validate the definition's `reference` field against FlashInfer/SGLang ground truth.

# 3. Baseline solution
# If solutions/baseline/{op_type}/{definition_name}/*.json already exists in the HF clone,
# copy it through unchanged. Otherwise create a FlashInfer-API-wrapper baseline (NOT a
# copy of `reference`) and run `flashinfer-bench run` to generate eval traces.

# 4. Copy workload JSONL and safetensors blobs
cp -r tmp/flashinfer-trace/workloads/{op_type}/{definition_name}.jsonl \
      tmp/worktrees/trace-{definition_name}/workloads/{op_type}/
cp -r tmp/flashinfer-trace/blob/workloads/{op_type}/{definition_name}/ \
      tmp/worktrees/trace-{definition_name}/blob/workloads/{op_type}/

# 5. Copy eval traces (must show all entries PASSED)
cp tmp/flashinfer-trace/traces/{op_type}/{definition_name}.jsonl \
   tmp/worktrees/trace-{definition_name}/traces/{op_type}/

cd tmp/worktrees/trace-{definition_name}
git add definitions/{op_type}/{definition_name}.json \
        tests/references/test_{definition_name}.py \
        solutions/baseline/{op_type}/{definition_name}/ \
        workloads/{op_type}/{definition_name}.jsonl \
        blob/workloads/{op_type}/{definition_name}/ \
        traces/{op_type}/{definition_name}.jsonl
git commit -m "Add {definition_name}: definition + reference test + baseline solution + workloads + traces

Model: {hf_repo_id}
SGLang: {sglang_commit_sha}
FlashInfer: {flashinfer_commit_sha}
Workload entries: {num_workload_entries}
"
git push origin workloads-{date}-{definition_name}
python -c "
from huggingface_hub import HfApi
HfApi().create_pull_request(
    repo_id='flashinfer-ai/flashinfer-trace',
    repo_type='dataset',
    title='Add {definition_name}: definition + reference test + baseline solution + workloads + traces',
    description='...',
    head='workloads-{date}-{definition_name}',
)
"
# Record the resulting PR URL as pr2_url
```

## Phase 4b: PR 1 — GitHub flashinfer-bench (docs/model_coverage.mdx only)

After PR 2 is open and you have its URL, open PR 1. The diff must touch **only**
`docs/model_coverage.mdx`.

Inside `tmp/worktrees/bench-{definition_name}/`:

```bash
# Edit docs/model_coverage.mdx: mark {definition_name} row as ✅ for this model
# (and update the per-model summary table).

cd tmp/worktrees/bench-{definition_name}
pre-commit run --all-files
git add docs/model_coverage.mdx
git commit -m "docs: mark {definition_name} as covered for {model_display_name}

Tracks the dataset addition at:
{pr2_url}
{If fi_missing: FlashInfer issue: flashinfer-ai/flashinfer#{issue_number}}
"
git push origin feat/def-{definition_name}
gh pr create \
  --repo flashinfer-ai/flashinfer-bench \
  --title "docs: mark {definition_name} as covered for {model_display_name}" \
  --body "$(cat <<EOF
## Summary
- Marks \`{definition_name}\` ({op_type}) as covered for **{model_display_name}** in
  \`docs/model_coverage.mdx\`.
- Definition JSON, reference test, baseline solution, workloads, blobs, and eval traces
  all live in the HuggingFace dataset — see ${pr2_url}.
${If fi_missing: - ⚠️ FlashInfer kernel missing — tracking issue: flashinfer-ai/flashinfer#{issue_number}}

## Files changed
- \`docs/model_coverage.mdx\`

## Linked PRs
- HuggingFace dataset PR: ${pr2_url}
EOF
)"
```

## Phase 4-cleanup: Remove worktrees after PRs are open

```bash
for def in {definition_names}; do
  git worktree remove tmp/worktrees/bench-${def}
  git -C tmp/flashinfer-trace worktree remove ../worktrees/trace-${def}
done
```

Update the manifest's `phase4` block with the recorded PR URLs.

---

## PR Review Checklist

Run after both PRs are open. **Both PRs must pass all items before the definition is
considered complete.** If any item fails, fix and re-push before requesting merge.

### PR 1 — GitHub flashinfer-bench (coverage doc only)

1. **Coverage**: `docs/model_coverage.mdx` updated — row for `{name}` shows ✅ for
   `{model_display_name}`, and the per-model summary table reflects the new count.
2. **Single-file change**: the diff touches **only** `docs/model_coverage.mdx`. No
   `flashinfer_trace/...` paths, no `tests/references/...`, no workload files, no blobs.
   (If anything else appears in the diff it belongs in PR 2 instead.)
3. **PR 2 link**: PR description links to the HuggingFace PR 2 by full URL.
4. **fi_missing note (if applicable)**: if the kernel is `fi_missing`, PR description links
   the FlashInfer kernel-request issue (`flashinfer-ai/flashinfer#{issue_number}`).
5. **pre-commit clean**: `pre-commit run --all-files` passes locally before push.

### PR 2 — HuggingFace flashinfer-trace (canonical dataset)

1. **Definition JSON**: `definitions/{op_type}/{name}.json` exists in the PR.
2. **Definition tags**: definition JSON has `status:verified` (or `status:unverified` when
   the FlashInfer kernel is missing), plus `fi_api:*` and `ep:*`/`tp:*` where applicable.
3. **Reference test**: `tests/references/test_{name}.py` exists in the PR and pytest runs
   green against the definition's `reference` field. PR description includes the full
   pytest stdout.
4. **Workloads**: `workloads/{op_type}/{name}.jsonl` exists and is non-empty.
5. **Blobs**: `blob/workloads/{op_type}/{name}/*.safetensors` files exist.
6. **Baseline solution**: `solutions/baseline/{op_type}/{name}/flashinfer_wrapper_*.json`
   exists — this must be a FlashInfer API wrapper (calls `BatchDecodeWithPagedKVCacheWrapper`
   or `BatchPrefillWithPagedKVCacheWrapper`), **not** a copy of the definition's `reference`.
7. **Eval traces**: `traces/{op_type}/{name}.jsonl` exists and every entry has
   `evaluation.status == "PASSED"` — no failures allowed.
8. **SGLang log**: PR description contains a `## SGLang Collection Log` section with the
   full stdout from the `collect_workloads.py sglang` run (model loading, workload counts,
   dump dir info). Workloads must be SGLang-collected (not synthetic) — real workloads have
   diverse `(batch_size, kv_length)` pairs drawn from actual inference. A uniform sweep like
   `batch_size=4096` with 1-page contexts is a red flag for synthetic data.
9. **Provenance**: commit/PR body records `Model`, `SGLang` and `FlashInfer` commit SHAs,
   and the workload-entry count.

---

## Fixing PR checklist failures

When a checklist item fails on an already-open PR, fix it in the same worktree and push a
follow-up commit to the same branch — both PR 1 and PR 2 update in place. **Never close and
re-open a PR for a fixable item**, and **never amend the head commit after the PR has been
reviewed** (push a new commit instead).

### PR 1 — flashinfer-bench (coverage doc only)

| Failed item | Fix |
|-------------|-----|
| 1. Coverage row not ✅ | Edit `docs/model_coverage.mdx`: flip the `{name}` row to ✅ for `{model_display_name}` and bump the per-model summary count. Commit + push to `feat/def-{name}`. |
| 2. Diff touches paths other than `docs/model_coverage.mdx` | `git restore --staged --source=origin/main -- :^docs/model_coverage.mdx` in the bench worktree; commit the cleaned diff. Anything you remove here belongs in the PR 2 worktree — re-stage it there if needed. |
| 3. Missing PR 2 link | `gh pr edit {pr1_number} --body-file -` and re-paste the body with the full HF PR URL. |
| 4. Missing fi_missing issue link | Same — append the `flashinfer-ai/flashinfer#{issue_number}` line to the PR body. |
| 5. pre-commit failed | Run `pre-commit run --all-files` in the bench worktree, fix what it reports, commit + push (do **not** use `--no-verify`). |

### PR 2 — flashinfer-trace (HuggingFace)

| Failed item | Fix |
|-------------|-----|
| 1. Definition JSON missing | Copy from `tmp/flashinfer-trace/definitions/{op_type}/{name}.json` into the trace worktree at the same path. Commit + `git push origin workloads-{date}-{name}`. |
| 2. Definition tags wrong | Edit the `tags` array in the JSON inside the worktree (`status:verified`/`status:unverified`, `fi_api:*`, `tp:*`/`ep:*`). Re-validate with `flashinfer-bench validate --dataset tmp/worktrees/trace-{name}`. Commit + push. |
| 3. Reference test missing or red | Run `/add-reference-tests --definition-name {name}`, save output under `tests/references/test_{name}.py` in the trace worktree, then `pytest tests/references/test_{name}.py -v` until green. Paste the full pytest stdout into the PR body via `huggingface_hub.HfApi().edit_discussion(...)` (or the dataset web UI). |
| 4. Workload JSONL missing/empty | Re-run `/collect-workloads --definition-names {name} --model-name {model}`, then copy the regenerated `workloads/{op_type}/{name}.jsonl` into the trace worktree. Commit + push. |
| 5. Blobs missing | Same flow as item 4 — `collect-workloads` writes the safetensors to `blob/workloads/{op_type}/{name}/`; copy them into the worktree. |
| 6. Baseline solution wrong (copies `reference` instead of wrapping FlashInfer) | Replace `solutions/baseline/{op_type}/{name}/flashinfer_wrapper_*.json` with one that calls the actual FlashInfer wrapper (`BatchDecodeWithPagedKVCacheWrapper` etc.). Re-run `flashinfer-bench run` to regenerate the eval trace, then commit both. |
| 7. Eval traces have non-PASSED entries | Inspect failing entries first — usually a baseline/wrapper bug or a tolerance issue. Fix the baseline (item 6) or the workload (items 4–5), re-run `flashinfer-bench run`, and commit the regenerated `traces/{op_type}/{name}.jsonl` only after every entry shows `evaluation.status == "PASSED"`. |
| 8. SGLang collection log missing or shows synthetic data | Re-run `/collect-workloads` with the real SGLang config (correct `--model-name`, full prompt set). Capture stdout to a file and paste it into the PR body under `## SGLang Collection Log`. A red flag is uniform `(batch_size, kv_length)` pairs — that's synthetic, not real inference. |
| 9. Provenance missing | Append `Model: {hf_repo_id}`, `SGLang: {sglang_sha}`, `FlashInfer: {flashinfer_sha}`, and `Workload entries: {count}` to the PR description (and the next commit message if you push another commit). |

After any PR 2 fix, refresh the PR description so the pytest stdout / SGLang log reflect
the latest state — old log output can mask the real status.

If the failure is a structural mistake (e.g. PR 1 contains workload files, PR 2 doesn't
contain the definition JSON), the cleanest recovery is to fix the worktrees and force-push
**only the per-definition feature branch** (never the dataset's `main`). Coordinate with the
reviewer before force-pushing a PR they've already reviewed.

---

## Agent TASK.md template

Write `.claude/TASK.md` into each agent's bench worktree. Keep it short — the canonical
content lives in [Phase 4a/4b](#phase-4a-pr-2--huggingface-flashinfer-trace) and the
[PR Review Checklist](#pr-review-checklist).

```markdown
## Objective
Submit two PRs for definition `{name}` per Phase 4a then Phase 4b in
`.claude/skills/submit-onboarding-prs/SKILL.md`. PR 2 (HF flashinfer-trace) opens first;
PR 1 (flashinfer-bench coverage doc) opens second and links to PR 2.

## Worktrees
- bench: `tmp/worktrees/bench-{name}/`  (branch `feat/def-{name}`)
- trace: `tmp/worktrees/trace-{name}/`  (branch `workloads-{date}-{name}`)

## Staging paths (already populated by earlier phases)
- definition: `tmp/flashinfer-trace/definitions/{op_type}/{name}.json`
- workloads:  `tmp/flashinfer-trace/workloads/{op_type}/{name}.jsonl`
- blobs:      `tmp/flashinfer-trace/blob/workloads/{op_type}/{name}/`

## Done criteria
Every item in the PR Review Checklist (PR 1 + PR 2) passes. Use the
[fix-up table](#fixing-pr-checklist-failures) when an item is missing after the PR is open.

## Progress reporting
Append to `.agent-progress.md` after each step (Status / Done / Current / Next / Blockers,
plus the two PR URLs when each is open).

## GPU work
Use `tools/gpu-lock` before any SGLang workload collection:
`tools/gpu-lock --gpus <N> --exec-timeout 1800 -- python collect_workloads.py ...`
where N matches the TP value (1 GPU for TP=1, 4 GPUs for TP=4, etc.).
```

---

## Output: run-manifest update

When invoked with `--manifest`, append/update the `phase4` block per definition:

```json
{
  "phase4": {
    "gqa_paged_decode_h40_kv8_d128_ps1": {
      "flashinfer_trace_pr": "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/discussions/42",
      "flashinfer_bench_pr": "https://github.com/flashinfer-ai/flashinfer-bench/pull/57"
    }
  }
}
```

## Error Handling

- **HuggingFace PR creation fails**: requires `huggingface_hub` authenticated with write
  access to `flashinfer-ai/flashinfer-trace`. Fall back to opening the PR manually from the
  worktree.
- **GitHub PR creation fails**: requires `gh` authenticated with write access to
  `flashinfer-ai/flashinfer-bench`. Print the diff and PR body for manual submission.
- **`pre-commit` failure**: do not bypass with `--no-verify`. Fix the formatting and create
  a new commit.

## See Also

- [onboard-model](../onboard-model/SKILL.md) — pipeline that invokes this skill at Phase 4
- [discover-models](../discover-models/SKILL.md) — Phase 1 counterpart
- [collect-workloads](../collect-workloads/SKILL.md) — produces workload JSONL + blobs
- [add-reference-tests](../add-reference-tests/SKILL.md) — produces the reference test in PR 2
