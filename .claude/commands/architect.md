---
description: Enter architect loop — continuous monitoring of parallel definition onboarding agents
allowed-tools: Bash(git *), Bash(tools/*), Bash(ls*), Bash(mkdir*), Bash(test*)
---

# Architect Loop — Parallel Definition Onboarding Monitor

You are now operating as the FlashInfer-Bench **co-architect** in continuous monitoring mode.

## Setup — Resume Detection

Before starting, check for an interrupted session:

```bash
tools/architect resume
```

If there are **incomplete tasks or stale agents**, this is a **resumed session**:
1. Report what was found (completed tasks, incomplete tasks, dirty worktrees)
2. Ask the user which tasks to re-spawn vs skip
3. For tasks the user wants to continue: `tools/architect spawn <name>` after reviewing the task spec
4. For completed tasks (all 3 PRs open): report them for cleanup

If there are **no active worktrees**, this is a fresh start — proceed normally.

Then, activate the polling loop:

```bash
tools/architect-setup-loop.sh $ARGUMENTS
```

If the loop is already active, the script will tell you. Otherwise it creates the state file and the stop hook will keep the loop running.

## Polling Workflow (every iteration)

1. **Check status**: `tools/architect status`
2. **Check progress**: `tools/architect progress`
3. **For each worktree** (bench-* entries), evaluate and act:

| Condition | Action |
|-----------|--------|
| Agent stopped + both PRs open + clean | Report: ready for cleanup (`tools/architect cleanup`) |
| Agent stopped + some PRs missing + has commits | Rescue: `tools/architect rescue <name> --action commit -m "WIP: agent checkpoint"` |
| Agent running | Report progress summary (status, current step, PRs opened so far) |
| Ready to spawn (has task spec, no agent) | Report: suggest user runs `tools/architect spawn <name>` |
| GPU locked | Report: GPU busy, workload collection in progress |
| Idle (no task spec, no agent) | Report: idle worktree, needs task spec |

4. **Report**: One-line summary per task (name, agent status, PRs 1/2 open/pending)
5. **Completion**: If all active tasks have both PRs open, output: `<promise>ALL_TASKS_DONE</promise>`

## Post-Completion Review (when both PRs open)

After a definition reaches all-PRs-open state, run this checklist:

**PR 1 (GitHub flashinfer-bench):**
1. **Definition JSON**: Verify `flashinfer_trace/definitions/{op_type}/{name}.json` exists
2. **Reference test**: Verify `flashinfer_trace/tests/references/test_{name}.py` exists
3. **Coverage**: Verify `docs/model_coverage.mdx` updated to ✅ for this definition
4. **Test results**: Verify PR description includes reference test stdout
5. **PR2 link**: Verify PR description includes a link to the HuggingFace PR 2 (workload addition)
6. **Tags**: Check `status:verified` (or `status:unverified` if FlashInfer missing), `fi_api:*`, `ep:*`/`tp:*` if applicable

**PR 2 (HuggingFace flashinfer-trace):**
1. **Workloads**: Verify `workloads/{op_type}/{name}.jsonl` exists
2. **Blobs**: Verify `blob/workloads/{op_type}/{name}/*.safetensors` exist
3. **Baseline solution**: Verify `solutions/baseline/{op_type}/{name}/flashinfer_wrapper_*.json` exists (FlashInfer API wrapper, NOT a reference implementation)
4. **Eval trace**: Verify `traces/{op_type}/{name}.jsonl` exists and all entries have `evaluation.status == "PASSED"`
5. **Definition JSON**: Verify `definitions/{op_type}/{name}.json` copied from PR 1
6. **Reference test**: Verify `tests/references/test_{name}.py` copied from PR 1

Report findings per definition. Do NOT modify worktrees yourself — report issues back to the agent or user.

## Rules

- **Zero self-work**: You coordinate, never write definitions or collect workloads yourself
- **Auto-cleanup only when safe**: all 3 PRs open + clean worktrees + no running agents
- **Do NOT spawn agents**: report readiness and let the user decide (GPU contention risk)
- **Keep reports concise**: focus on changes and actions taken
- **GPU awareness**: workload collection requires GPU; never suggest spawning multiple GPU agents simultaneously

## Stopping

- User runs `/cancel-architect` to stop the loop
- Or output `<promise>ALL_TASKS_DONE</promise>` when all definitions have both PRs open

## Creating Tasks for Agents

Task specs go in `.claude/TASK.md` inside each worktree.
The `spawn` command reads from `.claude/TASK.md`.

**Preferred workflow:**
```bash
tools/architect create <def-name>    # Scaffolds bench + trace worktrees + .claude/TASK.md
# Review/edit tmp/worktrees/bench-<def-name>/.claude/TASK.md
tools/architect spawn <def-name>     # Agent reads TASK.md, works on 3 PRs
```

**Every TASK.md for definition onboarding must include:**
```
## Objective
Submit 2 PRs for definition <name>:
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
- `traces/{name}_baseline.jsonl` (all entries must have `evaluation.status == "PASSED"`)
- `definitions/{op_type}/{name}.json` (copied from PR 1)
- `tests/references/test_{name}.py` (copied from PR 1)

## Progress Reporting
Write .agent-progress.md after every major step:
  Status: in_progress | completed | blocked
  Done: <what's done>
  Current: <what you're doing now>
  Next: <next step>
  Blockers: <if any>
  - PR 1: <url or pending>
  - PR 2: <url or pending>

## GPU Work
Use tools/gpu-lock before any SGLang workload collection:
  tools/gpu-lock --gpus <N> --exec-timeout 1800 -- python collect_workloads.py ...
Where N matches the TP value (1 GPU for TP=1, 4 GPUs for TP=4, etc.)
```

## CLI Reference

```
tools/architect create <name>        # Create bench+trace worktrees + scaffold TASK.md
tools/architect status               # All worktrees overview
tools/architect progress             # All agent progress reports
tools/architect check <name>         # Detailed single task view
tools/architect prs <name>           # Show PR URLs extracted from progress file
tools/architect cleanup              # List definitions with all 3 PRs open (ready for review)
tools/architect spawn <name>         # Spawn agent (non-blocking)
tools/architect attach <name>        # Attach to running agent interactively
tools/architect kill <name>          # Stop running agent
tools/architect resume               # Detect interrupted session, report status
tools/architect resume --rescue      # + auto-commit dirty worktrees
tools/architect resume --respawn     # + re-spawn agents for incomplete tasks
tools/architect remove <name> -y     # Remove worktrees (after PRs merged)
```
