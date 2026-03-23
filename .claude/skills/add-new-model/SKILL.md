---
name: add-new-model
description: Add support for a new model to flashinfer-bench by coordinating architecture analysis, definition extraction, reference tests, and model metadata updates. Use when the user asks to add a model, onboard a new architecture, or create the full model addition workflow.
---

# Add New Model

Use this skill for end-to-end model onboarding.

This skill is the canonical workflow for model addition. It coordinates the other project skills but does not replace their specialized instructions.

## Scope

Use this skill when the task involves one or more of:

- adding a brand new model to the repository
- extracting definitions for a model not yet covered
- updating `web/apps/web/data/models.ts`
- creating reference tests for newly added definitions
- explaining the current internal review and external sync lifecycle for model onboarding

Do not use this skill for definition extraction alone. For that, use [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md).

## Internal And External Trace Roles

Model onboarding currently uses two trace layers:

- Internal trace folder: `flashinfer_trace/`
- Canonical external dataset: [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)

Current lifecycle:

1. extract definitions into internal `flashinfer_trace/`
2. open a PR in this repository
3. review and merge here
4. manually sync the internal result to the external dataset afterward

Because of this lifecycle:

- new definitions should be generated into the internal trace folder first
- PR review should happen in this repository
- this skill should not treat the external dataset as the primary edit surface

## Required Inputs

Gather these inputs before making changes:

- model name used in the repository
- Hugging Face repo ID
- SGLang model implementation or model class
- whether the task includes model metadata updates in `web/apps/web/data/models.ts`
- whether new reference tests are required

## Workflow

### Step 1: Prepare repositories

If `tmp/sglang`, `tmp/flashinfer`, or `tmp/sgl-cookbook` are missing or stale, use [clone-repos](../clone-repos/SKILL.md).

### Step 2: Collect architecture facts

Use Hugging Face and SGLang as the sources for model architecture facts.

Collect:

- layer count
- hidden size
- attention head counts
- key/value head counts if applicable
- intermediate size
- MoE or sparse-attention specific constants if applicable

Also inspect sgl-cookbook for TP and EP configurations that affect generated definitions.

### Step 3: Extract definitions

Use [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md) to generate or update definitions under `flashinfer_trace/definitions/`.

This step should:

- inspect the SGLang implementation
- compute TP and EP variants when needed
- deduplicate against existing definitions
- keep new definition files in the internal trace folder

### Step 4: Add reference tests

For each new or changed definition that needs validation, use [add-reference-tests](../add-reference-tests/SKILL.md).

Reference tests belong under:

- `flashinfer_trace/tests/references/`

### Step 5: Update model metadata

If the model should appear in the web UI, update:

- `web/apps/web/data/models.ts`

Map model modules to the generated definitions. Keep this file aligned with the definitions actually added to the internal trace folder.

### Step 6: Summarize for review

Prepare a concise summary for the repository PR that covers:

- model being added
- new definitions added or reused
- new reference tests added
- model metadata updated
- anything that still requires manual follow-up

## Boundaries

This skill owns the end-to-end flow description.

The following skills stay specialized:

- [clone-repos](../clone-repos/SKILL.md): repository setup
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md): definition extraction
- [add-reference-tests](../add-reference-tests/SKILL.md): reference test generation
- [track-models](../track-models/SKILL.md): model coverage tracking

Do not duplicate their low-level implementation details here unless needed for orchestration.

## Outputs

Typical outputs of this workflow are:

- new or updated files in `flashinfer_trace/definitions/`
- new or updated files in `flashinfer_trace/tests/references/`
- new or updated entries in `web/apps/web/data/models.ts`
- a repository PR for internal review

The external dataset sync happens after merge and is currently manual.

## Quick Start

```bash
# 1. Prepare external repos if needed
/clone-repos

# 2. Extract definitions for the model
/extract-kernel-definitions --model-name deepseek_v3

# 3. Add reference tests for the new definitions
/add-reference-tests --op-type mla_paged
```

Then update `web/apps/web/data/models.ts` if the model metadata should be surfaced in the web app.
