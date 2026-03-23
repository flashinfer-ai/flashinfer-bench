# FlashInfer-Bench Repo Guide

This document gives Claude and other agents repo-level context for working in `flashinfer-bench`.

Use this file for repository structure, trace data boundaries, and source-of-truth guidance.
Do not use this file as the detailed procedure for adding a model. End-to-end model onboarding lives in `.claude/skills/add-new-model/SKILL.md`.

## Project Overview

`flashinfer-bench` is a framework for:

- standardizing the FlashInfer Trace schema
- collecting real workloads from model serving
- benchmarking candidate kernel implementations
- storing benchmark traces
- applying the best known implementation at runtime

In short:

1. define kernel contracts
2. collect workloads
3. benchmark solutions
4. store traces
5. dispatch future calls using measured results

## Core Concepts

### Definition

A `Definition` is the formal contract for a kernel-like operation.

It specifies:

- the operation name and `op_type`
- symbolic axes
- input and output signatures
- constraints
- a reference implementation

### Workload

A `Workload` is a concrete instantiation of a `Definition`.
It binds runtime axis values and input data needed to replay the computation.

### Solution

A `Solution` is a concrete implementation of a `Definition`.
It may be written in Python, Triton, CUDA/C++, TileLang, or other supported forms.

### Trace

A `Trace` records the result of evaluating a `Solution` on a `Workload`.
It links definition, workload, solution, correctness, and performance.

### TraceSet

`TraceSet` is the main dataset abstraction in Python.
It bridges on-disk dataset layout and in-memory runtime logic, and many subsystems use it as their entry point.

### Model

A `Model` is the web-layer mapping from model modules to definitions.
Model metadata lives in `web/apps/web/data/models.ts`.

## Internal And External Trace

There are two distinct trace-related locations:

### Internal trace folder

`flashinfer_trace/` is the repo-local internal trace folder.

Use it for:

- definition JSON files reviewed in this repository
- reference tests for definition reference implementations
- schema-adjacent assets that should live in source control

### External trace dataset

The canonical external trace dataset is [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace).

Use it for the published dataset view of:

- `definitions/`
- `solutions/`
- `workloads/`
- `traces/`

Do not treat any workspace-local clone path as canonical. Local clones are environment-specific and should not be documented here as fixed locations.

## Current Definition Lifecycle

The current definition lifecycle is:

1. a skill extracts definitions from SGLang
2. generated definitions are written into internal `flashinfer_trace/`
3. changes are reviewed through a PR in this repository
4. after merge, the internal definitions are manually synced to the external dataset

This means:

- definition generation and review happen in this repository first
- the external dataset is not the primary review surface for new definitions
- the external dataset receives synced results after repository review

## Runtime Mental Model

The core runtime pipeline is:

1. `Definition` describes the contract
2. tracing turns real execution into replayable `Workload` data
3. benchmark runs `Solution` candidates on those workloads and records `Trace` results
4. apply uses existing traces to dispatch future calls to the best known implementation

Two relationships matter:

- benchmark is where correctness and performance become measured data
- `apply(...)` is the shared front door for workload collection and optimized dispatch

The internal trace folder is source-controlled and reviewable.
The external dataset is the published dataset-shaped artifact consumed by tracing, benchmark, and apply workflows.

## Repository Structure

### `flashinfer_bench/`

The main Python package and runtime core.

Important subpackages:

- `data/`: core data structures such as `Definition`, `Workload`, `Solution`, `Trace`, and `TraceSet`
- `tracing/`: workload collection runtime
- `bench/`: benchmark orchestration, runners, evaluators, timing
- `compile/`: builders and build registry
- `apply/`: runtime dispatch to the best known implementation
- `integration/`: framework-specific integration glue
- `serve/`: HTTP evaluation service
- `agents/`: agent-facing helpers
- `cli/`: command line entry points

### `flashinfer_trace/`

Repo-managed internal trace assets.

Think of this as the internal, reviewable trace layer in source control, especially:

- `flashinfer_trace/definitions/`
- `flashinfer_trace/tests/references/`

### `tests/`

Repository tests for runtime behavior and infrastructure.
These are different from reference tests under `flashinfer_trace/tests/references/`.

### `web/apps/web/`

Web UI for browsing kernels, traces, and models.
`web/apps/web/data/models.ts` is the source of truth for model metadata shown in the app.

### `docs/`

Published documentation source.

### `serve/`

Benchmark evaluation service.
This is not a general model inference server; it schedules and executes benchmark work over dataset-backed workloads.

## Supported `op_type`

Supported `op_type` information may be summarized here, but the source of truth is the trace dataset structure itself.

When you need the current set of supported `op_type` values, inspect:

- internal definitions under `flashinfer_trace/definitions/`
- external dataset definitions under `definitions/` in [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)

At the time of writing, the internal definitions currently include directories such as:

- `dsa_paged`
- `gdn`
- `gemm`
- `gqa_paged`
- `gqa_ragged`
- `mla_paged`
- `moe`
- `rmsnorm`
- `sampling`

Treat the dataset structure, not this prose list, as the source of truth.

## Benchmark Usage

Benchmarking evaluates candidate solutions against definition-backed workloads and writes trace results.

Conceptually, benchmark does the following:

1. load a dataset through `TraceSet`
2. select definitions, workloads, and solutions
3. build candidate solutions through the builder registry
4. run them with correctness and performance evaluation
5. store the resulting traces

The main CLI entry points are:

- `flashinfer-bench run` for local benchmark execution
- `flashinfer-bench serve` for HTTP-based benchmark evaluation
- `flashinfer-bench report ...` for analyzing stored trace results

## Where To Look By Task

### Tracing workloads

Start with:

- `flashinfer_bench/tracing/tracing.py`
- `flashinfer_bench/tracing/runtime.py`
- `flashinfer_bench/tracing/config.py`
- `flashinfer_bench/tracing/policy.py`

Then inspect:

- `flashinfer_bench/integration/flashinfer/`

Why: tracing converts runtime kernel calls into structured workload materialization.

### Running benchmarks

Start with:

- `flashinfer_bench/bench/benchmark.py`
- `flashinfer_bench/bench/config.py`
- `flashinfer_bench/bench/runner/`
- `flashinfer_bench/bench/evaluators/`

Why: this is where definitions, workloads, and solutions become measured traces.

### Runtime dispatch or apply behavior

Start with:

- `flashinfer_bench/apply/apply_api.py`
- `flashinfer_bench/apply/runtime.py`
- `flashinfer_bench/apply/key.py`
- `flashinfer_bench/apply/table.py`

Why: apply consumes benchmark evidence and decides which implementation runs at runtime.

### Definition correctness or reference behavior

Start with:

- `flashinfer_trace/definitions/`
- `flashinfer_trace/tests/references/`

Why: this is the internal, reviewable source-controlled layer for definition contracts and reference validation.

### Model coverage or web metadata

Start with:

- `web/apps/web/data/models.ts`
- `web/apps/web/lib/data-loader.ts`
- `docs/`

Why: this is where model metadata and dataset visualization are surfaced.

### Benchmark service behavior

Start with:

- `flashinfer_bench/serve/app.py`
- `flashinfer_bench/serve/scheduler.py`
- `flashinfer_bench/serve/task_store.py`

Why: the serve subsystem exposes benchmark orchestration as a service, not end-user inference.

### Dataset-facing questions

When the question is about published trace contents, workload coverage, submitted solutions, or synced definitions, reason against the external dataset [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace).

## Common Misunderstandings

### `flashinfer_trace/` is the whole dataset

Not necessarily.
`flashinfer_trace/` is the internal repo-managed trace layer, while the canonical published dataset lives in [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace).

### Tracing and apply are unrelated entry points

They are different runtimes, but they share the `apply(...)` call path.
This matters when reading runtime interception logic.

### Benchmark only measures speed

Benchmark also validates correctness against the reference implementation and stores evaluation results as traces.

### `serve/` is for generic inference traffic

It is not.
The serve subsystem is a benchmark orchestration service over dataset-backed workloads.

## Available Skills

Project skills live under `.claude/skills/`.

Use:

- `add-new-model` for end-to-end model onboarding
- `clone-repos` to prepare SGLang, FlashInfer, and sgl-cookbook under `tmp/`
- `extract-kernel-definitions` to extract definitions from a chosen model implementation
- `add-reference-tests` to add reference tests for definitions
- `track-models` to update model coverage tracking

## Maintenance Notes

Update `CLAUDE.md` when any of the following change:

- the internal vs external trace boundary
- the definition review and sync lifecycle
- the canonical external dataset location
- repository structure relevant to agent navigation

Keep this file concise. Prefer high-value repo navigation guidance over long task procedures or duplicated reference material.

Update the relevant skill files when task procedures change.
