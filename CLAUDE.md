# FlashInfer-Bench Repo Guide

This document gives agents repo-level context for working in `flashinfer-bench`.
Use it for repository structure, trace dataset conventions, and source-of-truth guidance.
Task-specific procedures live in `.claude/skills/`.

## Project Overview

FlashInfer-Bench is a GPU kernel optimization benchmarking framework for:
- Standardizing FlashInfer Trace format
- Real-time workload tracing and collection
- Automated kernel optimization and replacement
- Performance leaderboards and tracking

### Core Concepts

1. **Definition**: Specifies an operation's interface (inputs/outputs, axes, reference implementation)
2. **Solution**: Concrete implementation of a Definition (Python/Triton/CUDA)
3. **Workload**: Specific input configuration and test case
4. **Trace**: Execution record containing correctness and performance data
5. **Model**: Hierarchical module structure mapping model components to Definitions

## Repository Structure

```
flashinfer-bench/
├── flashinfer_bench/           # Main Python package
│   ├── data/                   #   Definition, Solution, Workload, Trace data classes
│   ├── bench/                  #   Benchmarking engine + evaluators
│   ├── compile/                #   Build system (Python/Triton/CUDA)
│   ├── apply/                  #   Kernel auto-replacement API
│   ├── serve/                  #   Benchmark orchestration service (NOT inference)
│   ├── integration/            #   FlashInfer integration
│   ├── tracing/                #   Workload tracing utilities
│   └── agents/                 #   Agent orchestration tools
├── tests/                      # Pytest test suite
├── scripts/                    # Standalone scripts (workload collection, sanitization)
├── tools/                      # Developer tools (GPU locking, etc.)
├── docs/                       # Documentation (model coverage, op_type schemas)
├── web/                        # Web UI for visualization
├── examples/                   # Example code and benchmarks
├── .claude/skills/             # Agent skill definitions (see below)
└── tmp/                        # Cloned external repos, including the
                                #   flashinfer-trace HF dataset clone
                                #   (SGLang, FlashInfer, sgl-cookbook, flashinfer-trace)
```

## Trace Dataset

### Single source of truth: HuggingFace

The canonical (and only) trace dataset lives at
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
on HuggingFace. It contains definitions, reference tests, baseline solutions, workloads,
blobs, and evaluation traces.

There is **no** in-repo `flashinfer_trace/` directory in `flashinfer-bench`. The earlier
"internal trace layer" was removed in the trace-dataset refactor (PR #418); definitions,
reference tests, and workloads no longer live here. Skills that need to read or edit trace
content do so through a local clone of the HF dataset at `tmp/flashinfer-trace/`:

```
tmp/flashinfer-trace/                  # local clone of the HuggingFace dataset
├── definitions/{op_type}/{definition_name}.json
├── tests/references/test_{definition_name}.py
├── solutions/baseline/{op_type}/{definition_name}/...
├── workloads/{op_type}/{definition_name}.jsonl
├── blob/workloads/{op_type}/{definition_name}/*.safetensors
└── traces/{op_type}/{definition_name}.jsonl
```

Browse `tmp/flashinfer-trace/definitions/` to see the current set of supported op_types
once `/clone-repos` has been run.

### Lifecycle

1. Run `/clone-repos` to ensure `tmp/flashinfer-trace/` is checked out and up to date.
2. Generate or update content under `tmp/flashinfer-trace/` and commit on a feature branch.
3. Open a PR against the HuggingFace dataset repo (PR 2 in the onboard-model flow).
4. Open a companion PR against `flashinfer-bench` that updates **only** `docs/model_coverage.mdx`
   to reflect the new coverage (PR 1 in the onboard-model flow).

The HuggingFace dataset is the primary edit surface — flashinfer-bench owns code, docs,
and the coverage doc, not the trace data itself.

### Definition JSON Structure

Each definition JSON follows a common structure:

```json
{
  "name": "...",
  "description": "...",
  "op_type": "...",
  "tags": ["stage:decode", "status:verified", "model:...", "fi_api:...", "tp:N"],
  "axes": { "batch_size": {"type": "var"}, "num_heads": {"type": "const", "value": 16} },
  "constraints": ["len_indptr == batch_size + 1"],
  "inputs": { "tensor_name": {"shape": ["axis1", "axis2"], "dtype": "bfloat16"} },
  "outputs": { "output": {"shape": ["axis1", "axis2"], "dtype": "bfloat16"} },
  "reference": "import torch\n\ndef run(...):\n    ..."
}
```

Key conventions:
- **Axes**: `type: "var"` for runtime dimensions (batch_size, seq_len); `type: "const"` with
  `value` for model-specific constants (num_heads, hidden_size)
- **Tags**: `stage:`, `status:`, `model:`, `fi_api:`, `tp:`, `ep:`, `quantization:` prefixes
- **Reference**: Plain PyTorch `run()` function serving as ground truth
- **TP/EP**: Some kernel types (attention, MoE) produce separate definitions per tensor/expert
  parallelism setting because parallelism changes constant axis values (e.g., head counts,
  local expert counts). Other kernel types (normalization, GEMM, RoPE, sampling) are
  parallelism-agnostic. See the `extract-kernel-definitions` skill for the full rules.

Refer to `docs/flashinfer-trace/definition.mdx` for the complete schema documentation.

## Documentation Structure (`docs/`)

```
docs/
├── Getting Started        # index, installation, quickstart
├── Tutorials              # run-benchmark, cli, server-api, bring-your-own-kernel
├── FlashInfer Trace       # definition, workload, solution, trace schemas
├── Dataset                # model_coverage
└── Op Type Reference      # per-op-type specs (gemm, gqa, mla, moe, sampling, ...)
```

Navigation is defined in `docs/docs.json`. Page files live under `docs/start/`,
`docs/tutorials/`, `docs/flashinfer-trace/`, and `docs/op-types/`.

## Where To Look By Task

### Understanding data structures

Start with `flashinfer_bench/data/`. This package defines `Definition`, `Solution`,
`Workload`, `Trace`, and `TraceSet` — the core data classes used throughout the codebase.

### Running or writing benchmarks

Start with `flashinfer_bench/bench/` for the benchmarking engine, and
`flashinfer_bench/compile/` for the build system that compiles solutions.

### Kernel auto-replacement at runtime

Start with `flashinfer_bench/apply/`. The `apply(...)` function is the shared entry point
for both optimized kernel dispatch and workload tracing.

### Model coverage or web metadata

Start with `web/apps/web/data/` for the web UI data layer, and `docs/` for
model coverage documentation.

### Benchmark service behavior

Start with `flashinfer_bench/serve/`. This subsystem exposes benchmark orchestration as
a service — it is **not** an inference server.

### Dataset-facing questions

When the question is about published trace contents, workload coverage, or synced definitions,
reason against the external dataset
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace).

### Agent and skill workflows

Start with `.claude/skills/`. Each subdirectory contains a `SKILL.md` with full instructions.

- **onboard-model**: End-to-end pipeline for discovering new LLMs and onboarding them
  (repo updates, model discovery, definition generation, workload collection, PR submission)
- **extract-kernel-definitions**: Extract kernel schemas from SGLang model implementations
  with deduplication, generate Definition JSON files
- **collect-workloads**: Collect real workloads from SGLang inference runs using FlashInfer
  logging API, sanitize and submit to flashinfer-trace
- **collect-workloads-bench**: Collect workloads using `bench_sharegpt.py` with model-specific
  server configs from `model_configs.json`
- **add-reference-tests**: Add pytest tests to validate reference implementations against
  FlashInfer or SGLang ground truth
- **track-models**: Track open-source LLMs and update `docs/model_coverage.mdx` with kernel
  support status
- **clone-repos**: Clone SGLang, FlashInfer, sgl-cookbook, and flashinfer-trace to `tmp/`

## Common Misunderstandings

### Tracing and apply are unrelated entry points

They are different runtimes, but the `apply(...)` call path is a shared entry point for both
optimized dispatch and workload collection (tracing). This matters when reading runtime
interception logic in `flashinfer_bench/apply/`.

### Benchmark only measures speed

Benchmark also validates correctness against the reference implementation and stores
evaluation results as traces.

### `serve/` is for generic inference traffic

It is not. The serve subsystem is a benchmark orchestration service over dataset-backed
workloads.

## Contributing New Operation Types

To add a new op_type beyond what currently exists:

1. Create operation documentation in `docs/op-types/`
2. Create Definition JSON files under `tmp/flashinfer-trace/definitions/{new_op_type}/`
   (the HuggingFace dataset clone — submit via a PR to `flashinfer-ai/flashinfer-trace`)
3. Provide a Python reference implementation in the definition's `reference` field
4. Create Solution implementations (Triton/CUDA optimized)
5. Optionally create a FlashInfer adapter in `flashinfer_bench/integration/`

The existing op_type directories under `tmp/flashinfer-trace/definitions/` serve as templates.

## Maintenance Notes

Update `CLAUDE.md` when any of the following change:

- The internal vs external trace boundary or sync lifecycle
- Repository directory structure
- Core concept definitions
- The definition JSON schema conventions

Update the relevant `.claude/skills/*.md` files when task procedures change.
Keep this file focused on repo-level context. Skill-specific procedures and
op_type-specific details belong in their respective skill files.

## References

- [FlashInfer Documentation](https://docs.flashinfer.ai)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [HuggingFace Hub](https://huggingface.co/models)
- [Definition Schema Documentation](docs/flashinfer-trace/definition.mdx)
- [Operation Type Schema](docs/op-types/)
