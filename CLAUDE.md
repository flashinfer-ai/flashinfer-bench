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
├── flashinfer_trace/           # Internal trace dataset (see below)
│   ├── definitions/            #   Kernel definition JSON files, by op_type
│   ├── tests/                  #   Definition tests and reference tests
│   └── workloads/              #   Workload JSONL files
├── tests/                      # Pytest test suite
├── scripts/                    # Standalone scripts (workload collection, sanitization)
├── tools/                      # Developer tools (GPU locking, etc.)
├── docs/                       # Documentation (model coverage, op_type schemas)
├── web/                        # Web UI for visualization
├── examples/                   # Example code and benchmarks
├── .claude/skills/             # Agent skill definitions (see below)
└── tmp/                        # Cloned external repos (SGLang, FlashInfer, sgl-cookbook)
```

## Trace Dataset

### Internal Trace Layer (`flashinfer_trace/`)

The `flashinfer_trace/` directory is the repo-managed trace layer. Definitions are organized
by `op_type` subdirectory:

```
flashinfer_trace/
├── definitions/{op_type}/{definition_name}.json
├── tests/references/test_{definition_name}.py
└── workloads/{op_type}/{definition_name}.jsonl
```

Browse `flashinfer_trace/definitions/` to see the current set of supported op_types.
Each op_type subdirectory contains one JSON file per kernel definition.

### External Dataset

The canonical published dataset lives at
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
on HuggingFace. It contains definitions, baseline solutions, workloads, and evaluation traces.

### Lifecycle

1. Generate or update definitions in the internal `flashinfer_trace/` directory
2. Open a PR in this repository for review
3. After merge, manually sync to the external HuggingFace dataset

New definitions should always be generated into the internal trace layer first.
The external dataset is not the primary edit surface.

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

Refer to `flashinfer_trace/definition.md` for the complete schema documentation.

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

### `flashinfer_trace/` is the complete dataset

Not necessarily. `flashinfer_trace/` is the internal repo-managed trace layer. The canonical
published dataset lives at
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
on HuggingFace and may contain additional content (baseline solutions, evaluation traces).

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

1. Create operation documentation in `docs/op_type_schema/`
2. Create Definition JSON files under `flashinfer_trace/definitions/{new_op_type}/`
3. Provide a Python reference implementation in the definition's `reference` field
4. Create Solution implementations (Triton/CUDA optimized)
5. Optionally create a FlashInfer adapter in `flashinfer_bench/integration/`

The existing op_type directories under `flashinfer_trace/definitions/` serve as templates.

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
- [Definition Schema Documentation](flashinfer_trace/definition.md)
- [Operation Type Schema](docs/op_type_schema/)
