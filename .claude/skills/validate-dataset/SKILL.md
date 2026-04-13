---
name: validate-dataset
description: Validate the correctness and completeness of a FlashInfer Trace dataset. Use when checking dataset quality, verifying definitions/workloads/solutions/traces, debugging data issues, or preparing a dataset for release.
---

# Validate Dataset

Run structural, schema, and coverage checks on a FlashInfer Trace dataset. Produces a JSON report and optional text summary.

## Usage

```bash
# Full validation (including GPU benchmark)
/validate-dataset

# Skip GPU benchmark
/validate-dataset --disable-gpu

# Validate specific op_type or definition
/validate-dataset --op-types mla_paged --disable-gpu
/validate-dataset --definitions rmsnorm_h4096 --disable-gpu
```

## Parameters

- `--dataset`: Dataset root (default: `FIB_DATASET_PATH` env var)
- `--op-types`: Filter by op_type (union with `--definitions`)
- `--definitions`: Filter by definition name
- `--checks`: Comma-separated subset of: `layout,definition,workload,solution,trace,baseline,benchmark`
- `--disable-gpu`: Skip benchmark checks (default: GPU enabled)
- `--outputs`: Comma-separated output targets: `stdout,json,text` (default: all three)
- `--output-folder`: Report output directory (default: `<dataset>/reports/`)

## Implementation

### Files

| File | Purpose |
|------|---------|
| `flashinfer_bench/data/validate.py` | Core validation logic: `validate_dataset()` |
| `flashinfer_bench/data/validate_render.py` | Report models and rendering: `render_report()` |
| `flashinfer_bench/cli/main.py` | CLI entry: `flashinfer-bench validate`, `flashinfer-bench validate-render` |

### Steps

1. **Run validation via CLI**:
   ```bash
   flashinfer-bench validate --dataset tmp/flashinfer-trace --disable-gpu
   ```

2. **Or call Python API directly**:
   ```python
   from flashinfer_bench.data.validate import validate_dataset
   report = validate_dataset(dataset="tmp/flashinfer-trace", disable_gpu=True)
   ```

3. **Re-render an existing report**:
   ```bash
   flashinfer-bench validate-render reports/report-20260413-120000.json
   ```

### Check Categories

| Category | What it checks |
|----------|---------------|
| `layout` | Duplicate definition names, directory structure, path-field consistency, blob file existence |
| `definition` | Pydantic schema, reference Python syntax, axis references, descriptions, build reference (GPU) |
| `workload` | Schema, axes > 0, shape inference, safetensors blobs, no solution/evaluation |
| `solution` | Pydantic schema, entry_point format, sources, path-field consistency |
| `trace` | Schema, must have solution+evaluation, solution existence, workload coverage |
| `baseline` | Baseline solution/trace existence, build baseline solutions (GPU), PASSED status, workload coverage |
| `benchmark` | GPU: run baseline + reference via `Benchmark.run_all()` (lightweight config) |

### Report Output

- JSON and text reports are timestamped (`report-YYYYMMDD-HHMMSS.json/.txt`) to prevent overwriting
- Reports go to `<dataset>/reports/` by default
- See `docs/flashinfer-trace/validate.mdx` for full report structure and examples

## See Also

- [docs/flashinfer-trace/validate.mdx](../../docs/flashinfer-trace/validate.mdx) — full documentation
- [collect-workloads](../collect-workloads/SKILL.md) — collect workloads before validation
- [add-reference-tests](../add-reference-tests/SKILL.md) — add reference tests for definitions
