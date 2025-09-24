# Bring Your Own Kernel

This guide walks through **Contribute a Solution**, **Contribute a Workload**, and **Add a New Kernel**, plus testing/benchmarking tips and a minimal end‑to‑end example.

## Contribute a Solution

If you want to supply a custom implementation for an existing **Definition**:

1. **Pick the Definition** that matches your kernel.
   - Ensure the axes matches exactly.

2. **Author your implementation** to the exact signature.
   - Your function signature must accept the same inputs and return a dict mapping output names → tensors.
   - You may use Triton, CUDA, cutlass, or PyTorch — any backend is fine.

3. **Package Solution metadata** and add tests against the `reference`.
   - Include target arch, toolchain, author (human or LLM), known constraints.

4. **Benchmark** and record an `Trace` entry with the evaluation result.
   - Run the benchmark runner on representative **Workloads**.

### Solution PR Checklist

* Matches an existing **Definition** (or a new one was added, see below)
* Implementation returns the exact output dict
* Tests pass across representative workloads
* Benchmark results recorded

## Contribute a Workload

Workloads reflect **real serving requests** and are critical for fair comparisons.

* Enable `FLASHINFER_BENCH_ENABLE_TRACING=1` to capture axis values and input tensors.
* (Optional) Submit a PR with your trace lines and a short README.

### Tracing with CLI (env-vars)

1. **Choose an output dataset root** (optional):

```bash
export FLASHINFER_BENCH_DATASET_PATH=/root/flashinfer-trace
# defaults to ./flashinfer-trace if unset
```

2. **Enable tracing and run your engine or script:**

```bash
export FLASHINFER_BENCH_ENABLE_TRACING=1
python run_engine.py  # your serving or batch script
```

By default, all kernels with a matching **Definition** are traced.

3. **What gets saved & where (default layout):**

```
$FLASHINFER_BENCH_DATASET_PATH/
└── workloads/
    ├── *.jsonl               # workload records (FlashInfer Trace format)
    └── safetensors/          # tensor payloads (when dumped)
```

Writing tensors to file is **async** (background thread) to reduce runtime overhead.

### Tracing in code (fine-grained control)

If you want to target a subset of kernels / customize policies:

```python
import flashinfer_bench as fb

# 1) Pick which kernels to trace and how
from flashinfer_bench import TracingConfig

gqa_tracing = TracingConfig(
    tensor_dump_policy="dump_non_float",   # keep ragged/int tensors; skip large float payloads
    dedup_policy="shape_only",             # save first occurrence per input-shape signature
)

configs = {
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_tracing,
    # more kernel definitions...
}

# 2) Enable, run, then finalize
with fb.enable_tracing(tracing_configs=configs, dataset_dir="/root/flashinfer-trace"):
    run_engine()  # your inference loop
```

**Policies you can use right away:**

* `tensor_dump_policy`: `"dump_all"`, `"dump_none"`, `"dump_non_float"`, or a list of input names to dump.
* `dedup_policy`: `"keep_all"`, `"shape_only"`, `"keep_first_k"` (e.g., first k calls), or a custom callable `Workload -> key`.
  These reduce disk/time while keeping representative samples.

## Add a New Kernel Definition

When no existing Definition fits your operator, create one:

* Provide a descriptive `name`, `type`, and `description` according to the schema.
* Enumerate `axes` (mark as `var` or `const`).
* Specify `inputs`/`outputs` shapes and dtypes.
* Include a correct Python `reference` implementation.

### Workload Tracing (for your new Definition)
Once your Definition is ready, you can trace workloads as described above.

### Testing & Benchmarking
After you have a Definition and at least one Solution, run the benchmark suite:

```bash
pytest -v tests/benchmarking/test_benchmark_runner.py
```

This will validate correctness and generate benchmark results for all matching workloads.

## New Kernel PR checklist

* **Axes & identity** finalized (consts/vars + values)
* **Name** reflects const axes (`h32_kv4_d128_ps1`)
* **Inputs/outputs** specified and **reference** passes tests
* **Constraints** validate typical workloads (indptr/indices lengths, etc.)
* **Tracing configs** (keep non-floats, shape-only dedup) and a few captured workloads
* **Benchmarks**: at least one baseline solution & results recorded
* **Apply** works end-to-end with decorator **or** imperative call
