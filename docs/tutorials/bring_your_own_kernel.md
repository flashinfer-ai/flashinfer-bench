# Bring Your Own Kernel

This guide gives instructions on how to add Definitions, Solutions, capture Workloads, and record Evaluations by walking through each **component of the Trace**, with an end-to-end “apply at runtime” flow.

# Trace

A **Trace** is an atomic, immutable record of a single benchmark run. It links a specific `Solution` to a specific `Definition`, fixes the exact `workload` (shapes + data), and stores the complete `evaluation`. A folder of Definitions, Solutions, and Traces is your benchmark database.

## JSON Schema (top level)

| Field        | Type   | Required | Description                                       |
| ------------ | ------ | -------- | ------------------------------------------------- |
| `definition` | string | Yes      | The `name` of the `Definition` used in this run.  |
| `solution`   | string | Yes      | The `name` of the `Solution` tested.              |
| `workload`   | object | Yes      | Concrete shapes and input data used for this run. |
| `evaluation` | object | Yes      | Results, logs, and environment snapshot.          |

---

## Component 1: `definition`

**What it is.** The operator’s contract: axes (const/var), inputs/outputs, constraints, and a correct (not necessarily fast) `reference`.

**Identity rule.** Two kernels are the **same Definition** iff:

* They have the **same axes**,
* Each axis has the **same role** (`const` vs `var`),
* All `const` axes have the **same values**.

**Naming.** Include the meaningful `const` axes in the name:

```
<type>_<stage>_<axis tokens>
e.g. gqa_paged_decode_h32_kv4_d128_ps1
```

**How to add a new kernel Definition.**

1. Refer to schema, choose a `name` and `type`; write a clear `description` and helpful `tags`.
2. Specify `axes` with `type: const|var` (+ `value` for const).
3. Add `constraints` that relate axes to inputs (e.g., CSR shapes).
4. Specify `inputs`/`outputs` (names, shapes by axes, dtypes, optional layouts).
5. Provide a correct Python `reference` returning a tuple of outputs.
6. (Optional) Provide minimal tests that run the `reference` on tiny shapes.

---

## Component 2: `solution`

**What it is.** A concrete implementation of a Definition’s interface (Triton/CUDA/CUTLASS/PyTorch, etc.) plus metadata.

**Name & metadata.**

* Suggested: `<def_name>__<backend>__<author>__vX`
* Include: target archs, toolchain, author (human or LLM), constraints/notes.

**Interface.** Your function must take the Definition’s `inputs` and **return** the tuple of `outputs`.

**How to add a Solution.**

1. Add the implementation of the kernel (matching signature).
2. Add unit tests vs `reference` across a representative shapes.
3. Provide metadata co-located with the code.

---

## Component 3: `workload`

**What it is.** The concrete axes + input data that instantiate a Definition for one run.

| Field    | Type   | Required | Description                                   |
| -------- | ------ | -------- | --------------------------------------------- |
| `axes`   | object | Yes      | Map of **var** axis → concrete int value.     |
| `inputs` | object | Yes      | Map of **input name** → **actual input**.     |

**How to capture workloads.**

* **Env-vars (zero-code):**

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

* **Tracing in code (fine-grained control)**

If you want to target a subset of kernels / customize policies:

```python
import flashinfer_bench as fb

# 1) Pick which kernels to trace and how
from flashinfer_bench import TracingConfig

gqa_tracing = TracingConfig(
    tensor_dump_policy="dump_non_float",   # keep scalar and int tensors; skip large float payloads
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

---

## Component 4: `evaluation`

**What it is.** The result bundle for one `(definition, solution, workload)` run.

**How to benchmark to produce Evaluations.**
Run the benchmarker over your `(definition, solution, workload)` triples:

  CLI:
    ```bash
    flashinfer-bench run --local ./flashinfer-trace --warmup-runs 10 --iterations 50 --save-results
    ```

  Python:
    ```python
    from flashinfer_bench import Benchmark, TraceSet
    trace_set = TraceSet.from_path("/dataset")
    benchmark = Benchmark(trace_set, warmup_runs=10, iterations=50, save_results=True)
    benchmark.run_all()  # scans dataset, runs all triples, appends traces
    ```

---

## Putting it together: Trace lifecycle

1. **Add the Definition**

   * Finalize axes (`const` vs `var`), constraints, I/O shapes, and `reference`.
   * Identity is locked by the axes set/roles/const values.

2. **Add one or more Solutions**

   * Implement the exact interface; return `{output_name: tensor}`.
   * Provide metadata and unit tests vs `reference`.

3. **Capture Workloads**

   * Run with tracing (env-vars or code) over real requests to collect shapes and, when helpful, actual inputs (esp. ragged index tensors).
   * Curate a small but representative set (use `shape_only` or `keep_first_k`).

4. **Benchmark → Emit Traces**

   * For each `(definition, solution, workload)` triple, run the benchmarker to produce one **Trace** JSON with `evaluation`.
   * Store logs and the environment snapshot alongside.

5. **Apply at runtime (end-to-end)**

   * Use runtime substitution to dispatch to the **best** ranked Solution for the current shapes.

---

# End-to-end “apply” (ties Trace back to serving)

**Decorator form (preferred):**

```python
import torch, torch.nn.functional as F
import flashinfer

@flashinfer.apply(lambda A, B: f"gemm_n_{B.shape[0]}_k_{B.shape[1]}")
def gemm(A, B):
    return F.linear(A, B)  # fallback/reference or a simple baseline
```

**Turn on runtime substitution:**

```bash
export FLASHINFER_BENCH_ENABLE_APPLY=1
python serve_or_benchmark.py
```

At call time, `apply` looks up the **Definition** (by name or via the lambda), matches the current **workload** (axes +, when required, data properties), and dispatches to the **best** `Solution` according to your recorded **Traces** (with correctness constraints and numeric tolerances enforced).

---

### Quick checklists

**Definition**

* Axes names/roles/const values finalized (identity).
* Inputs/outputs typed; constraints validate shapes.
* Reference returns exact `{output_name: tensor}`.

**Solution**

* Signature matches Definition; tests vs reference.
* Metadata (arch, toolchain, author, notes).
* Benchmarked to produce Traces.

**Workload**

* Representative axis values.
* Inputs saved (keep ragged/int tensors when relevant).
* Use dedup + sensible dump policy.

**Evaluation**

* Status + logs + environment saved.
* Correctness/performance filled per status rules.
