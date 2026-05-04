---
name: collect-workloads
description: Auto-collect workloads from SGLang inference runs using FlashInfer's logging API. Defaults to level-3 logging + the sanitize-fi-log skill (lightweight, random tensors). Opt in to level-10 logging + the sanitize-fi-dumps skill when real tensors are required (correctness eval, replay).
---

# Collect Workloads

Collect real-world workloads by running SGLang inference with FlashInfer's logging
API, then converting the resulting log/dumps into per-definition workload JSONL
files for the
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
HuggingFace dataset.

**No code changes to SGLang or FlashInfer are required.**  Collection works
entirely through FlashInfer's built-in logging API.

## Two collection modes

| | **Default — level 3** | **Opt-in — level 10** |
|---|---|---|
| When to use | Shape-coverage workloads, fast iteration, large logs (no tensor I/O) | Correctness eval / replay against the *exact* tensors a real run saw |
| FlashInfer env | `FLASHINFER_LOGLEVEL=3` (+ `FLASHINFER_TRACE_DUMP=1`) | `FLASHINFER_LOGLEVEL=10` + `FLASHINFER_DUMP_DIR` + `FLASHINFER_DUMP_*` filters |
| Output of SGLang run | One log file per rank (`fi_apilog_<pid>.log`) + per-definition trace JSONs (auto-dumped) | A dump directory of `metadata.jsonl` + safetensors per call, per rank |
| Sanitize step | [`/sanitize-fi-log`](../sanitize-fi-log/SKILL.md) | [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md) |
| Tensor storage in workload | Every tensor → `{"type": "random"}` | Structural ints (indptrs/indices) → safetensors; floats → `{"type": "random"}`; scalars → `{"type": "scalar", "value": …}` |
| Suitable for `flashinfer-bench run --solutions baseline` | No (no real structural tensors) | Yes |

**Default for new collections:** level 3.  Pick level 10 explicitly when you
need a published workload set with eval traces.

## Default flow — level 3 (lightweight)

```bash
# 1. Run SGLang with level-3 logging + trace-template auto-dump.
#    The trace dump produces one JSON per (op, shape) — the same files you'd
#    normally pull from FlashInfer's tests/trace/fi_trace_*.
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=fi_apilog_%i.log     # one log file per rank PID
export FLASHINFER_TRACE_DUMP=1
export FLASHINFER_TRACE_DUMP_DIR=$PWD          # writes definition JSONs alongside the logs

tools/gpu-lock --gpus 8 -- \
  python3 examples/sglang_bench/bench_serving.py \
    --model      <model-key> \
    --model-path /path/to/model \
    --dataset    random --isl 1024 --osl 1024 \
    --batch-sizes 64 128 \
    --num-batches 2

# 2. Convert one rank's log into per-definition workload JSONLs.
python3 scripts/sanitize_fi_log.py \
    --log-file  fi_apilog_<pid>.log \
    --output-dir workloads_out/ \
    --trace-dir  $PWD \
    --strict
```

Pick any one rank's log — for a TP=N model the kernel call distributions are
identical across ranks (we've validated this against several DSR1+SGLang runs;
each rank parses to the same per-definition counts).  The result is one JSONL
per matched definition (e.g. `rmsnorm_h7168.jsonl`,
`gemm_fp8_nt_groupwise_n2112_k7168.jsonl`) plus one raw JSONL per unmatched API.

See [`/sanitize-fi-log`](../sanitize-fi-log/SKILL.md) for the matching pipeline,
strict vs lenient mode, and how to triage raw entries.

### When to drop `--strict`

Strict mode rejects any call whose log shape doesn't fully agree with the
matched definition's schema (rank, dtype, const axes).  When all your trace
templates are correct that's the cleanest signal.  If a known
template-vs-runtime mismatch is being investigated, drop `--strict` so the
matcher's lenient name-based fallback fires and the workload is still emitted
under the right definition.

### What level 3 does *not* give you

- No real tensor values — you can't run `flashinfer-bench run --solutions baseline`
  on these workloads to produce eval traces.
- Therefore not appropriate for PR 2 to the HF flashinfer-trace dataset, which
  requires baseline-eval traces showing all entries `evaluation.status == "PASSED"`.

## Opt-in flow — level 10 (real tensors)

Use this when you need a workload set that drives correctness eval.  Hand off
to [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md) for the sanitize step;
that skill owns the dump-dir-handling details (plan/run pairing, kv_indices
trim, blob layout, env-var preparation).

### Streaming collection (preferred orchestrator)

```bash
# Must run under gpu-lock so CUDA_VISIBLE_DEVICES is set
tools/gpu-lock --gpus 8 --exec-timeout 10800 -- \
  python3 scripts/collect_stream.py \
    --def-name  gqa_paged_decode_h5_kv1_d128_ps64 \
    --model-key llama-4-scout-ps64 \
    --model-path /path/to/model \
    --batch-sizes 64 128 \
    --pr-num 263 \
    [--peer-node-addr nvl72089-T16] \
    [--trace-dir tmp/flashinfer-trace]

# Ragged prefill — add disable-radix-cache + disable-piecewise-cuda-graph
tools/gpu-lock --gpus 8 --exec-timeout 10800 -- \
  python3 scripts/collect_stream.py \
    --def-name  gqa_ragged_prefill_causal_h5_kv1_d128 \
    --model-key llama-4-scout \
    --model-path /path/to/model \
    --batch-sizes 64 128 \
    --pr-num 265 \
    --extra-server-flag --disable-radix-cache --disable-piecewise-cuda-graph

# Paged prefill — add enable-deterministic-inference
tools/gpu-lock --gpus 8 --exec-timeout 10800 -- \
  python3 scripts/collect_stream.py \
    --def-name  gqa_paged_prefill_causal_h5_kv1_d128_ps64 \
    --model-key llama-4-scout-ps64 \
    --model-path /path/to/model \
    --batch-sizes 64 128 \
    --pr-num 264 \
    --extra-server-flag --disable-cuda-graph --enable-deterministic-inference
```

`collect_stream.py` is the level-10 orchestrator: per batch size it sets the
`FLASHINFER_LOGLEVEL=10` + `FLASHINFER_DUMP_*` env vars (see
[`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md#preparing-the-dump-dir-level-10-sglang-run)
for the env-var details), runs `bench_serving.py` until `DUMP_MAX_COUNT` is hit,
calls `sanitize_dumps.py --max-new-workloads 4`, pushes the incremental JSONL +
blobs to HF, then `rm -rf`s the dump dir before the next batch size.  After all
batch sizes: `flashinfer-bench run --solutions baseline` → eval trace → push.

**Key flags** (see `python3 scripts/collect_stream.py --help` for the full set):
- `--dump-count 500` (default) — budget per server session.
- `--workloads-per-batch 4` (default) — workloads added per batch size.
- `--num-batches 2` (default) — inference rounds; budget typically hit in round 1.
- `--no-eval` — skip eval+trace push (useful when flashinfer-bench is unavailable).
- `--no-push` — dry run: collect and sanitize without uploading.
- `--replace-first` — replace instead of append on first batch size.

**Auto-detection from definition tags:**
- `tp:N` → sets `--tp N` (use `CUDA_VISIBLE_DEVICES=0,0` to simulate TP=2 on 1 GPU).
- `page_size` const axis → sets `--page-size N`.

### Older non-streaming entry point

`scripts/collect_workloads.py` runs the same pipeline but collects all batch
sizes into a single dump dir before pushing once.  Prefer `collect_stream.py`
for new collections — its incremental push avoids losing work if the run
crashes after partial collection.

## Workflow (level-10 path)

### Phase 0: Install Latest Packages

```bash
git -C tmp/flashinfer pull && git -C tmp/sglang pull
conda run -n flashinfer_bench pip install -e tmp/flashinfer --no-build-isolation
conda run -n flashinfer_bench pip install -e "tmp/sglang/python[all]"
```

### Phase 1: Resolve Target Definitions

- `--definitions <name> [name ...]`: specific definitions by name.
- `--op-type <type>`: all definitions under `definitions/{op_type}/`.
- `--all`: all definitions in the repo.

### Phase 2: FlashInfer Logging Configuration

Parses `fi_api:<dotted.api.name>` tags from each definition to build
`FLASHINFER_DUMP_INCLUDE`:

- Wrapper class APIs (e.g. `BatchDecodeWithPagedKVCacheWrapper`) → include
  `.run`, and `.plan` if the definition has `int32`/`int64` inputs.
- **`BatchPrefillWithRaggedKVCacheWrapper`**: SGLang calls
  `.forward()`/`.forward_return_lse()` (not `.run()`) — those are automatically
  added to `FLASHINFER_DUMP_INCLUDE` for Ragged wrappers.
- Plain function APIs (e.g. `rmsnorm`) → include by function name.

Env vars set automatically (full list in
[`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md#preparing-the-dump-dir-level-10-sglang-run)):
`FLASHINFER_LOGLEVEL=10`, `FLASHINFER_DUMP_DIR`, `FLASHINFER_DUMP_SAFETENSORS=1`,
`FLASHINFER_DUMP_INCLUDE`, `FLASHINFER_DUMP_EXCLUDE=*.__init__`,
`FLASHINFER_DUMP_MAX_COUNT=500`, `FLASHINFER_DUMP_MAX_SIZE_GB=30`.

### Phase 2b (optional): Piggyback definition trace dump

Setting `FLASHINFER_TRACE_DUMP=1` and `FLASHINFER_TRACE_DUMP_DIR=<dir>` alongside
the logging vars tells FlashInfer to write a Definition JSON for every
`@flashinfer_api(trace=...)`-decorated call (one file per unique (op, shape)).
**One SGLang run can produce both workload tensors and definition JSONs**,
which is the fastest way to pick up a shape that turned out to be missing from
the dataset (typical case: a new page-size variant or a quant-config variant).

```bash
export FLASHINFER_TRACE_DUMP=1
export FLASHINFER_TRACE_DUMP_DIR=tmp/dumps/fi_trace_{def_name}
# ...your existing FLASHINFER_LOGLEVEL/FLASHINFER_DUMP_* vars stay unchanged...
```

After the run, stage the new JSONs into the dataset with the snippet from
[`/extract-kernel-definitions` Path A3](../extract-kernel-definitions/SKILL.md#a3-dedupe-and-stage-into-the-dataset),
then normalize them for the validator with the
[A3b](../extract-kernel-definitions/SKILL.md#a3b-normalize-the-staged-jsons-for-flashinfer-bench-validate)
fix-up snippet.  The trace dump is independent of the logging API and adds
negligible overhead, so it's safe to leave on for any collection run — both
level-3 and level-10 paths benefit.

### Phase 3: SGLang Inference

**Inference source**: synthetic random prompts (default, `--dataset random`)
or real ShareGPT prompts (`--dataset sharegpt`).

- **`random`** (default): generates token-id prompts of a chosen length via
  `sample_random_requests` (ported from InferenceX
  `utils/bench_serving/benchmark_serving.py`).  Use when you need controlled
  prefill length and a guaranteed decode budget.  Each request decodes for
  exactly `--osl` tokens because `ignore_eos=True` is on by default.
  - Recommended pairs: `--isl 1024 --osl 1024` (decode-heavy, big-batch decode
    shapes) and `--isl 8192 --osl 1024` (prefill-heavy).
  - `--random-range-ratio` jitters lengths uniformly in `[ratio*len, len]`.
    Leave at `1.0` for exact lengths.
- **`sharegpt`**: real prompts from `anon8231489123/ShareGPT_Vicuna_unfiltered`.
  Length distribution is uncontrolled; use only when prompt realism matters
  more than coverage.

**Batch sizes**: `[8, 32, 64, 128]` — powers of 2 matching SGLang CUDA graph
capture points, run multiple rounds each for KV-length diversity.

**Dispatch**: sustained inflight (matches InferenceX `--request-rate inf`).
SGLang's async semaphore caps concurrent requests at `batch_size` and backfills
on each completion, so new prefills overlap with ongoing decodes — yielding
mixed prefill+decode batches and varied intra-batch kv_lens.

**Per-batch-size isolation** (`--restart-per-batch-size`): pass this flag to
`bench_serving.py` when using `FLASHINFER_DUMP_MAX_COUNT`.  Without it, the
first batch size exhausts the dump budget (DUMP_MAX_COUNT is a global counter
per server process) and later batch sizes capture nothing.  With it, each
batch size gets its own server session and therefore its own fresh counter.

Standard collection invocation with isolation:
```bash
FLASHINFER_DUMP_MAX_COUNT=500 \
FLASHINFER_DUMP_INCLUDE="BatchDecodeWithPagedKVCacheWrapper*" \
FLASHINFER_DUMP_EXCLUDE="*.__init__" \
... \
python3 examples/sglang_bench/bench_serving.py \
  --model <model-key> \
  --model-path /path/to/model \
  --dataset random --isl 1024 --osl 1024 \
  --batch-sizes 64 128 \
  --num-batches 4 \
  --restart-per-batch-size \
  --disable-cuda-graph
```

For prefill-heavy coverage, run a second pass with `--isl 8192 --osl 1024`.
For ShareGPT prompts pass `--dataset sharegpt` (no `--isl`/`--osl` needed).

**Three execution modes** (chosen automatically based on definition type):

| Mode | When | How |
|------|------|-----|
| SGLang offline Engine | Decode-only definitions | `engine.generate()` with exact batch size per call — guarantees decode sees `B` concurrent sequences |
| SGLang HTTP server (paged) | Paged-prefill definitions | Launches server with `--enable-deterministic-inference` to force `use_ragged=False`, sends prefix-sharing requests via `/v1/chat/completions` |
| SGLang HTTP server (ragged) | Ragged-prefill definitions (`BatchPrefillWithRaggedKVCacheWrapper`) | Launches server with `--disable-piecewise-cuda-graph` (no `--enable-deterministic-inference`), sends requests with `max_tokens=1` |

**Critical ragged prefill flags**: `--disable-cuda-graph` alone is insufficient.
SGLang always captures a piecewise CUDA graph for prefill; during capture
`is_in_piecewise_cuda_graph()=True` forces `use_ragged=False`, so the captured
graph only uses `BatchPrefillWithPagedKVCacheWrapper`.  Adding
`--disable-piecewise-cuda-graph` prevents the capture, ensuring every prefill
executes eagerly through `BatchPrefillWithRaggedKVCacheWrapper`.  Do **not**
add `--enable-deterministic-inference` for ragged — it forces `use_ragged=False`
entirely.

### Phase 4: Tensor Dump Sanitization

Hand off to [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md).  That skill
documents the `sanitize_dumps.py` invocation, plan/run pairing, tensor storage
policy, and dedupe rules.

### Phase 5: Baseline Evaluation

Runs the baseline solution against collected workloads before PR submission:
```bash
flashinfer-bench run --local {trace_dir} --definitions {def_name} --solutions baseline
# → writes {trace_dir}/traces/{def_name}_baseline.jsonl
```

All entries must have `evaluation.status == "PASSED"`.  If any fail, do not
submit PR 2.

### Phase 6: Submit PR 2 (HuggingFace flashinfer-trace)

One HuggingFace PR per definition.  PR 1 (GitHub flashinfer-bench) must already
be open.

**PR 2 contents:**
1. `solutions/baseline/{op_type}/{def_name}/flashinfer_wrapper_*.json` —
   FlashInfer API wrapper (calls `BatchDecodeWithPagedKVCacheWrapper` or
   `BatchPrefillWithPagedKVCacheWrapper`, **not** `reference_impl`).
2. `workloads/{op_type}/{def_name}.jsonl`.
3. `blob/workloads/{op_type}/{def_name}/*.safetensors`.
4. `definitions/{op_type}/{def_name}.json` (copied from PR 1).
5. `tests/references/test_{def_name}.py` (copied from PR 1).
6. `traces/{op_type}/{def_name}.jsonl` (baseline eval trace, all PASSED).

**PR description must include** the full stdout of `collect_workloads.py sglang`
under `## SGLang Collection Log`.  The log must show real ShareGPT inference
with diverse `(batch_size, kv_length)` pairs — uniform tiny KV caches (e.g.
`batch_size=4096` with 1-page contexts) indicate synthetic data, not real
inference.

## Error Handling

**No tensor dumps generated** (level 10): see
[`/sanitize-fi-dumps` § Common errors](../sanitize-fi-dumps/SKILL.md#common-errors).

**`run()` not captured**: look for `cudaErrorStreamCaptureUnsupported` in the
SGLang log.  Fix: always pass both `--disable-cuda-graph` and
`--disable-piecewise-cuda-graph`.

**Ragged prefill yields 0 workloads**: two possible causes —
1. `--enable-deterministic-inference` is set, which forces `use_ragged=False`
   globally — never set this for ragged definitions.
2. piecewise CUDA graph is active (default even without
   `--enable-deterministic-inference`), so `is_in_piecewise_cuda_graph()=True`
   during capture forces `use_ragged=False`, and the cached graph always routes
   to `BatchPrefillWithPagedKVCacheWrapper`.  Fix: add
   `--disable-piecewise-cuda-graph`.  The script auto-detects ragged prefill
   definitions and adds this flag automatically.

**Constant axis mismatch across TP**: use `--skip-const-axis-check` when
collecting TP=1 dumps for a TP=2 definition (structural tensors are identical
across TP).

**SGLang not wired to target FlashInfer API**: grep for the `fi_api` function
name in `tmp/sglang/python/sglang/srt/`.  If missing, the `onboard-model` skill
handles submitting a SGLang PR to wire it in.

**Level-3 `--strict` rejects every call for an API**: the trace template's
shape spec doesn't match runtime.  See
[`/sanitize-fi-log` § Triaging raw entries](../sanitize-fi-log/SKILL.md#triaging-raw-entries)
for the diagnostic matrix and the recurring template bugs to grep for in
`flashinfer/flashinfer/trace/templates/`.

## Prerequisites

Run `/clone-repos` first.  Then:
```bash
/clone-repos
/extract-kernel-definitions --model-name <model>
/collect-workloads --op-type <op_type> --model-path /path/to/model
```

For the level-3 default path you can skip `/clone-repos` if you already have a
log file and a folder of trace JSONs — go straight to `/sanitize-fi-log`.
