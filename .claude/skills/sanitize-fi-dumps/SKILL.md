---
name: sanitize-fi-dumps
description: Convert FlashInfer level-10 tensor dump directories into workload JSONL + safetensors blobs for the flashinfer-trace dataset. Use when you need real tensor values (correctness eval, replay) — for shape-only workloads from a level-3 log, use the sanitize-fi-log skill instead.
---

# Sanitize FlashInfer Level-10 Tensor Dumps into Workloads

Convert a FlashInfer level-10 dump directory (created when `FLASHINFER_LOGLEVEL=10`
runs the model) into per-definition workload JSONL files plus the safetensors
blobs that hold the structural integer tensors. This is the "real-tensor" path:
unlike the level-3 sanitization, structural tensors (indptrs, indices) are
preserved bit-for-bit so the resulting workloads can drive correctness eval and
replay. Float activations are still stored as `{"type": "random"}` because their
values don't affect kernel benchmark numerics.

## When to use this vs `sanitize-fi-log`

| Need | Skill |
|---|---|
| Correctness eval / replay against the *exact* tensors a real run saw. | **this skill** (level 10) |
| Shape-coverage workloads, no need for real values. | `sanitize-fi-log` (level 3) |
| Convert a level-3 log you already have. | `sanitize-fi-log` |
| Drive `flashinfer-bench run --solutions baseline` on the workload. | **this skill** — needs structural int tensors |

Default for `/collect-workloads`: level 3 + `sanitize-fi-log`. Level 10 + this
skill is opt-in (`--with-tensors` / `FLASHINFER_LOGLEVEL=10`) when the workflow
needs real tensors — typically when submitting workloads to the
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
HF dataset paired with baseline-eval traces.

## Inputs

- A FlashInfer dump directory produced by an SGLang (or any) run with
  `FLASHINFER_LOGLEVEL=10` and the standard dump env vars (see "Preparing the
  dump dir" below).
- A clone of the flashinfer-trace HF dataset (`tmp/flashinfer-trace/`) — workloads
  are written into `workloads/{op_type}/{def_name}.jsonl` and blobs into
  `blob/workloads/{op_type}/{def_name}/*.safetensors`.

## Command

```bash
python3 scripts/sanitize_dumps.py \
    --dump-dir              ./workload_dumps_<timestamp> \
    --definitions           gqa_paged_decode_h32_kv8_d128_ps1 [more ...] \
    --flashinfer-trace-dir  tmp/flashinfer-trace \
    [--op-type <type>] \
    [--replace] \
    [--skip-const-axis-check] \
    [--max-new-workloads N]
```

Flags:

- `--definitions` — explicit list of definition names to sanitize against.
- `--op-type` — alternative: process every definition under that op_type.
- `--replace` — overwrite the workloads JSONL instead of appending.
- `--skip-const-axis-check` — bypass const-axis verification (useful when collecting
  TP=1 dumps for a TP=2 definition: structural tensors are identical across TP
  even though head-count consts differ).
- `--max-new-workloads N` — cap on workloads added per run (used by the
  streaming `collect_stream.py` path: appends N diverse workloads per batch
  size, then the dump dir is wiped).

## What the script does

1. **Match dumps to definitions** via `fi_api` function name. Each definition's
   `tags` carry an `fi_api:<dotted.api.name>` marker; the dump's `metadata.jsonl`
   records the qualified function name. Suffix-match links them.
2. **Pair `plan()` dumps with their `run()` dump** (same PID, plan immediately
   precedes run). Plan provides the structural tensors that aren't in run's args:
   `paged_kv_indptr → kv_indptr`, `paged_kv_indices → kv_indices`,
   `paged_kv_last_page_len → kv_last_page_len`, etc.
3. **Apply tensor storage policy** per input:
   - `int32` / `int64` (structural: indptrs, indices) → saved to a per-call
     safetensors blob keyed by `tensor_key`.
   - float activations (`q`, `k_cache`, `v_cache`, etc.) → `{"type": "random"}` —
     shapes validated against the def, values irrelevant for benchmarking.
   - scalars (`sm_scale`, `eps`, etc.) → `{"type": "scalar", "value": <v>}`.
4. **Trim `kv_indices` to `kv_indptr[-1]`** (SGLang over-allocates the KV pool
   and the trailing slots are unused).
5. **Deduplicate** to at most 2 entries per unique axes combination.

## Preparing the dump dir (level-10 SGLang run)

Set these before launching SGLang. Critically, `FLASHINFER_LOGLEVEL=10` must be
exported **before** any FlashInfer import, otherwise the decorator no-ops and
no tensors are dumped.

```bash
export FLASHINFER_LOGLEVEL=10
export FLASHINFER_DUMP_DIR=./workload_dumps_$(date +%Y%m%d_%H%M%S)
export FLASHINFER_DUMP_SAFETENSORS=1
export FLASHINFER_DUMP_INCLUDE="<fi_api patterns>"   # e.g. "BatchDecodeWithPagedKVCacheWrapper*"
export FLASHINFER_DUMP_EXCLUDE="*.__init__"
export FLASHINFER_DUMP_MAX_COUNT=500    # ~4 batches × 16 layers × 8 TP ranks per session
export FLASHINFER_DUMP_MAX_SIZE_GB=30
```

`DUMP_MAX_COUNT` sizing: 500 covers ~4 full forward passes for TP=8, 16-layer
models (4 × 16 × 8 = 512 `run()` calls). Use 500 as the standard value when
collecting per-batch-size with `--restart-per-batch-size`.

`DUMP_INCLUDE` filtering: parse `fi_api:<dotted.api.name>` tags from each target
definition. For wrapper-class APIs (`BatchDecodeWithPagedKVCacheWrapper`),
include `.run` and `.plan` (the latter feeds the plan/run pairing); for free
functions, include the function name. Add `.forward` / `.forward_return_lse`
when the target is `BatchPrefillWithRaggedKVCacheWrapper` — SGLang calls those
shorthands rather than `.run`.

## Output format

```
{flashinfer_trace_dir}/workloads/{op_type}/{def_name}.jsonl
{flashinfer_trace_dir}/blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors
```

Each JSONL line:

```json
{
  "definition": "gqa_paged_decode_h32_kv8_d128_ps1",
  "workload": {
    "uuid": "a1b2c3d4-...",
    "axes": {"len_indptr": 33, "num_kv_indices": 4096},
    "inputs": {
      "q":                  {"type": "random"},
      "k_cache":            {"type": "random"},
      "v_cache":            {"type": "random"},
      "kv_indptr":          {"type": "safetensors", "path": "...", "tensor_key": "kv_indptr"},
      "kv_indices":         {"type": "safetensors", "path": "...", "tensor_key": "kv_indices"},
      "kv_last_page_len":   {"type": "safetensors", "path": "...", "tensor_key": "kv_last_page_len"},
      "sm_scale":           {"type": "scalar", "value": 0.0883}
    }
  }
}
```

## Common errors

- **No tensor dumps generated** — `FLASHINFER_LOGLEVEL=10` not exported before
  the FlashInfer import; or `FLASHINFER_DUMP_INCLUDE` doesn't match the actual
  qualified function name; or `--attention-backend flashinfer` wasn't passed to
  SGLang.
- **`run()` not captured but `plan()` was** — `cudaErrorStreamCaptureUnsupported`
  in SGLang log. Always pass both `--disable-cuda-graph` and
  `--disable-piecewise-cuda-graph` to SGLang when collecting.
- **Constant axis mismatch across TP** — pass `--skip-const-axis-check` when
  collecting TP=1 dumps to populate a TP=2 definition; structural tensors are
  identical across TP, head-count consts are not.

## Streaming collection

`scripts/collect_stream.py` (in `/collect-workloads`) wraps this skill in a
per-batch-size loop:

1. `bench_serving.py` with `DUMP_MAX_COUNT=500` (exhausted in round 1 of 2).
2. `sanitize_dumps.py --max-new-workloads 4` — appends 4 diverse workloads.
3. Incremental HF push: updated JSONL + new blobs (no deletes).
4. `rm -rf` dump dir.
5. Next batch size.

After all batch sizes: `flashinfer-bench run --solutions baseline` →
trace JSONL → push as eval trace.
